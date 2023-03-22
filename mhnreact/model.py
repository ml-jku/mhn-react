# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

Model related functionality
"""
from .utils import top_k_accuracy
from .plotutils import plot_loss, plot_topk, plot_nte
from .molutils import convert_smiles_to_fp
import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from scipy import sparse
import logging
from tqdm import tqdm
import wandb

log = logging.getLogger(__name__)

class ChemRXNDataset(torch.utils.data.Dataset):
    "Torch Dataset for ChemRXN containing Xs: the input as np array, target: the target molecules (or nothing), and ys: the label"
    def __init__(self, Xs, target, ys, is_smiles=False, fp_size=2048, fingerprint_type='morgan'):
        self.is_smiles=is_smiles
        if is_smiles:
            self.Xs = Xs
            self.target = target
            self.fp_size = fp_size
            self.fingerprint_type = fingerprint_type
        else:
            self.Xs = Xs.astype(np.float32)
            self.target = target.astype(np.float32)
        self.ys = ys
        self.ys_is_sparse = isinstance(self.ys, sparse.csr.csr_matrix)

    def __getitem__(self, k):
        mol_fp = self.Xs[k]
        if self.is_smiles:
            mol_fp = convert_smiles_to_fp(mol_fp, fp_size=self.fp_size, which=self.fingerprint_type).astype(np.float32)
        
        target = None if self.target is None else self.target[k]
        if self.is_smiles and self.target:
            target = convert_smiles_to_fp(target, fp_size=self.fp_size, which=self.fingerprint_type).astype(np.float32)
        
        label = self.ys[k]
        if isinstance(self.ys, sparse.csr.csr_matrix):
            label = label.toarray()[0]
            
        return (mol_fp, target, label)

    def __len__(self):
        return len(self.Xs)

class ModelConfig(object):
    def __init__(self, **kwargs):
        self.fingerprint_type = kwargs.pop("fingerprint_type", 'morgan')
        self.template_fp_type = kwargs.pop("template_fp_type", 'rdk')
        self.num_templates = kwargs.pop("num_templates", 401)
        self.fp_size = kwargs.pop("fp_size", 2048)
        self.fp_radius = kwargs.pop("fp_radius", 4)

        self.device = kwargs.pop("device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = kwargs.pop("batch_size", 32)
        self.pooling_operation_state_embedding = kwargs.pop('pooling_operation_state_embedding', 'mean')
        self.pooling_operation_head = kwargs.pop('pooling_operation_head', 'max')

        self.dropout = kwargs.pop('dropout', 0.0)

        self.lr = kwargs.pop('lr', 1e-4)
        self.optimizer = kwargs.pop("optimizer", "Adam")

        self.activation_function = kwargs.pop('activation_function', 'ReLU')
        self.verbose = kwargs.pop("verbose", False)  # debugging or printing additional warnings / information set tot True

        self.hopf_input_size = kwargs.pop('hopf_input_size', 2048)
        self.hopf_output_size = kwargs.pop("hopf_output_size", 768)
        self.hopf_num_heads = kwargs.pop("hopf_num_heads", 1)
        self.hopf_asso_dim = kwargs.pop("hopf_asso_dim", 768)
        self.hopf_association_activation = kwargs.pop("hopf_association_activation", None)
        self.hopf_beta = kwargs.pop("hopf_beta",0.125) #  1/(self.hopf_asso_dim**(1/2) sqrt(d_k)
        self.norm_input = kwargs.pop("norm_input",False)
        self.norm_asso = kwargs.pop("norm_asso", False)

        # additional experimental hyperparams
        if 'hopf_n_layers' in kwargs.keys():
            self.hopf_n_layers = kwargs.pop('hopf_n_layers', 0)
        if 'mol_encoder_layers' in kwargs.keys():
            self.mol_encoder_layers = kwargs.pop('mol_encoder_layers', 1)
        if 'temp_encoder_layers' in kwargs.keys():
            self.temp_encoder_layers = kwargs.pop('temp_encoder_layers', 1)
        if 'encoder_af' in kwargs.keys():
            self.encoder_af = kwargs.pop('encoder_af', 'ReLU')

        # additional kwargs
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                log.error(f"Can't set {key} with value {value} for {self}")
                raise err


class Encoder(nn.Module):
    """Simple FFNN"""
    def __init__(self, input_size: int = 2048, output_size: int = 1024,
                    num_layers: int = 1, dropout: float = 0.3, af_name: str ='None', 
                    norm_in: bool = False, norm_out: bool = False):
        super().__init__()
        self.ws = []
        self.setup_af(af_name)
        self.norm_in = (lambda k: k) if not norm_in else torch.nn.LayerNorm(input_size, elementwise_affine=False)
        self.norm_out = (lambda k: k) if not norm_out else torch.nn.LayerNorm(output_size, elementwise_affine=False)
        self.setup_ff(input_size, output_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        x = self.norm_in(x)
        for i, w in enumerate(self.ws):
            if i==(len(self.ws)-1):
                x = self.dropout(w(x)) # all except last haf ff_af
            else:
                x = self.dropout(self.af(w(x)))
        x = self.norm_out(x)
        return x

    def setup_ff(self, input_size:int, output_size:int, num_layers=1):
        """setup feed-forward NN with n-layers"""
        for n in range(0, num_layers):
            w = nn.Linear(input_size if n==0 else output_size, output_size)
            torch.nn.init.kaiming_normal_(w.weight, mode='fan_in', nonlinearity='linear') # eqiv to LeCun init
            setattr(self, f'W_{n}', w) # consider doing a step-wise reduction
            self.ws.append(getattr(self, f'W_{n}'))

    def setup_af(self, af_name : str):
        """set activation function"""
        if af_name is None or (af_name == 'None'):
          self.af = lambda k: k
        else:
          try:
              self.af = getattr(nn, af_name)()
          except AttributeError as err:
              log.error(f"Can't find activation-function {af_name} in torch.nn")
              raise err


class MoleculeEncoder(Encoder):
    """
    Class for Molecule encoder: can be any class mapping Smiles to a Vector (preferable differentiable ;)
    """
    def __init__(self, config):
        self.config = config

class FPMolEncoder(Encoder):
    """
    Fingerprint Based Molecular encoder
    """
    def __init__(self, config):
        super().__init__(input_size = config.hopf_input_size*config.hopf_num_heads,
                   output_size = config.hopf_asso_dim*config.hopf_num_heads,
                   num_layers = config.mol_encoder_layers,
                   dropout = config.dropout,
                   af_name = config.encoder_af,
                   norm_in = config.norm_input,
                   norm_out = config.norm_asso,
                  )
        # number of layers = self.config.mol_encoder_layers
        # layer-dimension = self.config.hopf_asso_dim
        # activation-function = self.config.af

        self.config = config

    def forward_smiles(self, list_of_smiles: list):
        fp_tensor = self.convert_smiles_to_tensor(list_of_smiles)
        return self.forward(fp_tensor)

    def convert_smiles_to_tensor(self, list_of_smiles):
        fps = convert_smiles_to_fp(list_of_smiles, fp_size=self.config.fp_size, 
                                   which=self.config.fingerprint_type, radius=self.config.fp_radius)
        fps_tensor = torch.from_numpy(fps.astype(np.float32)).to(dtype=torch.float).to(self.config.device)
        return fps_tensor

class TemplateEncoder(Encoder):
    """
    Class for Template encoder: can be any class mapping a Smarts-Reaction to a Vector (preferable differentiable ;)
    """
    def __init__(self, config):
        super().__init__(input_size = config.hopf_input_size*config.hopf_num_heads,
                   output_size = config.hopf_asso_dim*config.hopf_num_heads,
                   num_layers = config.temp_encoder_layers,
                   dropout = config.dropout,
                   af_name = config.encoder_af,
                   norm_in = config.norm_input,
                   norm_out = config.norm_asso,
                  )
        self.config = config
        #number of layers
        #template fingerprint type
        #random template threshold
        #reactant pooling
        if config.temp_encoder_layers==0:
            print('No Key-Projection = Static Key/Templates')
            assert self.config.hopf_asso_dim==self.config.fp_size
            self.wks = []


class MHN(nn.Module):
    """
    MHN - modern Hopfield Network -- for Template relevance prediction
    """
    def __init__(self, config=None, layer2weight=0.05, use_template_encoder=True):
        super().__init__()
        if config:
            self.config = config
        else:
            self.config = ModelConfig()
        self.beta = self.config.hopf_beta
        # hopf_num_heads
        self.mol_encoder = FPMolEncoder(self.config)
        if use_template_encoder:
            self.template_encoder = TemplateEncoder(self.config)

        self.W_v = None
        self.layer2weight = layer2weight

        # more MHN layers -- added recursively
        if hasattr(self.config, 'hopf_n_layers'):
            di = self.config.__dict__
            di['hopf_n_layers'] -= 1
            if di['hopf_n_layers']>0:
                conf_wo_hopf_nlayers = ModelConfig(**di)
                self.layer = MHN(conf_wo_hopf_nlayers)
                if di['hopf_n_layers']!=0:
                    self.W_v = nn.Linear(self.config.hopf_asso_dim, self.config.hopf_input_size)
                    torch.nn.init.kaiming_normal_(self.W_v.weight, mode='fan_in', nonlinearity='linear') # eqiv to LeCun init

        self.softmax = torch.nn.Softmax(dim=1)

        self.lossfunction = nn.CrossEntropyLoss(reduction='none')#, weight=class_weights)
        self.pretrain_lossfunction = nn.BCEWithLogitsLoss(reduction='none')#, weight=class_weights)

        self.lr = self.config.lr

        if self.config.hopf_association_activation is None or (self.config.hopf_association_activation.lower()=='none'):
            self.af = lambda k: k
        else:
            self.af = getattr(nn, self.config.hopf_association_activation)()

        self.pooling_operation_head = getattr(torch, self.config.pooling_operation_head)

        self.X = None # templates projected to Hopfield Layer

        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.parameters(), lr=self.lr)
        self.steps = 0
        self.hist = defaultdict(list)
        self.to(self.config.device)

    def set_templates(self, template_list, which='rdk', fp_size=None, radius=2, learnable=False, njobs=1, only_templates_in_batch=False):
        self.template_list = template_list.copy()
        if fp_size is None:
            fp_size = self.config.fp_size
        if len(template_list)>=100000:
            import math
            print('batch-wise template_calculation')
            bs = 30000
            final_temp_emb = torch.zeros((len(template_list), fp_size)).float().to(self.config.device)
            for b in range(math.ceil(len(template_list)//bs)+1):
                self.template_list = template_list[bs*b:min(bs*(b+1), len(template_list))]
                templ_emb = self.update_template_embedding(which=which, fp_size=fp_size, radius=radius, learnable=learnable, njobs=njobs, only_templates_in_batch=only_templates_in_batch)
                final_temp_emb[bs*b:min(bs*(b+1), len(template_list))] = torch.from_numpy(templ_emb)
            self.templates = final_temp_emb
        else:
            self.update_template_embedding(which=which, fp_size=fp_size, radius=radius, learnable=learnable, njobs=njobs, only_templates_in_batch=only_templates_in_batch)
        
        self.set_templates_recursively()

    def set_templates_recursively(self):
        if 'hopf_n_layers' in self.config.__dict__.keys():
            if self.config.hopf_n_layers >0:
                self.layer.templates = self.templates
                self.layer.set_templates_recursively()

    def update_template_embedding(self,fp_size=2048, radius=4, which='rdk', learnable=False, njobs=1, only_templates_in_batch=False):
        print('updating template-embedding; (just computing the template-fingerprint and using that)')
        bs = self.config.batch_size

        split_template_list = [str(t).split('>')[0].split('.') for t in self.template_list]
        templates_np = convert_smiles_to_fp(split_template_list, is_smarts=True, fp_size=fp_size, radius=radius, which=which, njobs=njobs)

        split_template_list = [str(t).split('>')[-1].split('.') for t in self.template_list]
        reactants_np = convert_smiles_to_fp(split_template_list, is_smarts=True, fp_size=fp_size, radius=radius, which=which, njobs=njobs)

        template_representation = templates_np-(reactants_np*0.5)
        if learnable:
            self.templates = torch.nn.Parameter(torch.from_numpy(template_representation).float(), requires_grad=True).to(self.config.device)
            self.register_parameter(name='templates', param=self.templates)
        else:
            if only_templates_in_batch:
                self.templates_np = template_representation
            else:
                self.templates = torch.from_numpy(template_representation).float().to(self.config.device)
                
        return template_representation


    def np_fp_to_tensor(self, np_fp):
        return torch.from_numpy(np_fp.astype(np.float64)).to(self.config.device).float()

    def masked_loss_fun(self, loss_fun, h_out, ys_batch):
        if loss_fun == self.BCEWithLogitsLoss:
            mask = (ys_batch != -1).float()
            ys_batch = ys_batch.float()
        else:
            mask = (ys_batch.long() != -1).long()
        mask_sum = int(mask.sum().cpu().numpy())
        if mask_sum == 0:
            return 0

        ys_batch = ys_batch * mask

        loss = (loss_fun(h_out, ys_batch * mask) * mask.float()).sum() / mask_sum  # only mean from non -1
        return loss

    def compute_losses(self, out, ys_batch, head_loss_weight=None):

        if len(ys_batch.shape)==2:
            if ys_batch.shape[1]==self.config.num_templates: # it is in pretraining_mode
                loss = self.pretrain_lossfunction(out, ys_batch.float()).mean()
            else:
                # legacy from policyNN
                loss = self.lossfunction(out, ys_batch[:, 2]).mean()  # WARNING: HEAD4 Reaction Template is ys[:,2]
        else:
            loss = self.lossfunction(out, ys_batch).mean() 
        return loss

    def forward_smiles(self, list_of_smiles, templates=None):
        state_tensor = self.mol_encoder.convert_smiles_to_tensor(list_of_smiles)
        return self.forward(state_tensor, templates=templates)

    def forward(self, m, templates=None):
        """
        m: molecule in the form batch x fingerprint
        templates: None or newly given templates if not instanciated
        returns logits ranking the templates for each molecule
        """
        #states_emb = self.fcfe(state_fp)
        bs = m.shape[0] #batch_size
        #templates = self.temp_emb(torch.arange(0,2000).long())
        if (templates is None) and (self.X is None) and (self.templates is None):
            raise Exception('Either pass in templates, or init templates by runnting clf.set_templates')
        n_temp = len(templates) if templates is not None else len(self.templates)
        if self.training or (templates is None) or (self.X is not None):
            templates = templates if templates is not None else self.templates
            X = self.template_encoder(templates)
        else:
            X = self.X # precomputed from last forward run
        
        Xi = self.mol_encoder(m)

        Xi = Xi.view(bs, self.config.hopf_num_heads, self.config.hopf_asso_dim) # [bs, H, A]
        X = X.view(1, n_temp, self.config.hopf_asso_dim, self.config.hopf_num_heads) #[1, T, A, H]

        XXi = torch.tensordot(Xi, X, dims=[(2,1), (2,0)]) # AxA -> [bs, T, H]
        
        # pooling over heads
        if self.config.hopf_num_heads<=1:
            #QKt_pooled = QKt
            XXi = XXi[:,:,0] #torch.squeeze(QKt, dim=2)
        else:
            XXi = self.pooling_operation_head(XXi, dim=2) # default is max pooling over H [bs, T]
            if (self.config.pooling_operation_head =='max') or (self.config.pooling_operation_head =='min'):
                XXi = XXi[0] #max and min also return the indices =S

        out = self.beta*XXi # [bs, T, H] # softmax over dim=1 #pooling_operation_head
        
        self.xinew = self.softmax(out)@X.view(n_temp, self.config.hopf_asso_dim) # [bs,T]@[T,emb] -> [bs,emb]

        if self.W_v:
            # call layers recursive
            hopfout = self.W_v(self.xinew) # [bs,emb]@[emb,hopf_inp]  --> [bs, hopf_inp]
            # TODO check if using x_pooled or if not going through mol_encoder again
            hopfout = hopfout + m # skip-connection
            # give it to the next layer
            out2 = self.layer.forward(hopfout) #templates=self.W_v(self.K)
            out = out*(1-self.layer2weight)+out2*self.layer2weight

        return out

    def train_from_np(self, Xs, targets, ys, is_smiles=False, epochs=2, lr=0.001, bs=32,
                      permute_batches=False, shuffle=True, optimizer=None,
                      use_dataloader=True, verbose=False,
                      wandb=None, scheduler=None, only_templates_in_batch=False):
        """
        Xs in the form sample x states
        targets
        ys in the form sample x [y_h1, y_h2, y_h3, y_h4]
        """
        self.train()
        if optimizer is None:
            try:
                self.optimizer = getattr(torch.optim, self.config.optimizer)(self.parameters(), lr=self.lr if lr is None else lr)
            except AttributeError as err:
                log.error(f"Can't find optimizer {config.optimizer} in torch.optim")
                raise err
            optimizer = self.optimizer

        dataset = ChemRXNDataset(Xs, targets, ys, is_smiles=is_smiles,
                                 fp_size=self.config.fp_size, fingerprint_type=self.config.fingerprint_type)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, sampler=None,
                   batch_sampler=None, num_workers=0, collate_fn=None,
                   pin_memory=False, drop_last=False, timeout=0,
                   worker_init_fn=None)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            running_loss_dict = defaultdict(int)
            batch_order = range(0, len(Xs), bs)
            if permute_batches:
                batch_order = np.random.permutation(batch_order)

            for step, s in tqdm(enumerate(dataloader),mininterval=2):
                batch = [b.to(self.config.device, non_blocking=True) for b in s]
                Xs_batch, target_batch, ys_batch = batch

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                out = self.forward(Xs_batch)
                total_loss = self.compute_losses(out, ys_batch)

                loss_dict = {'CE_loss': total_loss}

                total_loss.backward()

                optimizer.step()
                if scheduler:
                    scheduler.step()
                self.steps += 1

                # print statistics
                for k in loss_dict:
                    running_loss_dict[k] += loss_dict[k].item()
                try:
                    running_loss += total_loss.item()
                except:
                    running_loss += 0

                rs = min(100,len(Xs)//bs) # reporting/logging steps
                if step % rs == (rs-1):  # print every 2000 mini-batches
                    if verbose: print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, step + 1, running_loss / rs))
                    self.hist['step'].append(self.steps)
                    self.hist['loss'].append(running_loss/rs)
                    self.hist['trianing_running_loss'].append(running_loss/rs)

                    [self.hist[k].append(running_loss_dict[k]/rs) for k in running_loss_dict]

                    if wandb:
                        wandb.log({'trianing_running_loss': running_loss / rs})

                    running_loss = 0.0
                    running_loss_dict = defaultdict(int)

        if verbose: print('Finished Training')
        return optimizer

    def evaluate(self, Xs, targets, ys, split='test', is_smiles=False, bs = 32, shuffle=False, wandb=None, only_loss=False):
        self.eval()
        y_preds = np.zeros( (ys.shape[0], self.config.num_templates), dtype=np.float16)
        
        loss_metrics = defaultdict(int)
        new_hist = defaultdict(float)
        with torch.no_grad():
            dataset = ChemRXNDataset(Xs, targets, ys, is_smiles=is_smiles,
                                     fp_size=self.config.fp_size, fingerprint_type=self.config.fingerprint_type)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=shuffle, sampler=None,
                       batch_sampler=None, num_workers=0, collate_fn=None,
                       pin_memory=False, drop_last=False, timeout=0,
                       worker_init_fn=None)

            #for step, s in eoutputs = self.forward(batch[0], batchnumerate(range(0, len(Xs), bs)):
            for step, batch in enumerate(dataloader):#
                batch = [b.to(self.config.device, non_blocking=True) for b in batch]
                ys_batch = batch[2]
                
                if hasattr(self, 'templates_np'):
                    outputs = []
                    for ii in range(10):
                        tlen = len(self.templates_np)
                        i_tlen = tlen//10
                        templates = torch.from_numpy(self.templates_np[(i_tlen*ii):min(i_tlen*(ii+1), tlen)]).float().to(self.config.device)
                        outputs.append( self.forward(batch[0], templates = templates ) )
                    outputs = torch.cat(outputs, dim=0)
                        
                else:
                    outputs = self.forward(batch[0]) 

                loss = self.compute_losses(outputs, ys_batch, None)

                # not quite right because in every batch there might be different number of valid samples
                weight = 1/len(batch[0])#len(Xs[s:min(s + bs, len(Xs))]) / len(Xs)

                loss_metrics['loss'] += (loss.item())

                if len(ys.shape)>1:
                    outputs = self.softmax(outputs) if not (ys.shape[1]==self.config.num_templates) else torch.sigmoid(outputs)
                else:
                    outputs = self.softmax(outputs)

                outputs_np = [None if o is None else o.to('cpu').numpy().astype(np.float16) for o in outputs]

                if not only_loss:
                    ks = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
                    topkacc, mrocc = top_k_accuracy(ys_batch, outputs, k=ks, ret_arocc=True, ret_mrocc=False)
                    # mrocc -- median rank of correct choice
                    for k, tkacc in zip(ks, topkacc):
                        #iterative average update
                        new_hist[f't{k}_acc_{split}'] += (tkacc-new_hist[f't{k}_acc_{split}']) / (step+1)
                        # todo weight by batch-size
                    new_hist[f'meanrank_{split}'] = mrocc

                y_preds[step*bs : min((step+1)*bs,len(y_preds))] = outputs_np


        new_hist[f'steps_{split}'] = (self.steps)
        new_hist[f'loss_{split}'] = (loss_metrics['loss'] / (step+1))

        for k in new_hist:
            self.hist[k].append(new_hist[k])

        if wandb:
            wandb.log(new_hist)


        self.hist[f'loss_{split}'].append(loss_metrics[f'loss'] / (step+1))

        return y_preds

    def save_hist(self, prefix='', postfix=''):
        HIST_PATH = 'data/hist/'
        if not os.path.exists(HIST_PATH):
            os.mkdir(HIST_PATH)
        fn_hist = HIST_PATH+prefix+postfix+'.csv'
        with open(fn_hist, 'w') as fh:
            print(dict(self.hist), file=fh)
        return fn_hist

    def save_model(self, prefix='', postfix='', name_as_conf=False):
        MODEL_PATH = 'data/model/'
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        if name_as_conf:
            confi_str = str(self.config.__dict__.values()).replace("'","").replace(': ','_').replace(', ',';')
        else:
            confi_str = ''
        model_name = prefix+confi_str+postfix+'.pt'
        torch.save(self.state_dict(), MODEL_PATH+model_name)
        return MODEL_PATH+model_name

    def plot_loss(self):
        plot_loss(self.hist)

    def plot_topk(self, sets=['train', 'valid', 'test'], with_last = 2):
        plot_topk(self.hist, sets=sets, with_last = with_last)

    def plot_nte(self, last_cpt=1, dataset='Sm', include_bar=True):
        plot_nte(self.hist, dataset=dataset, last_cpt=last_cpt, include_bar=include_bar)


class SeglerBaseline(MHN):
    """FFNN - only the Molecule Encoder + an output projection"""
    def __init__(self, config=None):
        config.template_fp_type = 'none'
        config.temp_encoder_layers = 0
        super().__init__(config, use_template_encoder=False)
        self.W_out = torch.nn.Linear(config.hopf_asso_dim, config.num_templates)
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.parameters(), lr=self.lr)
        self.steps = 0
        self.hist = defaultdict(list)
        self.to(self.config.device)

    def forward(self, m, templates=None):
        """
        m: molecule in the form batch x fingerprint
        templates: won't be used in this case
        returns logits ranking the templates for each molecule
        """
        bs = m.shape[0] #batch_size
        Xi = self.mol_encoder(m)
        Xi = self.mol_encoder.af(Xi) # is not applied in encoder for last layer
        out = self.W_out(Xi) # [bs, T] # softmax over dim=1
        return out

class StaticQK(MHN):
    """ Static QK baseline - beware to have the same fingerprint for mol_encoder as for the template_encoder (fp2048 r4 rdk by default)"""
    def __init__(self, config=None):
        if config:
            self.config = config
        else:
            self.config = ModelConfig()
        super().__init__(config)

        self.fp_size = 2048
        self.fingerprint_type = 'rdk'
        self.beta = 1

    def update_template_embedding(self, which='rdk', fp_size=2048, radius=4, learnable=False):
        bs = self.config.batch_size
        split_template_list = [t.split('>>')[0].split('.') for t in self.template_list]
        self.templates = torch.from_numpy(convert_smiles_to_fp(split_template_list, 
                                                               is_smarts=True, fp_size=fp_size, 
                                                               radius=radius, which=which).max(1)).float().to(self.config.device)


    def forward(self, m, templates=None):
        """

        """
        #states_emb = self.fcfe(state_fp)
        bs = m.shape[0] #batch_size
        
        Xi = m #[bs, emb]
        X = self.templates #[T, emb])

        XXi = Xi@X.T # [bs, T]
        
        # normalize
        t_sum = templates.sum(1) #[T]
        t_sum = t_sum.view(1,-1).expand(bs, -1) #[bs, T]
        XXi = XXi / t_sum
        
        # not neccecaire because it is not trained
        out = self.beta*XXi # [bs, T] # softmax over dim=1
        return out

class Retrosim(StaticQK):
    """ Retrosim-like baseline only for template relevance prediction """
    def fit_with_train(self, X_fp_train, y_train):
        self.templates = torch.from_numpy(X_fp_train).float().to(self.config.device)
        # train_samples, num_templates
        self.sample2acttemplate = torch.nn.functional.one_hot(torch.from_numpy(y_train), self.config.num_templates).float()
        tmpnorm = self.sample2acttemplate.sum(0)
        tmpnorm[tmpnorm==0] = 1
        self.sample2acttemplate = (self.sample2acttemplate / tmpnorm).to(self.config.device) # results in an average after dot product

    def forward(self, m, templates=None):
        """
        """
        out = super().forward(m, templates=templates)
        # bs, train_samples

        # map out to actual templates
        out = out @ self.sample2acttemplate

        return out