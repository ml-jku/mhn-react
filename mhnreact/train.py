# -*- coding: utf-8 -*-
"""
Author: Philipp Seidl
        ELLIS Unit Linz, LIT AI Lab, Institute for Machine Learning
        Johannes Kepler University Linz
Contact: seidl@ml.jku.at

Training
"""

from .utils import str2bool, lgamma, multinom_gk, top_k_accuracy
from .data import load_templates, load_dataset_from_csv, load_USPTO
from .model import ModelConfig, MHN, StaticQK, SeglerBaseline, Retrosim
from .molutils import convert_smiles_to_fp, FP_featurizer, smarts2appl, getTemplateFingerprint, disable_rdkit_logging
from collections import defaultdict
import argparse
import os
import numpy as np
import pandas as pd
import datetime
import sys
from time import time
import matplotlib.pyplot as plt
import torch
import multiprocessing
import warnings
from joblib import Memory

cachedir = 'data/cache/'
memory = Memory(cachedir, verbose=0, bytes_limit=80e9)

def parse_args():
    parser = argparse.ArgumentParser(description="Train MHNreact.",
                                     epilog="--", prog="Train")
    parser.add_argument('-f', type=str)
    parser.add_argument('--model_type', type=str, default='mhn', 
                        help="Model-type: choose from 'segler', 'fortunato', 'mhn' or 'staticQK', default:'mhn'")
    parser.add_argument("--exp_name", type=str, default='', help="experiment name, (added as postfix to the file-names)")
    parser.add_argument("-d", "--dataset_type", type=str, default='sm', 
                        help="Input Dataset 'sm' for Scheider-USPTO-50k 'lg' for USPTO large or 'golden' or use keyword '--csv_path to specify an input file', default: 'sm'")
    parser.add_argument("--csv_path", default=None, type=str, help="path to preprocessed trainings file + split columns, default: None")
    parser.add_argument("--split_col", default='split', type=str, help="split column of csv, default: 'split'")
    parser.add_argument("--input_col", default='prod_smiles', type=str, help="input column of csv, default: 'pro_smiles'")
    parser.add_argument("--reactants_col", default='reactants_can', type=str, help="reactant colum of csv, default: 'reactants_can'")
    
    parser.add_argument("--fp_type", type=str, default='morganc',
                        help="Fingerprint type for the input only!: default: 'morgan', other options: 'rdk', 'ECFP', 'ECFC', 'MxFP', 'Morgan2CBF' or a combination of fingerprints with '+'' for max-pooling and '&' for concatination e.g. maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+layered+mhfp, default: 'morganc'")
    parser.add_argument("--template_fp_type", type=str, default='rdk', 
                        help="Fingerprint type for the template fingerprint, default: 'rdk'")
    parser.add_argument("--device", type=str, default='best', 
                        help="Device to run the model on, preferably 'cuda:0', default: 'best' (takes the gpu with most RAM)")
    parser.add_argument("--fp_size", type=int, default=4096, 
                        help="fingerprint-size used for templates as well as for inputs, default: 4096")
    parser.add_argument("--fp_radius", type=int, default=2, help="fingerprint-radius (if applicable to the fingerprint-type), default: 2")
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs, default: 10')

    parser.add_argument("--pretrain_epochs", type=int, default=0,
                        help="applicability-matrix pretraining epochs if applicable (e.g. fortunato model_type), default: 0")
    parser.add_argument("--save_model", type=str2bool, default=False, help="save the model, default: False")

    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate for encoders, default: 0.2")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning-rate, dfeault: 5e-4")
    parser.add_argument("--hopf_beta", type=float, default=0.05, help="hopfield beta parameter, default: 0.125")
    parser.add_argument("--hopf_asso_dim", type=int, default=512, help="association dimension, default: 512")
    parser.add_argument("--hopf_num_heads", type=int, default=1, help="hopfield number of heads, default: 1")
    parser.add_argument("--hopf_association_activation", type=str, default='None',
                        help="hopfield association activation function recommended:'Tanh' or 'None', other: 'ReLU', 'SeLU', 'GeLU', or 'None' for more, see torch.nn, default: 'None'")

    parser.add_argument("--norm_input", default=True, type=str2bool, 
                        help="input-normalization, default: True")
    parser.add_argument("--norm_asso", default=True, type=str2bool, 
                        help="association-normalization, default: True")

    # additional experimental hyperparams
    parser.add_argument("--hopf_n_layers", default=1, type=int, help="Number of hopfield-layers, default: 1")
    parser.add_argument("--mol_encoder_layers", default=1, type=int, help="Number of molecule-encoder layers, default: 1")
    parser.add_argument("--temp_encoder_layers", default=1, type=int, help="Number of template-encoder layers, default: 1")
    parser.add_argument("--encoder_af", default='ReLU', type=str,
                        help="Encoder-NN intermediate activation function (before association_activation function), default: 'ReLU'")
    parser.add_argument("--hopf_pooling_operation_head", default='mean', type=str, help="Pooling operation over heads default=max, (max, min, mean, ...), default: 'mean'")

    parser.add_argument("--splitting_scheme", default=None, type=str, help="Splitting_scheme for non-csv-input, default: None, other options: 'class-freq', 'random'")

    parser.add_argument("--concat_rand_template_thresh", default=-1, type=int, help="Concatinates a random vector to the tempalte-fingerprint at all templates with num_training samples > this threshold; -1 (default) means deactivated")
    parser.add_argument("--repl_quotient", default=10, type=float, help="Only if --concat_rand_template_thresh >= 0 - Quotient of how much should be replaced by random in template-embedding, (default: 10)")
    parser.add_argument("--verbose", default=False, type=str2bool, help="If verbose, will print out more stuff, default: False")
    parser.add_argument("--batch_size", default=128, type=int, help="Training batch-size, default: 128")
    parser.add_argument("--eval_every_n_epochs", default=1, type=int, help="Evaluate every _ epochs (Evaluation is costly for USPTO-Lg), default: 1")
    parser.add_argument("--save_preds", default=False, type=str2bool, help="Save predictions for test split at the end of training, default: False")
    parser.add_argument("--wandb", default=False, type=str2bool, help="Save to wandb; login required, default: False")
    parser.add_argument("--seed", default=None, type=int, help="Seed your run to make it reproducible, defualt: None")

    parser.add_argument("--template_fp_type2", default=None, type=str, help="experimental template_fp_type for layer 2, default: None")
    parser.add_argument("--layer2weight",default=0.2, type=float, help="hopf-layer2 weight of p, default: 0.2")

    parser.add_argument("--reactant_pooling", default='max', type=str, help="reactant pooling operation over template-fingerprint, default: 'max', options: 'min','mean','lgamma'")


    parser.add_argument("--ssretroeval", default=False, type=str2bool, help="single-step retro-synthesis eval, default: False")
    parser.add_argument("--addval2train", default=False, type=str2bool, help="adds the validation set to the training set, default: False")
    parser.add_argument("--njobs",default=-1, type=int, help="Number of jobs, default: -1 -> uses all available")
    
    parser.add_argument("--eval_only_loss", default=False, type=str2bool, help="if only loss should be evaluated (if top-k acc may be time consuming), default: False")
    parser.add_argument("--only_templates_in_batch", default=False, type=str2bool, help="while training only forwards templates that are in the batch, default: False")
    
    parser.add_argument("--plot_res", default=False, type=str2bool, help="Plotting results for USPTO-sm/lg, default: False")
    args = parser.parse_args()
    
    if args.njobs ==-1:
        args.njobs = int(multiprocessing.cpu_count())
        
    if args.device=='best':
        from .utils import get_best_gpu
        try:
            args.device = get_best_gpu()
        except:
            print('couldnt get the best gpu, using cpu instead')
            args.device = 'cpu'
    
    # some save checks on model type
    if (args.model_type == 'segler') & (args.pretrain_epochs>=1):
        print('changing model type to fortunato because of pretraining_epochs>0')
        args.model_type = 'fortunato'
    if ((args.model_type == 'staticQK') or (args.model_type == 'retrosim')) & (args.epochs>1):
        print('changing epochs to 1 (StaticQK is not lernable ;)')
        args.epochs=1
        if args.template_fp_type != args.fp_type:
            print('fp_type must be the same as template_fp_type --> setting template_fp_type to fp_type')
            args.template_fp_type = args.fp_type
    if args.save_model & (args.fp_type=='MxFP'):
        warnings.warn('Currently MxFP is not recommended for saving the model paprameter (fragment dict for others would need to be saved or compued again, currently not implemented)')
    
    return args

@memory.cache(ignore=['njobs'])
def featurize_smiles(X, fp_type='morgan', fp_size=4096, fp_radius=2, njobs=1, verbose=False):
    X_fp = {}
    
    if fp_type in ['MxFP','MACCS','Morgan2CBF','Morgan4CBF', 'Morgan6CBF', 'ErG','AtomPair','TopologicalTorsion','RDK']:
        print('computing', fp_type)
        if fp_type == 'MxFP':
            fp_types = ['MACCS','Morgan2CBF','Morgan4CBF', 'Morgan6CBF', 'ErG','AtomPair','TopologicalTorsion','RDK']
        else:
            fp_types = [fp_type]

        remaining = int(fp_size)
        for fp_type in fp_types:
            print(fp_type,end=' ')
            feat = FP_featurizer(fp_types=fp_type,
                                 max_features= (fp_size//len(fp_types)) if (fp_type != fp_types[-1]) else remaining )
            X_fp[f'train_{fp_type}'] = feat.fit(X['train'])
            X_fp[f'valid_{fp_type}'] = feat.transform(X['valid'])
            X_fp[f'test_{fp_type}'] = feat.transform(X['test'])

            remaining -= X_fp[f'train_{fp_type}'].shape[1]
            #X_fp['train'].shape, X_fp['test'].shape
        X_fp['train'] = np.hstack([ X_fp[f'train_{fp_type}'] for fp_type in fp_types])
        X_fp['valid'] = np.hstack([ X_fp[f'valid_{fp_type}'] for fp_type in fp_types])
        X_fp['test'] = np.hstack([ X_fp[f'test_{fp_type}'] for fp_type in fp_types])
    
    else: #fp_type in ['rdk','morgan','ecfp4','pattern','morganc','rdkc']:
        if verbose: print('computing', fp_type, 'folded')
        for split in X.keys():
            X_fp[split] = convert_smiles_to_fp(X[split], fp_size=fp_size, which=fp_type, radius=fp_radius, njobs=njobs, verbose=verbose)

    return X_fp


def compute_template_fp(fp_len=2048, reactant_pooling='max', do_log=True):
    """Pre-Compute the template-fingerprint"""
    # combine them to one fingerprint
    comb_template_fp = np.zeros((max(template_list.keys())+1,fp_len if reactant_pooling!='concat' else fp_len*6))
    for i in template_list:
        tpl = template_list[i]
        try:
            pr, rea = str(tpl).split('>>')
            idxx = temp_part_to_fp[pr]
            prod_fp = templates_fp['fp'][idxx]
        except:
            print('err', pr, end='\r')
            prod_fp = np.zeros(fp_len)

        rea_fp = templates_fp['fp'][[temp_part_to_fp[r] for r in str(rea).split('.')]] # max-pooling

        if reactant_pooling=='only_product':
            rea_fp = np.zeros(fp_len)
        if reactant_pooling=='max':
            rea_fp = np.log(1 + rea_fp.max(0))
        elif reactant_pooling=='mean':
            rea_fp = np.log(1 + rea_fp.mean(0))
        elif reactant_pooling=='sum':
            rea_fp = np.log(1 + rea_fp.mean(0))
        elif reactant_pooling=='lgamma':
            rea_fp = multinom_gk(rea_fp, axis=0)
        elif reactant_pooling=='concat':
            rs = str(rea).split('.')
            rs.sort()
            for ii, r in enumerate(rs):
                idx = temp_part_to_fp[r]
                rea_fp = templates_fp['fp'][idx]
                comb_template_fp[i, (fp_len*(ii+1)):(fp_len*(ii+2))] = np.log(1 + rea_fp)
        
        comb_template_fp[i,:prod_fp.shape[0]] = np.log(1 + prod_fp) #- rea_fp*0.5
        if reactant_pooling!='concat':
            #comb_template_fp[i] = multinom_gk(np.stack([np.log(1+prod_fp), rea_fp]))
            #comb_template_fp[i,fp_len:] = rea_fp
            comb_template_fp[i,:rea_fp.shape[0]] = comb_template_fp[i, :rea_fp.shape[0]] - rea_fp*0.5
            
    return comb_template_fp


def set_up_model(args, template_list=None):
    hpn_config = ModelConfig(num_templates = int(max(template_list.keys()))+1,
                             #len(template_list.values()),  #env.num_templates, #
                             dropout=args.dropout,
                             fingerprint_type=args.fp_type,
                             template_fp_type = args.template_fp_type,
                             fp_size = args.fp_size,
                             fp_radius= args.fp_radius,
                             device=args.device,
                             lr=args.lr,
                             hopf_beta=args.hopf_beta,  #1/(128**0.5),#1/(2048**0.5),
                             hopf_input_size=args.fp_size,
                             hopf_output_size=None,
                             hopf_num_heads=args.hopf_num_heads,
                             hopf_asso_dim=args.hopf_asso_dim,

                             hopf_association_activation = args.hopf_association_activation,  #or ReLU, Tanh works better, SELU, GELU
                             norm_input = args.norm_input,
                             norm_asso = args.norm_asso,
                             
                             hopf_n_layers= args.hopf_n_layers,
                             mol_encoder_layers=args.mol_encoder_layers,
                             temp_encoder_layers=args.temp_encoder_layers,
                             encoder_af=args.encoder_af,
                             
                             hopf_pooling_operation_head = args.hopf_pooling_operation_head,
                             batch_size=args.batch_size,
                             )
    print(hpn_config.__dict__)

    if args.model_type=='segler': # baseline
        clf = SeglerBaseline(hpn_config)
    elif args.model_type=='mhn':
        clf = MHN(hpn_config, layer2weight=args.layer2weight)
    elif args.model_type=='fortunato': # pretraining with applicability-matrix
        clf = SeglerBaseline(hpn_config)
    elif args.model_type=='staticQK': # staticQK
        clf = StaticQK(hpn_config)
    elif args.model_type=='retrosim': # staticQK
        clf = Retrosim(hpn_config)
    else:
        raise NotImplementedError
        
    return clf, hpn_config

def set_up_template_encoder(args, clf, label_to_n_train_samples=None, template_list=None):

    if isinstance(clf, SeglerBaseline):
        clf.templates = []
    elif args.model_type=='staticQK':
        clf.template_list = list(template_list.values())
        clf.update_template_embedding(which=args.template_fp_type, fp_size=args.fp_size, radius=args.fp_radius, njobs=args.njobs)
    elif args.model_type=='retrosim':
        #clf.template_list = list(X['train'].values())
        clf.fit_with_train(X_fp['train'], y['train'])
    else:
        import hashlib
        PATH = './data/cache/'
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        fn_templ_emb = f'{PATH}templ_emb_{args.fp_size}_{args.template_fp_type}{args.fp_radius}_{len(template_list)}_{int(hashlib.sha512((str(template_list)).encode()).hexdigest(), 16)}.npy'
        if (os.path.exists(fn_templ_emb)): # load the template embedding
            print(f'loading tfp from file {fn_templ_emb}')
            templ_emb = np.load(fn_templ_emb)
            # !!! beware of different fingerprint types
            clf.template_list = list(template_list.values())

            if args.only_templates_in_batch:
                clf.templates_np = templ_emb
                clf.templates = None
            else:
                clf.templates = torch.from_numpy(templ_emb).float().to(clf.config.device)
        else:
            if args.template_fp_type=='MxFP':
                clf.template_list = list(template_list.values())
                clf.templates = torch.from_numpy(comb_template_fp).float().to(clf.config.device)
                clf.set_templates_recursively()
            elif args.template_fp_type=='Tfidf':
                clf.template_list = list(template_list.values())
                clf.templates = torch.from_numpy(tfidf_template_fp).float().to(clf.config.device)
                clf.set_templates_recursively()
            elif args.template_fp_type=='random':
                clf.template_list = list(template_list.values())
                clf.templates = torch.from_numpy(np.random.rand(len(template_list),args.fp_size)).float().to(clf.config.device)
                clf.set_templates_recursively()
            else:
                clf.set_templates(list(template_list.values()), which=args.template_fp_type, fp_size=args.fp_size, 
                                  radius=args.fp_radius, learnable=False, njobs=args.njobs, only_templates_in_batch=args.only_templates_in_batch)
                #if len(template_list)<100000:
                np.save(fn_templ_emb, clf.templates_np if args.only_templates_in_batch else clf.templates.detach().cpu().numpy().astype(np.float16))

        # concatinate the current fingerprint with a random fingerprint if the threshold is above
        if (args.concat_rand_template_thresh != -1) & (args.repl_quotient>0):
            REPLACE_FACTOR = int(args.repl_quotient) # default was 8

            # fold the original fingerprint
            pre_comp_templates = clf.templates_np if args.only_templates_in_batch else clf.templates.detach().cpu().numpy()
             
            # mask of labels with mor than 49 training samples
            l_mask = np.array([label_to_n_train_samples[k]>=args.concat_rand_template_thresh for k in template_list])
            print(f'Num of templates with added rand-vect of size {pre_comp_templates.shape[1]//REPLACE_FACTOR} due to >=thresh ({args.concat_rand_template_thresh}):',l_mask.sum())

            # remove the bits with the lowest variance
            v = pre_comp_templates.var(0)
            idx_lowest_var_half = v.argsort()[:(pre_comp_templates.shape[1]//REPLACE_FACTOR)]
            
            # the new zero-init-vectors
            pre = np.zeros([pre_comp_templates.shape[0], pre_comp_templates.shape[1]//REPLACE_FACTOR]).astype(np.float)
            print(pre.shape, l_mask.shape, l_mask.sum()) #(616, 1700) (11790,) 519
            print(pre_comp_templates.shape, len(template_list)) #(616, 17000) 616
            # only the ones with >thresh will receive a random vect
            pre[l_mask] = np.random.rand(l_mask.sum(), pre.shape[1])

            pre_comp_templates[:,idx_lowest_var_half] = pre

            #clf.templates = torch.from_numpy(pre_comp_templates).float().to(clf.config.device)
            if pre_comp_templates.shape[0]<100000:
                print('adding template_matrix to params')
                param = torch.nn.Parameter(torch.from_numpy(pre_comp_templates).float(), requires_grad=False)
                clf.register_parameter(name='templates+noise', param=param)
                clf.templates = param.to(clf.config.device)
                clf.set_templates_recursively()
            else: #otherwise might cause memory issues
                print('more than 100k templates')
                if args.only_templates_in_batch:
                    clf.templates = None
                    clf.templates_np = pre_comp_templates
                else:
                    clf.templates = torch.from_numpy(pre_comp_templates).float()
                    clf.set_templates_recursively()

    
    # set's this for the first layer!!
    if args.template_fp_type2=='MxFP':
        print('first_layer template_fingerprint is set to MxFP')
        clf.templates = torch.from_numpy(comb_template_fp).float().to(clf.config.device)
    elif args.template_fp_type2=='Tfidf':
        print('first_layer template_fingerprint is set to Tfidf')
        clf.templates = torch.from_numpy(tfidf_template_fp).float().to(clf.config.device)
    elif args.template_fp_type2=='random':
        print('first_layer template_fingerprint is set to random')
        clf.templates = torch.from_numpy(np.random.rand(len(template_list),args.fp_size)).float().to(clf.config.device)
    elif args.template_fp_type2=='stfp':
        print('first_layer template_fingerprint is set to stfp ! only works with 4096 fp_size')
        tfp = getTemplateFingerprint(list(template_list.values()))
        clf.templates = torch.from_numpy(tfp).float().to(clf.config.device)
        
    return clf


if __name__ == '__main__':
    
    args = parse_args()
    
    run_id = str(time()).split('.')[0]
    fn_postfix = str(args.exp_name) + '_' + run_id
        
    if args.wandb:
        import wandb
        wandb.init(project='mhn-react', entity='phseidl', name=args.dataset_type+'_'+args.model_type+'_'+fn_postfix, config=args.__dict__)
    else:
        wandb=None
        
    if not args.verbose:
        disable_rdkit_logging()
        
    if args.seed is not None:
        from .utils import seed_everything
        seed_everything(args.seed)
        print('seeded with',args.seed)

    # load csv or data
    if args.csv_path is None:
        X, y = load_USPTO(which=args.dataset_type)
        template_list = load_templates(which=args.dataset_type)
    else:
        X, y, template_list, test_reactants_can = load_dataset_from_csv(**vars(args))

    if args.addval2train:
        print('adding val to train')
        X['train'] = [*X['train'],*X['valid']]
        y['train'] = np.concatenate([y['train'],y['valid']])

    splits = ['train', 'valid', 'test']

    #TODO split up in seperate class
    if args.splitting_scheme == 'class-freq':
        X_all = np.concatenate([X[split] for split in splits], axis=0)
        y_all = np.concatenate([y[split] for split in splits])

        # sort class by frequency / assumes class-index is ordered (wich is mildely violated)
        res = y_all.argsort()

        # use same split proportions
        cum_split_lens = np.cumsum([len(y[split]) for split in splits]) #cumulative split length

        X['train'] = X_all[res[0:cum_split_lens[0]]]
        y['train'] = y_all[res[0:cum_split_lens[0]]]

        X['valid'] = X_all[res[cum_split_lens[0]:cum_split_lens[1]]]
        y['valid'] = y_all[res[cum_split_lens[0]:cum_split_lens[1]]]

        X['test'] = X_all[res[cum_split_lens[1]:]]
        y['test'] = y_all[res[cum_split_lens[1]:]]
        for split in splits:
            print(split, y[split].shape[0], 'samples (', y[split].max(),'max label)')

    if args.splitting_scheme == 'remove_once_in_train_and_not_in_test':
        print('remove_once_in_train')
        from collections import Counter
        cc = Counter()
        cc.update(y['train'])
        classes_set_only_once_in_train = set(np.array(list(cc.keys()))[ (np.array(list(cc.values())))==1])
        not_in_test = set(y['train']).union(y['valid']) - (set(y['test']))
        classes_set_only_once_in_train = (classes_set_only_once_in_train.intersection(not_in_test))
        remove_those_mask = np.array([yii in classes_set_only_once_in_train for yii in y['train']])
        X['train'] = np.array(X['train'])[~remove_those_mask]
        y['train'] = np.array(y['train'])[~remove_those_mask]
        print(remove_those_mask.mean(),'%', remove_those_mask.sum(), 'samples removed')

    if args.splitting_scheme == 'random':
        print('random-splitting-scheme:8-1-1')
        if args.ssretroeval:
            print('ssretroeval not available')
            raise NotImplementedError
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        def _unpack(lod):
            r = []
            for k,v in lod.items():
                [r.append(i) for i in v]
            return r

        X_all = _unpack(X)
        y_all = np.array( _unpack(y) )

        X['train'], X['test'], y['train'], y['test'] = train_test_split(X_all, y_all, test_size=0.2, random_state=70135)
        X['test'], X['valid'], y['test'], y['valid'] = train_test_split(X['test'], y['test'], test_size=0.5, random_state=70135)

        zero_shot = set(y['test']).difference( set(y['train']).union(set(y['valid'])) )
        zero_shot_mask = np.array([yi in zero_shot for yi in y['test']])
        print(sum(zero_shot_mask))
        #y['test'][zero_shot_mask] = list(zero_shot)[0] #not right but quick

    
    if args.model_type=='staticQK' or args.model_type=='retrosim':
        print('staticQK model: caution: use pattern, or rdk -fingerprint-embedding')

    fp_size = args.fp_size
    radius = args.fp_radius #quite important ;)
    fp_embedding = args.fp_type

    X_fp = featurize_smiles(X, fp_type=args.fp_type, fp_size=args.fp_size, fp_radius=args.fp_radius, njobs=args.njobs)

    if args.template_fp_type=='MxFP' or (args.template_fp_type2=='MxFP'):
        temp_part_to_fp = {}
        for i in template_list:
            tpl = template_list[i]
            for part in str(tpl).split('>>'):
                for p in str(part).split('.'):
                    temp_part_to_fp[p]=None
        for i, k in enumerate(temp_part_to_fp):
            temp_part_to_fp[k] = i

        fp_types = ['Morgan2CBF','Morgan4CBF', 'Morgan6CBF','AtomPair','TopologicalTorsion', 'Pattern', 'RDK']
        #MACCS ErG don't work --> errors with explicit / inplicit valence
        templates_fp = {}
        remaining = args.fp_size
        for fp_type in fp_types:
            #print(fp_type, end='\t')
            # if it's that last use up the remaining fps
            te_feat = FP_featurizer(fp_types=fp_type,
                                    max_features=(args.fp_size//len(fp_types)) if (fp_type != fp_types[-1]) else remaining,
                                    log_scale=False
                                    )
            templates_fp[fp_type] = te_feat.fit(list(temp_part_to_fp.keys())[:], is_smarts=True)
            #print(np.unique(templates_fp[fp_type]), end='\r')
            remaining -= templates_fp[fp_type].shape[1]
        templates_fp['fp'] = np.hstack([ templates_fp[f'{fp_type}'] for fp_type in fp_types])

    
    if args.template_fp_type=='MxFP' or (args.template_fp_type2=='MxFP'):
        comb_template_fp = compute_template_fp(fp_len= args.fp_size, reactant_pooling=args.reactant_pooling)


    
    if args.template_fp_type=='Tfidf' or (args.template_fp_type2 == 'Tfidf'):
        print('using tfidf template-fingerprint')
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = (list(template_list.values()))
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,12), max_features=args.fp_size)
        tfidf_template_fp = vectorizer.fit_transform(corpus).toarray()
        tfidf_template_fp.shape

    
    acutal_fp_size = X_fp['train'].shape[1]
    if acutal_fp_size != args.fp_size:
        args.fp_size = int(X_fp['train'].shape[1])
        print('Warning: fp-size has changed to', acutal_fp_size)

    
    label_to_n_train_samples = {}
    n_train_samples_to_label = defaultdict(list)
    n_templates = max(template_list.keys())+1 #max(max(y['train']), max(y['test']), max(y['valid']))
    for i in range(n_templates):
        n_train_samples = (y['train']==i).sum()
        label_to_n_train_samples[i] = n_train_samples
        n_train_samples_to_label[n_train_samples].append(i)

    
    up_to = 11
    n_samples = []
    masks = []
    ntes = range(up_to)
    mask_dict = {}

    for nte in ntes: # Number of training examples
        split = f'nte_{nte}'
        #print(split)
        mask = np.zeros(y['test'].shape)

        if isinstance(nte, int):
            for label_with_nte in n_train_samples_to_label[nte]:
                mask += (y['test'] == label_with_nte)

        mask = mask>=1
        masks.append(mask)
        mask_dict[str(nte)] = mask
        n_samples.append(mask.sum())

    # for greater than 10 # >10
    n_samples.append((np.array(masks).max(0)==0).sum())
    mask_dict['>10'] = (np.array(masks).max(0)==0)

    sum(n_samples), mask.shape

    ntes = range(50) #to 49
    for nte in ntes: # Number of training examples
        split = f'nte_{nte}'
        #print(split)
        mask = np.zeros(y['test'].shape)
        for label_with_nte in n_train_samples_to_label[nte]:
            mask += (y['test'] == label_with_nte)
        mask = mask>=1
        masks.append(mask)
    # for greater than 10 # >49
    n_samples.append((np.array(masks).max(0)==0).sum())
    mask_dict['>49'] = np.array(masks).max(0)==0

    print(n_samples)

    clf, hpn_config = set_up_model(args, template_list=template_list)
    clf = set_up_template_encoder(args, clf, label_to_n_train_samples=label_to_n_train_samples, template_list=template_list)
  
    if args.verbose:
        print(clf.config.__dict__)
        print(clf)

    wda = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-2)

    if args.wandb:
        wandb.watch(clf)

    
    # pretraining with applicablity matrix, if applicable
    if args.model_type == 'fortunato' or args.pretrain_epochs>1:
        print('pretraining on applicability-matrix -- loading the matrix')
        _, y_appl = load_USPTO(args.dataset_type, is_appl_matrix=True)
        if args.splitting_scheme == 'remove_once_in_train_and_not_in_test':
            y_appl['train'] = y_appl['train'][~remove_those_mask]

        # check random if the applicability is true for y
        splt = 'train'
        for i in range(500):
            i = np.random.randint(len(y[splt]))
            #assert ( y_appl[splt][i].indices == y[splt][i] ).sum()==1

        print('pre-training (BCE-loss)')
        for epoch in range(args.pretrain_epochs):
            clf.train_from_np(X_fp['train'], X_fp['train'], y_appl['train'], use_dataloader=True, is_smiles=False,
                          epochs=1, wandb=wandb, verbose=args.verbose, bs=args.batch_size, 
                          permute_batches=True, shuffle=True, optimizer=wda, 
                          only_templates_in_batch=args.only_templates_in_batch)
            y_pred = clf.evaluate(X_fp['valid'], X_fp['valid'], y_appl['valid'], 
                                  split='pretrain_valid', is_smiles=False, only_loss=True, 
                                  bs=args.batch_size,wandb=wandb)
            appl_acc = ((y_appl['valid'].toarray()) == (y_pred>0.5)).mean()
            print(f'{epoch:2.0f} -- train_loss: {clf.hist["loss"][-1]:1.3f}, loss_valid: {clf.hist["loss_pretrain_valid"][-1]:1.3f}, train_acc: {appl_acc:1.5f}')
    
    fn_hist = None
    y_preds = None

    for epoch in range(round(args.epochs / args.eval_every_n_epochs)):
        if not isinstance(clf, StaticQK):
            now = time()
            clf.train_from_np(X_fp['train'], X_fp['train'], y['train'], use_dataloader=True, is_smiles=False,
                          epochs=args.eval_every_n_epochs, wandb=wandb, verbose=args.verbose, bs=args.batch_size, 
                              permute_batches=True, shuffle=True, optimizer=wda, only_templates_in_batch=args.only_templates_in_batch)
            if args.verbose: print(f'training took {(time()-now)/60:3.1f} min for {args.eval_every_n_epochs} epochs')
        for split in ['valid', 'test']:
            print(split, 'evaluating', end='\r')
            now = time()
            #only_loss = ((epoch%5)==4) if args.dataset_type=='lg' else True
            y_preds = clf.evaluate(X_fp[split], X_fp[split], y[split], is_smiles=False, split=split, bs=args.batch_size, only_loss=args.eval_only_loss, wandb=wandb);

            if args.verbose: print(f'eval {split} took',(time()-now)/60,'min')
        if not isinstance(clf, StaticQK):
            try:
                print(f'{epoch:2.0f} -- train_loss: {clf.hist["loss"][-1]:1.3f}, loss_valid: {clf.hist["loss_valid"][-1]:1.3f}, val_t1acc: {clf.hist["t1_acc_valid"][-1]:1.3f}, val_t100acc: {clf.hist["t100_acc_valid"][-1]:1.3f}')
            except:
                pass

        now = time()
        ks = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
        for nte in mask_dict: # Number of training examples
            split = f'nte_{nte}'
            #print(split)
            mask = mask_dict[nte]

            topkacc = top_k_accuracy(np.array(y['test'])[mask], y_preds[mask, :], k=ks, ret_arocc=False)

            new_hist = {}
            for k, tkacc in zip(ks, topkacc):
                new_hist[f't{k}_acc_{split}'] = tkacc
            #new_hist[(f'arocc_{split}')] = (arocc)
            new_hist[f'steps_{split}'] = (clf.steps)

            for k in new_hist:
                clf.hist[k].append(new_hist[k])

        if args.verbose: print(f'eval nte-test took',(time()-now)/60,'min')

        fn_hist = clf.save_hist(prefix=f'USTPO_{args.dataset_type}_{args.model_type}_', postfix=fn_postfix)

    if args.save_preds:
        PATH = './data/preds/'
        if not os.path.exists(PATH):
            os.mkdir(PATH)
        pred_fn = f'{PATH}USPTO_{args.dataset_type}_test_{args.model_type}_{fn_postfix}.npy'
        print('saving predictions to',pred_fn)
        np.save(pred_fn,y_preds)
        args.save_preds = pred_fn

    
    if args.save_model:
        model_save_path = clf.save_model(prefix=f'USPTO_{args.dataset_type}_{args.model_type}_valloss{clf.hist.get("loss_valid",[-1])[-1]:1.3f}_',name_as_conf=False, postfix=fn_postfix)

        # Serialize data into file:
        import json
        json.dump( args.__dict__, open( f"data/model/{fn_postfix}_args.json", 'w' ) )
        json.dump( hpn_config.__dict__, 
                  open( f"data/model/{fn_postfix}_config.json", 'w' ) )

        print('model saved to', model_save_path)

    print(min(clf.hist.get('loss_valid',[-1])))
    
    if args.plot_res:
        from plotutils import plot_topk, plot_nte

        plt.figure()
        clf.plot_loss()
        plt.draw()

        plt.figure()
        plot_topk(clf.hist, sets=['valid'])
        if args.dataset_type=='sm':
            baseline_val_res = {1:0.4061, 10:0.6827, 50: 0.7883, 100:0.8400}
            plt.plot(list(baseline_val_res.keys()), list(baseline_val_res.values()), 'k.--')
        plt.draw()
        plt.figure()

        best_cpt = np.array(clf.hist['loss_valid'])[::-1].argmin()+1
        print(best_cpt)
        try:
            best_cpt = np.array(clf.hist['t10_acc_valid'])[::-1].argmax()+1
            print(best_cpt)
        except:
            print('err with t10_acc_valid')
        plot_nte(clf.hist, dataset=args.dataset_type.capitalize(), last_cpt=best_cpt, include_bar=True, model_legend=args.exp_name,
                 n_samples=n_samples, z=1.96)
        if os.path.exists('data/figs/'):
            try:
                os.mkdir(f'data/figs/{args.exp_name}/')
            except:
                pass
            plt.savefig(f'data/figs/{args.exp_name}/training_examples_vs_top100_acc_{args.dataset_type}_{hash(str(args))}.pdf')
        plt.draw()
        fn_hist = clf.save_hist(prefix=f'USTPO_{args.dataset_type}_{args.model_type}_', postfix=fn_postfix)

    
    if args.ssretroeval:
        print('testing on the real test set ;)')
        from .data import load_templates
        from .retroeval import run_templates, topkaccuracy
        from .utils import sort_by_template_and_flatten
        

        a = list(template_list.keys())
        #assert list(range(len(a))) == a
        templates = list(template_list.values())
        #templates = [*templates, *expert_templates]
        template_product_smarts = [str(s).split('>')[0] for s in templates]

        #execute all template
        print('execute all templates')
        test_product_smarts = [xi[0] for xi in X['test']] #added later
        smarts2appl = memory.cache(smarts2appl, ignore=['njobs','nsplits', 'use_tqdm'])
        appl = smarts2appl(test_product_smarts, template_product_smarts, njobs=args.njobs)
        n_pairs = len(test_product_smarts) * len(template_product_smarts)
        n_appl = len(appl[0])
        print(n_pairs, n_appl, n_appl/n_pairs)
        
        #forward
        split = 'test'
        print('len(X_fp[test]):',len(X_fp[split]))
        y[split] = np.zeros(len(X[split])).astype(np.int)
        clf.eval()
        if y_preds is None:
            y_preds = clf.evaluate(X_fp[split], X_fp[split], y[split], is_smiles=False,
                               split='ttest', bs=args.batch_size, only_loss=True, wandb=None);

        template_scores = y_preds #this should allready be test
        
        #### 
        if y_preds.shape[1]>100000:
            kth = 200
            print(f'only evaluating top {kth} applicable predicted templates')
            # only take top kth and multiply by applicability matrix
            appl_mtrx = np.zeros_like(y_preds, dtype=bool)
            appl_mtrx[appl[0], appl[1]] = 1

            appl_and_topkth = ([], [])
            for row in range(len(y_preds)):
                argpreds = (np.argpartition(-(y_preds[row]*appl_mtrx[row]), kth, axis=0)[:kth])
                # if there are less than kth applicable
                mask = appl_mtrx[row][argpreds]
                argpreds = argpreds[mask]
                #if len(argpreds)!=kth:
                #    print('changed to ', len(argpreds))

                appl_and_topkth[0].extend([row for _ in range(len(argpreds))])
                appl_and_topkth[1].extend(list(argpreds))   
            
            appl = appl_and_topkth
        ####
        
        print('running the templates')
        run_templates = run_templates #memory.cache( ) ... allready cached to tmp
        prod_idx_reactants, prod_temp_reactants =  run_templates(test_product_smarts, templates, appl, njobs=args.njobs)
        #sorted_results = sort_by_template(template_scores, prod_idx_reactants)
        #flat_results = flatten_per_product(sorted_results, remove_duplicates=True)
        #now aglomerates over same outcome
        flat_results = sort_by_template_and_flatten(y_preds, prod_idx_reactants, agglo_fun=sum) 
        accs = topkaccuracy(test_reactants_can, flat_results, [*list(range(1,101)), 100000])

        mtrcs2 = {f't{k}acc_ttest':accs[k-1] for k in [1,2,3,5,10,20,50,100,101]}
        if wandb:
            wandb.log(mtrcs2)
        print('Single-step retrosynthesis-evaluation, results on ttest:')
        #print([k[:-6]+'|' for k in mtrcs2.keys()])
        [print(k[:-6],end='\t') for k in mtrcs2.keys()]
        print()
        for k,v in mtrcs2.items():
            print(f'{v*100:2.2f}',end='\t')

    
    # save the history of this experiment
    EXP_DIR = 'data/experiments/'

    df = pd.DataFrame([args.__dict__])
    df['min_loss_valid'] = min(clf.hist.get('loss_valid', [-1]))
    df['min_loss_train'] = 0 if ((args.model_type=='staticQK') or (args.model_type=='retrosim')) else min(clf.hist.get('loss',[-1]))
    try:
        df['max_t1_acc_valid'] = max(clf.hist.get('t1_acc_valid', [0]))
        df['max_t100_acc_valid'] = max(clf.hist.get('t100_acc_valid', [0]))
    except:
        pass
    df['hist'] = [clf.hist]
    df['n_samples'] = [n_samples]

    df['fn_hist'] = fn_hist if fn_hist else None
    df['fn_model'] = '' if not args.save_model else model_save_path
    df['date'] = str(datetime.datetime.fromtimestamp(time()))
    df['cmd'] = ' '.join(sys.argv[:])


    if not os.path.exists(EXP_DIR):
        os.mkdir(EXP_DIR)

    df.to_csv(f'{EXP_DIR}{run_id}.tsv', sep='\t')
    df
