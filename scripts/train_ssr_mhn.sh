#!/bin/sh
#call from parent folder
eval "$(conda shell.bash hook)"
conda activate mhnreact_env
python -m mhnreact.train --batch_size=1024 --concat_rand_template_thresh=1 --device=best --dropout=0.4 \
--epochs=40 --fp_size=30000 --fp_type=maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+layered+mhfp --hopf_asso_dim=1024 --hopf_association_activation=None --hopf_beta=0.03 --lr=0.0001 --model_type=mhn --template_fp_type rdkc+pattern+morganc+layered+atompair+erg+topologicaltorsion+mhfp \
--exp_name=retro_selected --reactant_pooling lgamma --hopf_n_layers 2 --layer2weight 0.05 --template_fp_type2 rdk \
--dataset_type 50k --csv_path ./data/USPTO_50k_MHN_prepro.csv.gz --split_col split --ssretroeval True --seed 0 --eval_every_n_epochs 10 --addval2train True