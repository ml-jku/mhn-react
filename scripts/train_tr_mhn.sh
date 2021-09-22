#!/bin/sh
#call from parent folder
eval "$(conda shell.bash hook)"
conda activate mhnreact_env
#sm_MHN_no_test
python -m mhnreact.train --batch_size=1024 --concat_rand_template_thresh=2 --dataset_type=sm \
--device=best --dropout=0.2 --epochs=30 --exp_name=rerun --fp_size=4096 --fp_type=morgan --hopf_asso_dim=1024 \
--hopf_association_activation=None --hopf_beta=0.03 --temp_encoder_layers=1 --mol_encoder_layers=1 \
--norm_asso=True --norm_input=False --hopf_num_heads=1 --lr=0.001 --model_type=mhn --save_preds True --save_model True \
--exp_name=rerun --seed 0 --fp_radius 2