#!/bin/sh
#call from parent folder
eval "$(conda shell.bash hook)"
conda activate mhnreact_env
#sm_DNN_yes_test
python -m mhnreact.train --device best --dataset_type sm --exp_name rerun \
--model_type fortunato --pretrain_epochs 25 --epochs 10 --hopf_asso_dim 2048 \
--fp_type morgan --fp_size 4096 --dropout 0.15 --lr 0.0005 \
--mol_encoder_layers 1 --batch_size 256 --save_preds True --save_model True \
--exp_name=rerun --seed 0 --fp_radius 2