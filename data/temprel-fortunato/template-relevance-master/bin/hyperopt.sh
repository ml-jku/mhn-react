#!/bin/bash

crun="docker run -it --rm -v ${PWD}:/work -w /work -u $(id -u $USER):$(id -g $USER) registry.gitlab.com/mlpds_mit/askcos/template-relevance:latest"

for LAYERS in {1..3..1};
do
  for HIDDEN in {1000..3000..1000};
  do
    for HIGHWAY in 0 1 3 5;
    do
      NAME="${LAYERS}x_${HIDDEN}_h${HIGHWAY}"
      $crun python train.py --templates data/processed/retro.templates.uspto_50k.json.gz --num-hidden $LAYERS --hidden-size $HIDDEN --num-highway $HIGHWAY --nproc=40 --model-name $NAME > log.$NAME
      $crun python train_appl.py --num-hidden $LAYERS --hidden-size $HIDDEN --num-highway $HIGHWAY --nproc=40 --model-name ${NAME}_appl > log.${NAME}_appl
      $crun python train.py --pretrain-weights training/${NAME}_appl-weights.hdf5 --templates data/processed/retro.templates.uspto_50k.json.gz --num-hidden $LAYERS --hidden-size $HIDDEN --num-highway $HIGHWAY --nproc=40 --model-name ${NAME}_pretrained > log.${NAME}_pretrain
    done
  done
done
