#!/bin/bash
conda create -n mhnreact_env python=3.8
eval "$(conda shell.bash hook)"
conda activate mhnreact_env
conda install -c conda-forge rdkit
pip install torch scipy ipykernel matplotlib sklearn swifter
cd data/temprel-fortunato/template-relevance-master/
pip install -e .
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
# conda install -c conda-forge -c ljn917 rdchiral_cpp # consider using fast c_pp rdchiral -- doesn't work right now... -- try later ;)