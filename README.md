# MHNreact
[![arXiv](https://img.shields.io/badge/acs.jcim-1c01065-yellow.svg)](https://doi.org/10.1021/acs.jcim.1c01065)
[![arXiv](https://img.shields.io/badge/arXiv-2104.03279-b31b1b.svg)](https://arxiv.org/abs/2104.03279)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.6-red.svg)](https://pytorch.org/get-started/previous-versions/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml-jku/mhn-react/blob/main/notebooks/colab_MHNreact_demo.ipynb)

  **[Abstract](#modern-hopfield-networks-for-few-and-zero-shot-reaction-template-prediction)**
| **[Environment](#environment)**
| **[Data](#data-and-processing)**
| **[Training](#training)**
| **[Loading](#loading-in-trained-models-and-evaluation)**
| **[Citation](#citation)**

Adapting modern Hopfield networks [(Ramsauer et al., 2021)](#mhn) (MHN) to associate different data modalities, molecules and reaction templates, to improve predictive performance for rare templates and single-step retrosynthesis.

![overview_image](data/figs/overview_tikz_transp.png?raw=true "Overview Figure")

## Improving Few- and Zero-Shot Reaction Template Prediction Using Modern Hopfield Networks
[paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c01065)

Philipp Seidl, Philipp Renz, Natalia Dyubankova, Paulo Neves, Jonas Verhoeven, Marwin Segler, Jörg K. Wegner, Sepp Hochreiter, Günter Klambauer

Finding synthesis routes for molecules of interest is essential in the discovery of new drugs and materials. To find such routes, computer-assisted synthesis planning (CASP) methods are employed, which rely on a single-step model of chemical reactivity. In this study, we introduce a template-based single-step retrosynthesis model based on Modern Hopfield Networks, which learn an encoding of both molecules and reaction templates in order to predict the relevance of templates for a given molecule. The template representation allows generalization across different reactions and significantly improves the performance of template relevance prediction, especially for templates with few or zero training examples. With inference speed up to orders of magnitude faster than baseline methods, we improve or match the state-of-the-art performance for top-k exact match accuracy for k ≥ 3 in the retrosynthesis benchmark USPTO-50k.

## Minimal working example

Opent the following colab for a quick training example [train-colab](https://colab.research.google.com/github/ml-jku/mhn-react/blob/main/notebooks/colab_MHNreact_demo.ipynb).

## Environment

### Anaconda

When using `conda`, an environment can be set up using
```bash
conda env create -f env.yml
```
To activate the environment call ```conda activate mhnreact_env```.

Additionally one needs to install [template-relevance](https://gitlab.com/mefortunato/template-relevance) which is included in this package, as well as [rdchiral](https://github.com/connorcoley/rdchiral) using pip:
```bash
cd data/temprel-fortunato/template-relevance-master/
pip install -e .
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
```

You may need to adjust the CUDA version.
The code was tested with:
- rdkit 2021.03.1 and 2020.03.4
- python 3.7 and 3.8
- pytorch 1.6
- rdchiral
- template-relevance (./data/temprel-forunato)
- CGRtools (only for preparing USPTO-golden)


A second option is to run
```bash
conda create -n mhnreact_env python=3.8
eval "$(conda shell.bash hook)"
conda activate mhnreact_env
conda install -c conda-forge rdkit
pip install torch scipy ipykernel matplotlib sklearn swifter
cd data/temprel-fortunato/template-relevance-master/
pip install -e .
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
```
which is equivialent to running the script ```bash ./scripts/make_env.sh```

### Docker

Another option is the provided docker-file within ```./tools/docker/``` with the following command.

```
DOCKER_BUILDKIT=1 docker build -t mhnreact:latest -f Dockerfile ../..
```

## Data and processing

The preprocessed data is contained in this repository: ```./data/processed/uspto_sm_*``` files contain the preprocessed files for template relevance prediction.

For single-step retrosynthesis the preprocessed and split data can be found in ````./data/USPTO_50k_MHN_prepro.csv````

All preprocessing steps can be replicated and found in ````./examples/prepro_*.ipynb```` for USPTO-sm, USPTO-lg as well as for USPTO-50k and USPTO-full.
USPTO-lg as well as USPTO-full are not contained due to their size, and would have to be created using the coresponding notebook.


## Training

Models can be trained using ````python mhnreact/train.py -m````

Selected calls are documented within ````./notebooks/*_training_*.ipynb````.

Arguments are documented within the module which can be retreived by adding ```--help``` to the call. Within the ```notebooks``` folder there are notebooks containing several examples.

Some main parameters are: 
 - ``model_type``: Model-type, choose from 'segler', 'fortunato', 'mhn' or 'staticQK', default:'mhn'
 - ``dataset_type``: Dataset 'sm', 'lg' for template relevance prediction; (use --csv_path for single-step retrosynthesis input)
 - ``fp_type``: Fingerprint type for the input only!: default: 'morgan', other options: 'rdk', 'ECFP', 'ECFC', 'MxFP', 'Morgan2CBF'
 - ``template_fp_tpye``: Template-fingerprint type: default: 'rdk', other options: 'MxFP', 'rdkc', 'tfidf', 'random'
 - ``hopf_beta``: hopfield beta parameter, default=0.005
 - ``hopf_asso_dim``: association dimension, default=512
 - ``ssretroeval``: single-step retrosynthesis evaluation, default=False


an example call for single-step retrosynthesis is:
```bash
python -m mhnreact.train --model_type=mhn --fp_size=4096 --fp_type morgan --template_fp_type rdk --concat_rand_template_thresh 1 \
--exp_name test --dataset_type 50k --csv_path ./data/USPTO_50k_MHN_prepro.csv.gz --ssretroeval True --seed 0
```

## Loading in trained Models and Evaluation

How to load in trained models can be seen in ```./examples/20_evaluation.ipynb```. The model is then used to predict on a test set.

## Train on custom data

Preprocess the data in a format as can be found in ````./data/USPTO_50k_MHN_prepro.csv```` and use the argument ```--csv_path```.

## Citation

To cite this work, you can use the following bibtex entry:
 ```bibtex
@article{seidl2021modern,
	author = {Seidl, Philipp and Renz, Philipp and Dyubankova, Natalia and Neves, Paulo and Verhoeven, Jonas and Segler, Marwin and Wegner, J{\"o}rg K. and Hochreiter, Sepp and Klambauer, G{\"u}nter},
	title = {Improving Few- and Zero-Shot Reaction Template Prediction Using Modern Hopfield Networks},
	journal = {Journal of Chemical Information and Modeling},
	volume = {62},
	number = {9},
	pages = {2111-2120},
	institution = {Institute for Machine Learning, Johannes Kepler University, Linz},
	year = {2022},
	doi = {10.1021/acs.jcim.1c01065},
	url = {https://doi.org/10.1021/acs.jcim.1c01065},
}
 ```

## References
 - <span id="mhn">Ramsauer et al.(2020).</span> ICLR2021 ([pdf](https://arxiv.org/abs/2008.02217))

## Keywords
Drug Discovery, CASP, Machine Learning, Synthesis, Zero-shot, Modern Hopfield Networks
