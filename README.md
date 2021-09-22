# MHNreact
[![arXiv](https://img.shields.io/badge/arXiv-2104.03279-b31b1b.svg)](https://arxiv.org/abs/2104.03279)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.6-red.svg)](https://pytorch.org/get-started/previous-versions/)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

  **[Abstract](#modern-hopfield-networks-for-few-and-zero-shot-reaction-template-prediction)**
| **[Environment](#environment)**
| **[Data](#data-and-processing)**
| **[Training](#training)**
| **[Loading](#loading-in-trained-models-and-evaluation)**
| **[Citation](#citation)**

Adapting modern Hopfield networks [(Ramsauer et al., 2021)](#mhn) (MHN) to associate different data modalities, molecules and reaction templates, to improve predictive performance for rare templates and single-step retrosynthesis.

![overview_image](data/figs/overview_tikz_transp.png?raw=true "Overview Figure")

## Modern Hopfield Networks for Few- and Zero-Shot Reaction Template Prediction

Philipp Seidl, Philipp Renz, Natalia Dyubankova, Paulo Neves, Jonas Verhoeven, Marwin Segler, Jörg K. Wegner, Sepp Hochreiter, Günter Klambauer

Finding synthesis routes for molecules of interest is an essential step in the discovery of new drugs and materials. To find such routes, computer-assisted synthesis planning (CASP) methods are employed which rely on a model of chemical reactivity.
In this study, we model single-step retrosynthesis in a template-based approach using modern Hopfield networks (MHNs). We adapt MHNs to associate different modalities, reaction templates and molecules, which allows the model to leverage structural information about reaction templates.
This approach significantly improves the performance of template relevance prediction, especially for templates with few or zero training examples.
With inference speed several times faster than that of baseline methods, we improve predictive performance for top-k exact match accuracy for k≥5 in the retrosynthesis benchmark USPTO-50k.

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


A second option is to run (and hope for the best ;)
```bash
conda create -n mhnreact_env python=3.8
conda activate mhnreact_env #doesn't work unfortunately :S
pip install torch scipy ipykernel matplotlib sklearn swifter
cd data/temprel-fortunato/template-relevance-master/
pip install -e .
pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"
```

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

## Citation

To cite this work, you can use the following bibtex entry:
 ```bibtex
@report{seidl2021modern,
	author = {Seidl, Philipp and Renz, Philipp and Dyubankova, Natalia and Neves, Paulo and Verhoeven, Jonas and Segler, Marwin and Wegner, J{\"o}rg K. and Hochreiter, Sepp and Klambauer, G{\"u}nter},
	title = {Modern Hopfield Networks for Few- and Zero-Shot Reaction Prediction},
	institution = {Institute for Machine Learning, Johannes Kepler University, Linz},
	type = {preprint},
	date = {2021},
	url = {http://arxiv.org/abs/2104.03279},
	eprinttype = {arxiv},
	eprint = {2104.03279},
}
 ```

## References
 - <span id="mhn">Ramsauer et al.(2020).</span> ICLR2021 ([pdf](https://arxiv.org/abs/2008.02217))

## Keywords
Drug Discovery, CASP, Machine Learning, Synthesis, Zero-shot, Modern Hopfield Networks
