# Template relevance training

### Prerequisites

This repository is designed to function using the Docker container:
`registry.gitlab.com/mefortunato/template-relevance`
which contains [RDKit](https://github.com/rdkit), [TensorFlow](https://www.tensorflow.org/), and various other dependencies, including OpenMPI in order to scale this workflow up to a much larger number of reactions. Because this is such a small data set of reactions (described below in `Step 0`), the entire workflow can be run in just a few hours on a modest machine.

In order to run these scripts using the Docker container, first [make sure Docker is installed on your machine](https://docs.docker.com/install/), then run:
```
docker pull registry.gitlab.com/mefortunato/template-relevance
```

To make it easier to execute commands in the future, it is recommended to create an alias in `~/.bashrc` such as:
```
alias rdkit="docker run -it --rm -v ${PWD}:/work -w /work registry.gitlab.com/mefortunato/template-relevance"
```
or if you're using singularity:
```
alias rdkit="singularity exec -B $(pwd -P):/work --pwd /work /path/to/singularity/image.simg"
```

which will mount the current working directory (where ever you run `rdkit` from in the future) into `/work` inside the container. Any new files that get created will then also be available from the host machine, in the current working directory.

The rest of this README will assume you are running the commands from inside the container, i.e. using the above alias:
```
rdkit python myscript.py
```

In even simpler terms, preface the rest of the commands listed in this README with the alias shown above.

### Step 0 - Get the raw reaction data

Command(s) to run:
```
python get_uspto_50k.py 
```

This example uses a small set of reactions from [https://doi.org/10.1021/ci5006614](https://doi.org/10.1021/ci5006614):
```
Development of a Novel Fingerprint for Chemical Reactions and Its Application to Large-Scale Reaction Classification and Similarity  
Nadine Schneider, Daniel M. Lowe, Roger A. Sayle, Gregory A. Landrum
J. Chem. Inf. Model. 2015 55 1 39-53
```

The reactions were curated from the USPTO data set and provided in the SI. The script downloads the SI and puts the raw data into a new directory `data/raw`. If you plan to use a different set of reactions, it could be useful to run this script once to see the expected format of `data/raw/reactions.json.gz`, which is required for the following step.

### Step 1 - Extract templates

Command(s) to run:
```
python process.py
```

The `process.py` script (which uses `template_extractor.py` from the [rdchrial Python package](https://github.com/connorcoley/rdchiral)) will extract retrosynthetic templates for each reaction and attempt to verify their accuracy (i.e. - if I apply the template to the product do I recover the reactant(s)). Ultimately all reactions with valid templates and their associated template will be saved to disk via pandas at `data/processed/templates.df.json.gz`.

The `process.py` script will also perform a train/validation/test split according to the `--calc-split` flag which can be one of the following `['stratified', 'random']` and will create the following files for training a machine learning model:
* train.input.smiles.npy
* valid.input.smiles.npy
* valid.labels.classes.npy
* train.labels.classes.npy
* test.input.smiles.npy
* test.labels.classes.npy

If you would like to use your own data split, please create the appropriate corresponding files.

If you are using a dataset other than the default uspto_50k, you can give your dataset a name which will show up in various file names with the `--template-set-name` flag (this defaults to `uspto_50k`).

This script also prepares certain files for the ASKCOS software package in `data/processed` named `reactions.\[template_set_name\].json.gz` and `historian.\[template_set_name\].json.gz` in addition to `retro.templates.\[template_set_name\].json.gz`.

### Step 2 - Training a baseline model

Command(s) to run:
```
python train.py --templates data/processed/retro.templates.uspto_50k.json.gz
```

To train a template relevance baseline model (without pre-training on template applicability) use the `train.py` script. The available command line arguments are as follows:

* `--train-smiles` (str): Path to training smiles .npy file
* `--valid-smiles` (str): Path to validation smiles .npy file
* `--train-labels` (str): Path to training class labels .npy file
* `--valid-labels` (str): Path to validation class labels .npy file
* `--no-validation` (bool): Option to skip validation if no validation set exists
* `--num-classes` (int): Number of classes in dataset. Either this or the `--templates` flag is required.
* `--templates` (str): Path to JSON file of unique templates generated during processing step above. This or `--num-classes` is required.
* `--fp-length` (int): Length of Morgan fingerprint to use for input to model (default=2048)
* `--fp-radius` (int): Radius of Morgan fingerprint to use for input to model (default=2)
* `--pretrain-weights` (str): Path to model weights file to use to initialize model weights
* `--weight-classes` (bool): Boolean flag whether to weight contribution to loss by template popularity (default=True)
* `--num-hidden` (int): Number of hidden layers to use in neural network (default=1)
* `--hidden-size` (int): Size for hidden layers in neural network (default=1024)
* `--num_highway` (int): Number of highway layers (default=5)
* `--dropout` (float): Dropout to use after each hidden layer during training (default=0.2)
* `--learning-rate` (float): Learning rate to use with Adam optimizer (default=0.001)
* `--activation` (str): Type of activation to use (default='relu')
* `--batch-size` (int): Batch size to use during training (default=512)
* `--epochs` (int): Max number of epochs during training (default=25)
* `--early-stopping` (int): Number of epochs to use as patience for early stopping (default=3)
* `--model-name` (str): Name to give to model, used in various file names (default='template-relevance')
* `--nproc` (str): Number of processors to use for data pre-processing, 0 means don't processes in parallel (default=0)

### Step 3 - Test a baseline model

Command(s) to run:
```
python test.py --templates data/processed/retro.templates.uspto_50k.json.gz --model-weights training/template-relevance-weights.hdf5 --model-name template-relevance --accuracy --reciprocal-rank
```

This will produce output two files in a new `evaluation/` folder: `template-relevance.accuracy.json` and `template-relevance.recip_rank.json`.
