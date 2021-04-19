# MHNreact
Modern Hopfield Network (MHN) for template relevance prediction

# Modern Hopfield Networks for Few- and Zero-Shot Reaction Prediction

Philipp Seidl, Philipp Renz, 
Natalia Dyubankova, Paulo Neves, Jonas Verhoeven, Jörg K. Wegner, 
Sepp Hochreiter, Günter Klambauer

MHNreact is using a MHN [(Ramsauer et al., 2121)](#mhn) for reaction template relevance prediction. 

### Sub-repositories

This repository contains code for the LSTM addition, traffic forecasting and pendulum experiments.
The neural arithmetic experiments were conducted on a [fork](https://github.com/hoedt/stable-nalu) of the repository from [Madsen et al. (2020)](#nau).
The experiments in hydrology were conducted using the [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) framework.

## Paper

[Pre-print](https://arxiv.org/abs/2104.03279)

### Abstract

An essential step in the discovery of new drugs and materials is the synthesis of a molecule that exists so far only as an idea to test its biological and physical properties. While computer-aided design of virtual molecules has made large progress, computer-assisted synthesis planning (CASP) to realize physical molecules is still in its infancy and lacks a performance level that would enable large-scale molecule discovery. CASP supports the search for multi-step synthesis routes, which is very challenging due to high branching factors in each synthesis step and the hidden rules that govern the reactions. The central and repeatedly applied step in CASP is reaction prediction, for which machine learning methods yield the best performance. We propose a novel reaction prediction approach that uses a deep learning architecture with modern Hopfield networks (MHNs) that is optimized by contrastive learning. An MHN is an associative memory that can store and retrieve chemical reactions in each layer of a deep learning architecture. We show that our MHN contrastive learning approach enables few- and zero-shot learning for reaction prediction which, in contrast to previous methods, can deal with rare, single, or even no training example(s) for a reaction. On a well established benchmark, our MHN approach pushes the state-of-the-art performance up by a large margin as it improves the predictive top-100 accuracy from 0.858±0.004 to 0.959±0.004. This advance might pave the way to large-scale molecule discovery.

### Citation

To cite this work, you can use the following bibtex entry:
 ```bib
@report{seidl2021modern,
	author = {Seidl, Philipp and Renz, Philipp and Dyubankova, Natalia and Neves, Paulo and Verhoeven, Jonas and Wegner, J{\"o}rg K. and Hochreiter, Sepp and Klambauer, G{\"u}nter},
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
 - <span id="mhn">Ramsauer et al.(2020).</span> ICLR, ... ([pdf](https://arxiv.org/abs/2008.02217))
