# MHNreact
Modern Hopfield Network (MHN) for template relevance prediction

# Modern Hopfield Networks for Few- and Zero-Shot Reaction Prediction

Philipp Seidl, Philipp Renz, 
Natalia Dyubankova, Paulo Neves, Jonas Verhoeven, Marwin Segler, Jörg K. Wegner, 
Sepp Hochreiter, Günter Klambauer

MHNreact is using a modern Hopfield network [(Ramsauer et al., 2021)](#mhn) for reaction template relevance prediction. 


## Paper

[Pre-print](https://arxiv.org/abs/2104.03279)

### Abstract

Finding synthesis routes for molecules of interest is an essential step in the discovery
of new drugs and materials.
To find such routes, computer-assisted synthesis planning (CASP) methods are employed, which rely on a model of chemical reactivity.
In this study, we model single-step retrosynthesis in a template-based approach using modern Hopfield networks (MHNs).
We adapt MHNs to associate different modalities, reaction templates and molecules, which allows the model to leverage structural information about reaction templates.
This approach significantly improves the performance of template relevance prediction, especially for templates with few or zero training examples.
With inference speed several times faster than baseline methods, we improve predictive performance for top-k exact match accuracy for k≥5 in the retrosynthesis benchmark USPTO-50k. 

### Citation

To cite this work, you can use the following bibtex entry:
 ```bib
@report{seidl2021modern,
	author = {Seidl, Philipp and Renz, Philipp and Dyubankova, Natalia and Neves, Paulo and Verhoeven, Jonas and Segler, Marwin and Wegner, J{\"o}rg K. and Hochreiter, Sepp and Klambauer, G{\"u}nter},
	title = {Modern Hopfield Networks for Few- and Zero-Shot Reaction Template Prediction},
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
