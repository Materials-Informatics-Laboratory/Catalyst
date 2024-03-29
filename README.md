
![Screenshot](https://github.com/Materials-Informatics-Laboratory/Catalyst/tree/main/visuals/catalyst.jpg?raw=true)
visuals/catalyst.jpg
# Catalyst
General-purpose toolkit for building machine learning models to study atomic-scale structure-property relationships.

## Installation

The following dependencies need to be installed before installing `catalyst`. The installation time is typically within 10 minutes on a normal local machine.
- PyTorch (`pytorch>=1.8.1`)
- PyTorch-Geometric (`pyg>=2.0.1`): for implementing graph representations
- UMAP-learn (`umap-learn>=0.5.3`)
- Networkx (`networkx>=2.8.6`)
- Scipy (`scipy>=1.9.0`)
- Numpy (`numpy>=1.21.1`)
- Atomic Simulation Environment (`ase>= 3.22.1`): for reading/writing atomic structures

To install `catalyst`, clone this repo and run:
```bash
pip install -e /path/to/the/repo
```

The `-e` option signifies an [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/), which is well suited for development; this allows you to edit the source code without having to re-install.

To uninstall:
```bash
pip uninstall catalyst
```

## How to use

`catalyst` is intended to be a plug-and-play framework where you provide data in the form as an `ase` atoms object. Tools such as graph construction and machine learning model training/testing are provided for you to build models for your atomistic systems. `catalyst` abstracts much of this process and provides an easy-to-use API for rapid building of ML models. `catalyst` also provides access to post-process scripts and visualization tools to help you better understand how your models behave, and why. 

- The `src` folder contains the source code.
- The `examples` folder contains examples that explain how to use `catalyst`'s various functions.