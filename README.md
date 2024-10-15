<!-- badges: start -->
<!-- WARNING: -->
<!-- The ?branch=release-x.y.y is updated automatically by the initiate_version_release workflow -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
<!-- badges: end -->

![Screenshot](https://github.com/Materials-Informatics-Laboratory/Catalyst/blob/main/visuals/catalyst.jpg?raw=true)

# Catalyst
General-purpose toolkit for analyzing atomistic simulations via machine learning. Whether it's building regression models to predict materials properties or performing unsupervised projections and clustering to characterize and understand trends, Catalyst is the perfect tool. Catalyst is built around being user-friendly, abstracting much of the complex machine learning and providing an easy-to-user API that wraps heavy-lifting codes such as PyTorch. Catalyst provides a highly GPU-parallelized framework that streamlines regression model training so you can quickly build and deploy property models for your research needs. Catalyst comes with several built-in and pre-optimized raph neural network routines, such as ALIGNN-d and MeshGraph networks, but also provides the user with the ability to design their own ML architechtures (using PyTorch) and train them via our GPU-parallelized training routines. Catalyst is designed with the user in-mind and comes optimized and ready for all of your research needs. Catalyst has been optimized for both Windows and Linux and has been tested to work on systems ranging from small laptops to high performance computing environments. See the user manual for a detailed list of internal parameters, examples, and the theory behind Catalyst.

If you would like to see new features added to Catalyst feel free to request something via the Issues tab. Please also report bugs and we patch them as quickly as possible.

## Installation

The following dependencies need to be installed before installing `catalyst`. The installation time is typically within 10 minutes on a normal local machine.
- PyTorch (`pytorch=2.4.1`)
- PyTorch-Geometric (`pyg>=2.6.1`): for implementing graph representations
- Networkx (`networkx>=3.3`): for using SODAS sampling
- Scipy (`scipy>=1.13.0`)
- Numpy (`numpy>=2.0.2`)
- Atomic Simulation Environment (`ase>= 3.22.1`): for reading/writing atomic structures
- PeriodicTable (`periodictable >= 1.7.1`): for graph construction
- Numbda (`numba >= 0.60.0`): for CUDA calls

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

## Contact

- Questions regarding Catlayst should be directed to jc112358@bu.edu.

## Cite Catalyst!
- Please cite Catalyst using the reference below:
