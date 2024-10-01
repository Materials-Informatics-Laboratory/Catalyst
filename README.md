<!-- badges: start -->
<!-- WARNING: -->
<!-- The ?branch=release-x.y.y is updated automatically by the initiate_version_release workflow -->
[![status](https://joss.theoj.org/papers/1bbc43a2be86f5d3f831cedb5cf81812/status.svg)](https://joss.theoj.org/papers/1bbc43a2be86f5d3f831cedb5cf81812)
[![On Label CRAN Checks](https://github.com/furrer-lab/abn/actions/workflows/onlabel_CRAN_checks.yml/badge.svg?branch=release-3.1.3)](https://github.com/furrer-lab/abn/actions/workflows/onlabel_CRAN_checks.yml)
[![Codecov](https://img.shields.io/codecov/c/github/furrer-lab/abn)](https://app.codecov.io/gh/furrer-lab/abn)
[![GitHub R package version](https://img.shields.io/github/r-package/v/furrer-lab/abn)](https://github.com/furrer-lab/abn/tags)
![cran](https://www.r-pkg.org/badges/version-ago/abn) 
![downloads](https://cranlogs.r-pkg.org/badges/grand-total/abn) 
![LICENCE](https://img.shields.io/cran/l/abn)
<!-- badges: end -->

![Screenshot](https://github.com/Materials-Informatics-Laboratory/Catalyst/blob/main/visuals/catalyst.jpg?raw=true)

# Catalyst
General-purpose toolkit for analyzing atomistic simulations via graph-based machine learning. Currently, Catalyst can genearte and visualize graphs of both global and local atomistic environments, build grpah neural networks to both characterize systems and learn materials properties. The current learning process can be "black-box" or "interpretable" depending on your goals. There are many examples showcasing how to extract interpretable feature rankings. Catalyst also supports multi-GPU training via Pytroch's DistributedDataParallel package, and our current testing shows a speed-up of between 2-6 ordgers of magnitude when compared to serial training on GPU and CPU, respectively.

## Installation

The following dependencies need to be installed before installing `catalyst`. The installation time is typically within 10 minutes on a normal local machine.
- PyTorch (`pytorch>=1.8.1`)
- PyTorch-Geometric (`pyg>=2.0.1`): for implementing graph representations
- Networkx (`networkx>=2.8.6`): for using SODAS sampling
- Scipy (`scipy>=1.9.0`)
- Numpy (`numpy>=1.21.1`)
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
