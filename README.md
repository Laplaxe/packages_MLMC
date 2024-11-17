# Machine Learning for Monte Carlo

This directory contains a series of files to train and use machine learning architectures to aid the Monte Carlo sampling of the 2D Edwards-Anderson model.

The directory contains the following files:
- N4.py: file related to the N4 architecture (https://arxiv.org/pdf/2407.19483).
- data_loads.py: directory to load the data. The data are assumed to be in a dati_PT directory suitably formatted (see below).
- ellMade.py: file related to the $\ell$-MADE architecture (https://arxiv.org/pdf/2407.19483)
- geometry.py: file to handle the geometry of the problem (e.g., the ordering of the spins in a spiral update).
- global_steps.py: global steps for the MADE and NADE architectures (e.g generation of the configurations)
- made.py: file related to the MADE architecture.
- nade.py: file related to the NADE architecture.
- utilities.py: some utility functions (e.g., computation of the best-spaced temperatures based on the specific heat).
