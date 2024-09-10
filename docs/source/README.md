# DocOctopy


<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Code style: black" src="https://img.shields.io/badge/python-v3.10-blue"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="Code style: black" src="https://img.shields.io/badge/uses-pytorch & pytorch lighting-blueviolet"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code style-black-000000.svg"></a>


## What is this?
DocOctopy , the **research** library for kinematics, kinetics, and everything else you can think of that has to do with EMG and AIs.
> **Important**  
> Be aware that this project is used for **research**. Do not expect the same stability as from numpy for example.

## What papers use this package?
<p float="left">
  <a href="https://doi.org/10.1109/TBME.2024.3432800" target="_blank"> <img src="_static/papers/Learning.jpg" width="31%" /> </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://doi.org/10.33965/ijcsis_2024190101" target="_blank"> <img src="_static/papers/Analysis.jpg" width="28.4%" /> </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://doi.org/10.1101/2024.05.28.24307964" target="_blank"> <img src="_static/papers/Identification.jpg" width="31%" /> </a>
</p>

<p float="left">
<a href="https://doi.org/10/gtm4bt" target="_blank"> <img src="_static/papers/Influence.jpg" width="28.4%" /> </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://doi.org/10/gsgk4s" target="_blank"> <img src="_static/papers/Proportional.jpg" width="31%" /> </a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://doi.org/10/gq2f47" target="_blank"> <img src="_static/papers/Accurate.jpg" width="31%" /> </a>
</p>

## How to install?
> **WARNING**   
> Order matters!
- clone
- install [poetry](https://python-poetry.org/docs/#installation)
- based on your hardware:
  - GPU (NVIDIA): `poetry install -E gpu --with pytorch_gpu`
  - CPU: `poetry install -E cpu`
- for building the documentation add `docs` to the *with* flag: `poetry install --with docs`

## What is what?
This projects uses the following structure:
- doc_octopy: This is the main package. It contains:
  - datasets: Contains data loaders and creators as well as a lot of filters to preprocess the data.
  - models: Contains all models and their components.
  - utils: Various utilities from training loggers to constants.
- docs: Contains the documentation.
- examples: Contains examples on how to use the package.
- tests: Contains tests for the package.