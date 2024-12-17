# MyoVerse


<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Code style: black" src="https://img.shields.io/badge/python-%3E=3.10,%20%3C=3.13-blue"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="Code style: black" src="https://img.shields.io/badge/uses-pytorch & pytorch lighting-blueviolet"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code style-black-000000.svg"></a>

> [!TIP]
> Take a look at our [documentation](https://nsquaredlab.github.io/MyoVerse/README.html).

## What is this?
MyoVerse , the **research** library for kinematics, kinetics, and everything else you can think of that has to do with EMG and AIs.
> **Important**  
> Be aware that this project is used for **research**. Do not expect the same stability as from numpy for example.

## How to install?
> **WARNING**   
> Order matters!
- clone
- install [poetry](https://python-poetry.org/docs/#installation)
- based on your OS and hardware please check pytorch's [installation guide](https://pytorch.org/get-started/locally/)
- for building the documentation add `docs` to the *with* flag: `poetry install --with docs`

## What is what?
This projects uses the following structure:
- myoverse: This is the main package. It contains:
  - datasets: Contains data loaders and creators as well as a lot of filters to preprocess the data.
  - models: Contains all models and their components.
  - utils: Various utilities.
- docs: Contains the documentation.
- examples: Contains examples on how to use the package.
- tests: Contains tests for the package.

## What papers/preprints use this package?
| Journal / Preprint Server                                                                              | DOI                                                                              |
|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| IEEE Transactions on Biomedical Engineering                                                            | [10.1109/TBME.2024.3432800](https://doi.org/10.1109/TBME.2024.3432800)           |
| International Journal of Computer Science and Information Security                                     | [10.33965/ijcsis_2024190101](https://doi.org/10.33965/ijcsis_2024190101)         |
| medRxiv                                                                                                | [10.1101/2024.05.28.24307964](https://doi.org/10.1101/2024.05.28.24307964)       |
| Journal of Neural Engineering                                                                          | [10.1088/1741-2552/ad3498](https://doi.org/10.1088/1741-2552/ad3498)                            |
| IEEE Transactions on Neural Systems and Rehabilitation Engineering                                     | [10.1109/TNSRE.2023.3295060](https://doi.org/10.1109/TNSRE.2023.3295060)         |
| 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) | [10.1109/EMBC48229.2022.9870937](https://doi.org/10.1109/EMBC48229.2022.9870937) |
