<img src="./docs/source/_static/myoverse_logo.png" height="250">

<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Code style: black" src="https://img.shields.io/badge/python-%3E=3.10,%20%3C=3.13-blue"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="Code style: black" src="https://img.shields.io/badge/uses-pytorch & pytorch lighting-blueviolet"></a>

> [!TIP]
> Dive deeper into our features and usage with the official [documentation](https://nsquaredlab.github.io/MyoVerse/README.html).

# MyoVerse - The AI toolkit for myocontrol research

## What is MyoVerse? 
MyoVerse is your cutting-edge **research** companion for unlocking the secrets hidden within biomechanical data! It's specifically designed for exploring the complex interplay between **electromyography (EMG)** signals, **kinematics** (movement), and **kinetics** (forces).

Leveraging the power of **PyTorch** and **PyTorch Lightning**, MyoVerse provides a comprehensive suite of tools, including:
*   **Data loaders** and **preprocessing filters** tailored for biomechanical signals.
*   Peer-reviewed **AI models** and components for analysis and prediction tasks.
*   Essential **utilities** to streamline the research workflow.

Whether you're predicting movement from muscle activity, analyzing forces during motion, or developing novel AI approaches for biomechanical challenges, MyoVerse aims to accelerate your research journey.

> [!IMPORTANT]  
> MyoVerse is built for **research**. While powerful, it's evolving and may not have the same level of stability as foundational libraries like NumPy. We appreciate your understanding and contributions!

## Installation

### For Users (Using MyoVerse in your project)

1.  **Install MyoVerse:** Use pip to install the package directly from GitHub (replace `main` with a specific tag/branch if needed):
    ```bash
    pip install git+https://github.com/NsquaredLab/MyoVerse.git
    ```
2.  **Install PyTorch with GPU:** After installing MyoVerse, install PyTorch.
    *   **For Windows:** Visit the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and select the appropriate options (Stable, Windows, Pip, Python, your CUDA version). Copy the generated command, which will look something like this (example for CUDA 12.6):
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --upgrade
        ```
    *   **For Linux:** Pytorch GPU is already installed by MyoVerse.

### For Developers (Contributing to MyoVerse)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/NsquaredLab/MyoVerse.git # Replace with your actual repo URL if different
    cd MyoVerse
    ```
2.  **Install uv:** If you don't have it yet, install `uv`. Follow the instructions on the [uv GitHub page](https://github.com/astral-sh/uv).
3.  **Set up Virtual Environment & Install Dependencies:** Use `uv` to create and sync your virtual environment with the project's dependencies.
    ```bash
    uv sync --group dev
    ```
4.  **Install PyTorch with GPU:** After syncing other dependencies, install PyTorch.
    *   **For Windows:** Visit the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and select the appropriate options (Stable, Windows, Pip, Python, your CUDA version). Use `uv` to run the install command, for example:
        ```bash
        # Example for CUDA 12.6 - Get the correct command from PyTorch website!
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --upgrade
        ```
    *   **For Linux:** Pytorch GPU is already installed by MyoVerse.

## What is what?
This project uses the following structure:
- `myoverse`: This is the main package. It contains:
  - `datasets`: Contains data loaders, dataset creators, and a wide array of filters to preprocess your biomechanical data (e.g., EMG, kinematics).
  - `models`: Contains all AI models and their components, ready for training and evaluation.
  - `utils`: Various utilities to support data handling, model training, and analysis.
- `docs`: Contains the source files for the documentation.
- `examples`: Contains practical examples demonstrating how to use the package, including tutorials (`01_tutorials`) and specific use cases like applying filters (`02_filters`).
- `tests`: Contains tests to ensure package integrity and correctness.

## What papers/preprints use this package?
| Journal / Preprint Server                                                                              | DOI                                                                              |
|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| IEEE Transactions on Biomedical Engineering                                                            | [10.1109/TBME.2024.3432800](https://doi.org/10.1109/TBME.2024.3432800)           |
| International Journal of Computer Science and Information Security                                     | [10.33965/ijcsis_2024190101](https://doi.org/10.33965/ijcsis_2024190101)         |
| medRxiv                                                                                                | [10.1101/2024.05.28.24307964](https://doi.org/10.1101/2024.05.28.24307964)       |
| Journal of Neural Engineering                                                                          | [10.1088/1741-2552/ad3498](https://doi.org/10.1088/1741-2552/ad3498)                            |
| IEEE Transactions on Neural Systems and Rehabilitation Engineering                                     | [10.1109/TNSRE.2023.3295060](https://doi.org/10.1109/TNSRE.2023.3295060)         |
| 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC) | [10.1109/EMBC48229.2022.9870937](https://doi.org/10.1109/EMBC48229.2022.9870937) |
