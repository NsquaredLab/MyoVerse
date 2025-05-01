<img src="https://github.com/NsquaredLab/MyoVerse/blob/main/docs/source/_static/myoverse_logo.png?raw=true" height="250">

<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Code style: black" src="https://img.shields.io/badge/python-%3E=3.10,%20%3C=3.13-blue"></a>
<a href="https://www.pytorchlightning.ai/"><img alt="Code style: black" src="https://img.shields.io/badge/uses-pytorch & pytorch lighting-blueviolet"></a>

> [!TIP]
> Dive deeper into our features and usage with the official [documentation](https://nsquaredlab.github.io/MyoVerse/).

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

1.  **Install MyoVerse:** You can install the package from PyPI or directly from GitHub:
    ```bash
    # From PyPI
    pip install MyoVerse
    
    # OR from GitHub (replace `main` with a specific tag/branch if needed)
    pip install git+https://github.com/NsquaredLab/MyoVerse.git
    ```

2.  **Install PyTorch with GPU:** After installing MyoVerse, ensure you have the correct PyTorch version with GPU support:
    *   **For Windows:** You need to install PyTorch with CUDA support separately:
        ```bash
        pip install torch>=2.6.0+cu124 torchvision>=0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --upgrade
        ```
    *   **For Linux:** The standard PyTorch installation (without CUDA index URL) should work fine.

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
