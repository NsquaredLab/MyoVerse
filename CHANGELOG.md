# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.4] - 2025-05-01

### Added
- Added description to pyproject.toml

### Changed
- Version bump from 1.1.3 to 1.1.4

## [1.1.3] - 2025-05-01

### Added
- Package `__version__` attribute in `myoverse/__init__.py` that dynamically reads from pyproject.toml 
- Version number included in logging timestamp prefix

### Changed
- PyPI compatible dependency specifications for PyTorch with separate configurations for Linux and Windows
- Updated installation instructions in README for clarity on platform-specific PyTorch installations
- Enhanced GitHub links in project.urls section

## [1.1.2] - 2025-05-01

### Changed
- Updated README logo URL to use raw GitHub link for better cross-platform compatibility
- Revised Lightning to version 2.5.0.post0
- Removed PyQt5 and related packages from development dependencies
- Streamlined dependency list while maintaining compatibility

## [1.1.1] - 2025-04-05

### Changed
- Removed unnecessary string handling in `EMGDataset` class
- Mean and standard deviation are no longer nn.Parameter objects in `RaulNetV17` model. They should have been regular tensors
- Fix version of `lightning` to 2.5.0.post0 in `pyproject.toml` to avoid bug with `mlflow` integration

## [1.1.0] - 2025-04-03

### Added
- RaulNetV17 model for decoding kinematics from EMG data
- Workflow module for EMG data processing, model training, and result visualization
- New reStructuredText files for data types and functions documentation
- Unit tests for EMG data augmentation filters and default datasets

### Changed
- Refactored code for improved readability and consistency in optimizer configuration, visualization functions, and data class initializations
- Updated test data class to include dimensions when unchunked
- Downgraded zarr version from 3 to 2 in loader and supervised modules
- Refactored optimizer configuration and updated ground truth handling in training steps
- Refactored EMG and kinematics data classes to include dimensions when unchunked and streamlined chunk checking logic
- Enhanced kinematics visualization with interactive features and improved styling
- Updated zarr version and added mlflow and pyqt5 dependencies
- Renamed `AveragingSpatialFilter` to `ApplyFunctionSpatialFilter` in documentation (`filters.rst`).
- Refactored spatial filters (`spatial.py`) simplifying parameters and type hints.
- Refactored `SOSFrequencyFilter` for easier usability.
- Enhanced grid layout documentation and validation in `EMGData` class.

### Removed
- `ElectrodeSelector` and `GridReshaper` filters mentioned in documentation (`filters.rst`).
- Unnecessary spatial filters.

## [1.0.0] - 2024-05-20

### Added
- Examples for documentation
- Sections for Filters and Tutorials in README
- Example scripts for training models and dataset creation
- Enhanced documentation structure with autosummary for datasets and filters
- Unit tests for core functionality

### Changed
- Bump version to 1.0.0 and add pydata-sphinx-theme dependency
- Updated model definitions to indicate archival status
- Replaced pytorch_lightning with lightning
- Enhanced documentation for VirtualHandKinematics class by adding reference to MyoGestic
- Refactored activation function documentation and improved loss calculation formatting
- Enhanced dataset creation and loading by improving EMG dataset handling
- Added empty data checks and refined augmentation pipelines
- Refactored dataset creation with new CustomDataClass
- Enhanced EMG dataset loading to support Zarr files with improved validation
- Enhanced filter configurations by adding input_is_chunked parameter
- Improved EMGAugmentation class with additional parameters and better documentation
- Improved spatial filter API
- Enhanced filter methods to accept additional keyword arguments
- Enhanced layout algorithm for EMG visualizations with topological sorting and improved node positioning

## [0.1.0] - 2023-10-01

### Added
- Core EMG signal processing functionality with preprocessing filters
- Data loaders for biomechanical data
- PyTorch and PyTorch Lightning integration
- AI models for EMG analysis and movement prediction
- Kinematics and kinetics computation tools
- Visualization utilities
- Example notebooks for common use cases
- Initial documentation and API reference

## Publications

The following research papers have utilized MyoVerse:

- IEEE Transactions on Biomedical Engineering (2024): [10.1109/TBME.2024.3432800](https://doi.org/10.1109/TBME.2024.3432800)
- International Journal of Computer Science and Information Security (2024): [10.33965/ijcsis_2024190101](https://doi.org/10.33965/ijcsis_2024190101)
- medRxiv (2024): [10.1101/2024.05.28.24307964](https://doi.org/10.1101/2024.05.28.24307964)
- Journal of Neural Engineering (2024): [10.1088/1741-2552/ad3498](https://doi.org/10.1088/1741-2552/ad3498)
- IEEE Transactions on Neural Systems and Rehabilitation Engineering (2023): [10.1109/TNSRE.2023.3295060](https://doi.org/10.1109/TNSRE.2023.3295060)
- 2022 44th Annual International Conference of the IEEE EMBC: [10.1109/EMBC48229.2022.9870937](https://doi.org/10.1109/EMBC48229.2022.9870937)

[Unreleased]: https://github.com/NsquaredLab/MyoVerse/compare/v1.1.4...HEAD
[1.1.4]: https://github.com/NsquaredLab/MyoVerse/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/NsquaredLab/MyoVerse/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/NsquaredLab/MyoVerse/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/NsquaredLab/MyoVerse/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/NsquaredLab/MyoVerse/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/NsquaredLab/MyoVerse/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/NsquaredLab/MyoVerse/releases/tag/v0.1.0 