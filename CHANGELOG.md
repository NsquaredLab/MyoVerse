# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/NsquaredLab/MyoVerse/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/NsquaredLab/MyoVerse/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/NsquaredLab/MyoVerse/releases/tag/v0.1.0 