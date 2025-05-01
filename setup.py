import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

# Default Windows CUDA wheel URLs
TORCH_WIN_URL = "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl"
TORCHVISION_WIN_URL = "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl"

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        self._install_platform_dependencies()

    def _install_platform_dependencies(self):
        # Only handle Windows-specific dependencies if UV_TORCH_BACKEND is not set
        # This allows users to control PyTorch backend selection via the environment variable
        if sys.platform == 'win32' and 'UV_TORCH_BACKEND' not in os.environ:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                TORCH_WIN_URL, TORCHVISION_WIN_URL
            ])

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        self._install_platform_dependencies()
        
    def _install_platform_dependencies(self):
        # Only handle Windows-specific dependencies if UV_TORCH_BACKEND is not set
        if sys.platform == 'win32' and 'UV_TORCH_BACKEND' not in os.environ:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                TORCH_WIN_URL, TORCHVISION_WIN_URL
            ])

if __name__ == '__main__':
    setup(
        cmdclass={
            'install': PostInstallCommand,
            'develop': PostDevelopCommand,
        },
    ) 