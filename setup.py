import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_flash_attention():
    """Execute the setup_flash_attention.sh script to install flash attention."""
    script_name = "setup_flash_attention.sh"
    if not os.path.isfile(script_name):
        raise FileNotFoundError(
            f"Script '{script_name}' not found in the repository root."
        )

    # Run the script
    subprocess.check_call(["bash", script_name])


class InstallCommand(install):
    """Custom install command to build flash attention."""

    def run(self):
        if any("flash-attention" in arg for arg in sys.argv):
            install_flash_attention()
        install.run(self)


class DevelopCommand(develop):
    """Custom develop command to build flash attention."""

    def run(self):
        if any("flash-attention" in arg for arg in sys.argv):
            install_flash_attention()
        develop.run(self)


setup(
    name="pyrecover",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "torchvision",
        "torchaudio",
        "tqdm",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
            "mypy",
            "sphinx",
            "sphinx_rtd_theme",
        ],
    },
    cmdclass={
        "install": InstallCommand,
        "develop": DevelopCommand,
    },
    authors=["Shaswat Gupta", "Raphael Kreft"],
    authors_email=["shaswat.gupta.iitb@gmail.com", "raphaelmkreft@gmail.com"],
    description="Distributed checkpointing manager for SLURM environments",
    long_description=(lambda: open("README.md", "r", encoding="utf-8").read())(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shaswat-G/pyrecover",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)
