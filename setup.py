import os
import subprocess
import sys
from setuptools import setup, find_packages, Command
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_flash_attention():
    """Install flash attention by building cuda kernels"""
    flash_attention_repo = "flash-attention"
    # Clone the repository
    subprocess.check_call(
        ["git", "clone", "https://github.com/Dao-AILab/flash-attention.git"]
    )
    # First install hopper
    hopper_dir = os.path.join(flash_attention_repo, "hopper")
    if os.path.exists(hopper_dir):
        subprocess.check_call([sys.executable, "setup.py", "install"], cwd=hopper_dir)
    # Clean up
    subprocess.check_call(["rm", "-rf", flash_attention_repo])


class InstallCommand(install):
    """Custom install command to build flash attention"""
    def run(self):
        if any('flash-attention' in arg for arg in sys.argv):
            install_flash_attention()
        install.run(self)


class DevelopCommand(develop):
    """Custom install command to build flash attention"""
    def run(self):
        if any('flash-attention' in arg for arg in sys.argv):
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
        'flash-attention': ['flash-attention @ git+https://github.com/Dao-AILab/flash-attention.git'],
        'dev': [
            'pytest',
            'pytest-cov',
            'black',
            'flake8',
            'isort',
            'mypy',
            'sphinx',
            'sphinx_rtd_theme',
        ],
    },
    cmdclass={
        'install': InstallCommand,
        'develop': DevelopCommand,
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