# PyRecover
Robust distributed checkpointing and job management system for multi-GPU SLURM workloads

## Environment

For setting up environment with miniconda:
```
conda create -n slurcheck python=3.10
conda activate slurcheck

conda install -c conda-forge pytorch torchvision torchaudio cudatoolkit
conda install -c conda-forge pytest pytest-cov black flake8 isort mypy
conda install -c conda-forge sphinx sphinx_rtd_theme
```
