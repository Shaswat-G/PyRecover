from setuptools import setup, find_packages

setup(
    name="pyrecover",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "tqdm",
        "pyyaml",
    ],
    author="Shaswat Gupta",
    author_email="shaswat.gupta.iitb@gmail.com",
    description="Distributed checkpointing manager for SLURM environments",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Shaswat-G/pyrecover",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)