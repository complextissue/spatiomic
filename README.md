# spatiomic

[![Version](https://img.shields.io/pypi/v/spatiomic)](https://pypi.org/project/spatiomic/)
[![License](https://img.shields.io/pypi/l/spatiomic)](https://github.com/complextissue/spatiomic)
[![Python Version Required](https://img.shields.io/pypi/pyversions/spatiomic)](https://pypi.org/project/spatiomic/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub CI](https://github.com/complextissue/spatiomic/actions/workflows/ci.yml/badge.svg)](https://github.com/complextissue/spatiomic/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/complextissue/spatiomic/actions/workflows/docs.yml/badge.svg)](https://spatiomic.complextissue.com)
[![codecov](https://codecov.io/gh/complextissue/spatiomic/branch/main/graph/badge.svg?token=TLXB333GQV)](https://codecov.io/gh/complextissue/spatiomic)
![PyPi Downloads](https://img.shields.io/pepy/dt/spatiomic?label=PyPi%20downloads)

`spatiomic` is a computational library for the analysis of *spati*al prote*omic*s (with some functions also being useful for other *-omics*).

The main goal of this package is to organize different packages and methods that are commonly used when dealing with high-dimensional imaging data behind a single API that allows for scalable high-performance computing applications, whenever possible on the GPU.

## Installation

`spatiomic` is available through PyPi:

```bash
pip install spatiomic
```

For the best GPU-accelerated experience (optional), a CUDA-compatible GPU and installation of the `cupy`, `cuml`, `cuGraph` and `cuCIM` packages is necessary. Please consult the [RAPIDS.AI installation guide](https://docs.rapids.ai/install) for further information.

Please note that this is a preview version. Local installation can be done in a virtual environment by first running `make create-venv`, followed by `source .venv/bin/activate` to switch to the new virtual environemnt and `make install` or `make install-dev` (if packages for code quality checks, unittesting and documentation building are desired) at the root of this repository.

Installation time should not exceed 5 minutes on a standard desktop computer with an average network connection.

## Documentation

Detailled documentation is made available at: [https://spatiomic.complextissue.com](https://spatiomic.complextissue.com).

The documentation also contains a small simulated dataset used for clustering, for more information, please refer to the `Pixel-based clustering` section of the documentation.

### Building the documentation

The documentation can be build locally by navigating to the `docs` folder and running: `make html`.
This requires that the development requirements of the package as well as the package itself have been installed in the same virtual environment and that `pandoc` has been added, e.g. by running `brew install pandoc` on macOS operating systems.

## System requirements

### Hardware requirements

`spatiomic` does not come with any specific hardware requirements. For an optimal experience and analysis of very large datasets, a CUDA-enabled GPU and sufficient RAM (e.g., >= 48 Gb) is recommended.

### Software requirements

#### Operating systems

Though it should run on all systems that can run Python, `spatiomic` has specifically been confirmed to work on the following operating systems:

- Ubuntu 22.04
- Ubuntu 24.04
- macOS Sequoia 15.1.1

#### Python version & dependencies

`spatiomic` requires Python version 3.10 or above (3.12 recommended).

#### Code editors

We recommend developers use Visual Studio Code with the recommended extensions and settings contained in the `.vscode` folder to edit this codebase.

### GPUs

The use of a GPU is optional but greatly accelerates many common `spatiomic` analyses. While most recent CUDA-compatible devices are expected to work, the following GPUs have been tested:

- NVIDIA RTX 6000 Ada
- NVIDIA QUADRO RTX 8000
- NVIDIA V100

Using a modern computer (e.g., an M-series MacBook) without a CUDA-enabled GPU, the sample script provided in the `Full example` section of the documentation should take a few minutes, depending on your hardware, typically less than 3 minutes if all the data is already downloaded and the package is installed. With a CUDA-enabled GPU, it should be significantly faster.

## Attribution & License

### License

The software is provided under the GNU General Public License, version 3 (GPL-3.0). Please consult `LICENSE.md` for further information.
The `glasbey_light` color palette available through `so.plot.colormap` is part of `colorcet` and distributed under the Creative Commons Attribution 4.0 International Public License (CC-BY).

### Citation

`spatiomic` was developed for use with multiplexed immunofluorescence imaging data at [Aarhus University](https://au.dk/) by [Malte Kuehl](https://github.com/maltekuehl) with valuable inputs, code additions and feedback from other lab members, supervisors and collaborators. If you use this package in an academic setting, please cite this repository according to the information in the `CITATION.cff` file.
