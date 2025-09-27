<p align="center">
  <img src="docs/source/_static/logo.png" alt="spatiomic logo" width="200">
</p>

---

<!--
# spatiomic
This heading is in a comment to maintain semantic structure while using the logo as the visual title
-->

[![Version](https://img.shields.io/pypi/v/spatiomic)](https://pypi.org/project/spatiomic/)
[![License](https://img.shields.io/pypi/l/spatiomic)](https://github.com/complextissue/spatiomic)
[![Python Version Required](https://img.shields.io/pypi/pyversions/spatiomic)](https://pypi.org/project/spatiomic/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![GitHub CI](https://github.com/complextissue/spatiomic/actions/workflows/ci.yml/badge.svg)](https://github.com/complextissue/spatiomic/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/complextissue/spatiomic/branch/main/graph/badge.svg?token=TLXB333GQV)](https://codecov.io/gh/complextissue/spatiomic)
![PyPi Downloads](https://img.shields.io/pepy/dt/spatiomic?label=PyPi%20downloads)

`spatiomic` is a computational library for the analysis of *spati*al prote*omic*s (with some functions also being useful for other _-omics_).
It contains functions for pixel-level clustering, differential cluster abundance analysis, spatial statistics and much more.

The main goal of this package is to organize different packages and methods that are commonly used when dealing with high-dimensional imaging data behind a single API that allows for scalable high-performance computing applications, whenever possible on the GPU.

`spatiomic` has been published as part of `PathoPlex` in Nature: [https://www.nature.com/articles/s41586-025-09225-2](https://www.nature.com/articles/s41586-025-09225-2). It is part of the [scverse® ecosystem](https://scverse.org/packages/#ecosystem).

**📚 Full documentation and tutorials are available at [spatiomic.org](https://spatiomic.org)**

<p align="center">
  <a href="https://spatiomic.org">
    <img src="docs/source/_static/example.png" alt="spatiomic pixel clustering example" width="480">
  </a>
</p>

## Installation

`spatiomic` is available through PyPi:

```bash
uv add spatiomic
```

or

```bash
pip install spatiomic
```

For the best GPU-accelerated experience (optional), a CUDA-compatible GPU and installation of the `cupy`, `cuml`, `cuGraph` and `cuCIM` packages is necessary. You can install them using the `cuda-11` or `cuda-12` extras.

```bash
uv add spatiomic --optional cuda-12
```

or

```bash
pip install "spatiomic[cuda-12]"
```

Alternatively, you may want to install a RAPIDS.AI-enabled Docker container for GPU support, please refer to the [installation guide](https://docs.rapids.ai/install/).

Installation time should not exceed 5 minutes on a standard desktop computer with an average network connection.

## Documentation

Detailled documentation is made available at: [https://spatiomic.org](https://spatiomic.org).

The documentation also contains a small simulated dataset used for clustering, for more information, please refer to the `Pixel-based clustering` section of the documentation.

### Building the documentation

The documentation can be build locally by navigating to the `docs` folder and running: `make html`.
This requires that the development requirements of the package as well as the package itself have been installed in the same virtual environment and that `pandoc` has been added, e.g. by running `brew install pandoc` on macOS operating systems.

## System requirements

### Hardware requirements

`spatiomic` does not come with any specific hardware requirements. For an optimal experience and analysis of very large datasets, a CUDA-enabled GPU and sufficient RAM (e.g., >= 48 Gb) is recommended.

### Software requirements

#### Python version & dependencies

`spatiomic` requires Python version 3.11 or above (3.12 recommended).

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

```bibtex
@article{kuehlPathologyorientedMultiplexingEnables2025,
title = {Pathology-Oriented Multiplexing Enables Integrative Disease Mapping},
author = {Kuehl, Malte and Okabayashi, Yusuke and Wong, Milagros N. and Gernhold, Lukas and Gut, Gabriele and Kaiser, Nico and Schwerk, Maria and Gr{\"a}fe, Stefanie K. and Ma, Frank Y. and Tanevski, Jovan and Sch{\"a}fer, Philipp S. L. and Mezher, Sam and {Sarabia del Castillo}, Jacobo and {Goldbeck-Strieder}, Thiago and Zolotareva, Olga and Hartung, Michael and Delgado Chaves, Fernando M. and Klinkert, Lukas and Gnirck, Ann-Christin and Spehr, Marc and Fleck, David and Joodaki, Mehdi and Parra, Victor and Shaigan, Mina and Diebold, Martin and Prinz, Marco and Kranz, Jennifer and Kux, Johan M. and Braun, Fabian and Kretz, Oliver and Wu, Hui and Grahammer, Florian and Heins, Sven and Zimmermann, Marina and Haas, Fabian and Kylies, Dominik and Wanner, Nicola and Czogalla, Jan and Dumoulin, Bernhard and Zolotarev, Nikolay and Lindenmeyer, Maja and Karlson, Pall and Nyengaard, Jens R. and Sebode, Marcial and Weidemann, S{\"o}ren and Wiech, Thorsten and Groene, Hermann-Josef and Tomas, Nicola M. and {Meyer-Schwesinger}, Catherine and Kuppe, Christoph and Kramann, Rafael and Karras, Alexandre and Bruneval, Patrick and Tharaux, Pierre-Louis and Pastene, Diego and Yard, Benito and Schaub, Jennifer A. and McCown, Phillip J. and Pyle, Laura and Choi, Ye Ji and Yokoo, Takashi and Baumbach, Jan and S{\'a}ez, Pablo J. and Costa, Ivan and Turner, Jan-Eric and Hodgin, Jeffrey B. and {Saez-Rodriguez}, Julio and Huber, Tobias B. and Bjornstad, Petter and Kretzler, Matthias and Lenoir, Olivia and {Nikolic-Paterson}, David J. and Pelkmans, Lucas and Bonn, Stefan and Puelles, Victor G.},
year = {2025},
month = jul,
journal = {Nature},
issn = {1476-4687},
doi = {10.1038/s41586-025-09225-2},
abstract = {The expression and location of proteins in tissues represent key determinants of health and disease. Although recent advances in multiplexed imaging have expanded the number of spatially accessible proteins1--3, the integration of biological layers (that is, cell structure, subcellular domains and signalling activity) remains challenging. This is due to limitations in the compositions of antibody panels and image resolution, which together restrict the scope of image analysis. Here we present pathology-oriented multiplexing (PathoPlex), a scalable, quality-controlled and interpretable framework. It combines highly multiplexed imaging at subcellular resolution with a software package to extract and interpret protein co-expression patterns (clusters) across biological layers. PathoPlex was optimized to map more than 140 commercial antibodies at 80\,nm per pixel across 95 iterative imaging cycles and provides pragmatic solutions to enable the simultaneous processing of at least 40 archival biopsy specimens. In a proof-of-concept experiment, we identified epithelial JUN activity as a key switch in immune-mediated kidney disease, thereby demonstrating that clusters can capture relevant pathological features. PathoPlex was then used to analyse human diabetic kidney disease. The framework linked patient-level clusters to organ disfunction and identified disease traits with therapeutic potential (that is, calcium-mediated tubular stress). Finally, PathoPlex was used to reveal renal stress-related clusters in individuals with type\,2 diabetes without histological kidney disease. Moreover, tissue-based readouts were generated to assess responses to inhibitors of the glucose cotransporter SGLT2. In summary, PathoPlex paves the way towards democratizing multiplexed imaging and establishing integrative image analysis tools in complex tissues to support the development of next-generation pathology atlases.}
}
```
