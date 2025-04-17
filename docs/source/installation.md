# Installation

:::{card}
:class-card: sd-bg-warning
:class-body: sd-bg-text-warning
**spatiomic** only supports Python versions greater than or equal to **3.10**. Currently, not all optional dependencies are available for Python 3.13. For the best experience, please use Python 3.10, 3.11 or 3.12 (recommended).
:::


:::{card} Recommendation

For the best GPU-accelerated experience (optional), a CUDA-compatible GPU and installation of the `cupy`, `cuml`, `cugraph` and `cucim` packages is necessary. Please consult the [RAPIDS.AI installation guide](https://docs.rapids.ai/install) for further information.
:::


## Installation Options

Choose an option to install this package.

::::{tab-set}

:::{tab-item} PyPi
Install `spatiomic` package using `pip`:

```bash
python3 -m pip install spatiomic

# optionally, for GPU acceleration with CUDA 12
python3 -m pip install cupy-cuda12x
python3 -m pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu12==25.2.*" "cugraph-cu12==25.2.*" "nx-cugraph-cu12==25.2.*" \
    "cucim-cu12==25.2.*"

# optionally, for GPU acceleration with CUDA 11
python3 -m pip install cupy-cuda11x
python3 -m pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cuml-cu11==25.2.*" "cugraph-cu11==25.2.*" "nx-cugraph-cu11==25.2.*" \
    "cucim-cu11==25.2.*"
```
:::

:::{tab-item} GitHub
Install `spatiomic` from GitHub using `pip`:

```bash
python3 -m pip install git+git@github.com:complextissue/spatiomic.git
```
:::

:::{tab-item} Source
Install `spatiomic` from source:

```bash
# Clone repo
git clone --depth 1 https://github.com/complextissue/spatiomic.git
cd spatiomic
make install
```
:::

::::

:::{dropdown} Additional packages for GPU support
These packages are not required for `spatiomic` to work but may speed certain operations up significantly.

- [cupy](https://docs.cupy.dev/en/stable/index.html) for faster pre-/postprocessing and faster SOM calculations on the GPU.
- [cuml](https://github.com/rapidsai/cuml) for GPU-based AgglomerativeClustering, KMeans, UMAP, TSNE and PCA calculation.
- [cucim](https://github.com/rapidsai/cucim) for GPU-based phase_cross_correlation.

`spatiomic` will always try to perform heavy calculations on the GPU. However, for this to work, a CUDA-enabled system with `cupy` and the RAPIDS package `cuml` is required and have to be installed beforehand. If these packages are not available, `spatiomic` will default to CPU-based packages such as `numpy` and `sklearn`.

Note that `cupy` and `cucim` can be installed with the \[gpu\] flag when installing via `pip`. `cuml` however is not available as a PyPi package and thus has to be installed by the user of this package, for example with the provided `Dockerfile`.
:::
