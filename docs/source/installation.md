# Installation

:::{card}
:class-card: sd-bg-warning
:class-body: sd-bg-text-warning
**spatiomic** only supports Python versions greater than or equal to **3.11**. Currently, not all optional dependencies are available for Python 3.13. For the best experience, please use Python 3.12.
:::

:::{card} Recommendation

For the best GPU-accelerated experience (optional), a CUDA-compatible GPU and installation of the `cupy`, `cuml`, `cugraph` and `cucim` packages is necessary. Please consult the [RAPIDS.AI installation guide](https://docs.rapids.ai/install) for further information.
:::

## Installation Options

Choose an option to install this package.

::::{tab-set}

:::{tab-item} uv (Recommended)
Install `spatiomic` package using `uv`:

```bash
uv add spatiomic
```

For the best GPU-accelerated experience (optional), install with CUDA extras:

```bash
# For CUDA 12
uv add spatiomic --optional cuda-12

# For CUDA 11
uv add spatiomic --optional cuda-11
```

For Cellpose segmentation functionality, install with cellpose extras:

```bash
uv add spatiomic --optional cellpose
```

:::

:::{tab-item} pip
Install `spatiomic` package using `pip`:

```bash
pip install spatiomic
```

For the best GPU-accelerated experience (optional), install with CUDA extras:

```bash
# For CUDA 12
pip install "spatiomic[cuda-12]"

# For CUDA 11
pip install "spatiomic[cuda-11]"
```

For Cellpose segmentation functionality, install with cellpose extras:

```bash
pip install "spatiomic[cellpose]"
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
git clone --depth 1 https://github.com/complextissue/spatiomic.git
cd spatiomic
uv sync --dev --extra cellpose
```

:::

::::

:::{dropdown} Additional packages for GPU support
These packages are not required for `spatiomic` to work but may speed certain operations up significantly.

- [cupy](https://docs.cupy.dev/en/stable/index.html) for faster pre-/postprocessing and faster SOM calculations on the GPU.
- [cuml](https://github.com/rapidsai/cuml) for GPU-based AgglomerativeClustering, KMeans, UMAP, TSNE and PCA calculation.
- [cugraph](https://github.com/rapidsai/cugraph) for GPU-based graph operations.
- [cucim](https://github.com/rapidsai/cucim) for GPU-based phase_cross_correlation.

`spatiomic` will always try to perform heavy calculations on the GPU. However, for this to work, a CUDA-enabled system with these packages is required. If these packages are not available, `spatiomic` will default to CPU-based packages such as `numpy` and `sklearn`.

The easiest way to install these GPU dependencies is to use the `cuda-11` or `cuda-12` extras when installing `spatiomic`. Alternatively, you may want to install a RAPIDS.AI-enabled Docker container for GPU support. Please refer to the [RAPIDS installation guide](https://docs.rapids.ai/install/) for more information.
:::
