# spatiomic

`spatiomic` is a computational library for the analysis of spatial proteomics, mainly via pixel-based clustering, differential cluster abundance analysis and spatial statistics.

The main goal of this package is to organize different packages and methods that are commonly used when dealing with high-dimensional imaging data behind a single API that allows for scalable high-performance computing applications, whenever possible on the GPU.

`spatiomic` has been published as part of `PathoPlex` in Nature: [https://www.nature.com/articles/s41586-025-09225-2](https://www.nature.com/articles/s41586-025-09225-2).

Exemplary pixel-based clustering output:

<div style="width: 80%; max-width: 480px; margin: 0 auto; display: flex; justify-content: center;">
    <img alt="Mouse kidney image" src="./_static/example.png" style="border-radius: 0.25rem;" />
</div>

::::{grid} 2
:gutter: 4
:margin: 5 4 0 0
:padding: 0

:::{grid-item-card} Complex tissue lab
:margin: 0
:link: https://complextissue.com
Our group focuses on the development of novel imaging and computational workflows for spatial omics. Check out our website for more information.
:::

:::{grid-item-card} GitHub
:margin: 0
:link: https://github.com/complextissue/
Our code is published on GitHub, feel free to check it out.
:::

::::

:::{dropdown} Citation
`spatiomic` was developed at [Aarhus University](https://au.dk/) and the [Institute of Medical Systems Biology, Hamburg](https://ims.bio/) by [Malte Kuehl](https://github.com/maltekuehl/) with the help of many contributors. If you use this package in an academic setting, please cite our work according to the information in the `CITATION.cff` file in our [GitHub repository](https://github.com/complextissue/spatiomic/).
:::
