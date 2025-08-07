# Workflow overview

Here, you can find high-level descriptions of the steps `spatiomic` facilitates to go from raw data to biological insights. This workflow description is specifically intended for pixel-level clustering of spatial proteomics data. If your use case differs, please consult the API documentation and other examples instead.

You can find an implementation of this workflow with real data in the [full example notebook](./tutorials/full_example).

## Workflow Diagram

The flowchart below outlines the standard parts of a `spatiomic` analysis.

```{mermaid}
graph TD

    A1[Raw data] -- Registration, QC & segmentation --> A2[Imaging data];
    A2 --> B1{Weighted subsampling};
    B1 --> B2[Subsample];

    C1[Preprocessors] --> D1{Fitting};
    D1 --> C2[Fitted preprocessors];
    C2 --> D2{Transforming};
    C2 --> D3{Transforming};
    B2 --> D1;
    B2 --> D2;
    A2 --> D3;
    D2 --> E[Transformed subsample];
    D3 --> F[Transformed imaging data];

    subgraph Loading and processing
        A1; A2; B1; B2; Q; C1; C2; D1; D2; D3; E; F;
    end;

    G[Self-organizing map] --> H{Training};
    E --> H;
    H --> I[Fitted self-organizing map];
    I -- kNN graph of SOM nodes--> J[Nearest neighbors graph];
    J -- Leiden clustering --> K[SOM node clusters];
    K --> M{Label transfer};
    M --> N[Clustered imaging data];
    F --> M;

    subgraph Clustering
        G; H; I; J; K; M; N;
    end;

    K -- Statistical testing --> L[Cluster contributors];
    N --> O[Spatial cluster projections];
    O --> P[Cluster interpretation];
    L --> P;

    subgraph Interpretation
        O; P; L; R; S; T;
    end;

    Q[Group labels] --> R{Statistical testing};
    Q --> B1;
    N --> R;
    R --> S[Differential abundance];


    N --> T[Spatial statistics];

    P --> U[Biological hypotheses];
    S --> U;
    T --> U;
    U --> V[Confirmatory experiments];
    V --> W[Biological insights];

    subgraph Results
        U; V; W;
    end;
```

## Explanation

This section provides a detailed explanation of each step in the `spatiomic` workflow for pixel-level clustering of spatial proteomics data.

### Loading and Processing Data

#### Raw Data and Image Registration

Raw data from spatial proteomics experiments typically consists of multi-channel tiff files obtained from cyclic imaging protocols. Before analysis, these images often require:

- **Registration**: Alignment of images across cycles to ensure pixel correspondence
- **Quality control**: Assessment of signal quality, channel bleed-through, and imaging artifacts
- **Segmentation** (optional): Delineation of foreground from background or cell segmentation

The `spatiomic.data.read` class provides methods for parsing common microscopy formats (.tiff, .lif, .czi) via readlif, tifffile, and aicspylibczi bindings. The `process.register` class offers methods for image registration and registration evaluation.

#### Weighted Subsampling

Working with all pixels from all images is computationally intensive. To ensure representative sampling while reducing computational burden:

- The `data.subsample` class enables random subsampling
- Subsampling should be stratified by experimental condition, sample, and field of view

#### Preprocessing

Preprocessing standardizes immunofluorescence intensities across markers and samples:

- **Histogram clipping**: The `process.clip` class trims intensity values to exclude outliers
  - Can use percentile-based thresholds (e.g., 50th and 99.7th percentiles)
  - Can use absolute thresholds determined by expert annotation
- **Normalization**: The `process.normalize` class rescales intensities to a standard range
- **Z-scoring** (optional): The `process.zscore` class standardizes each channel

Preprocessing classes are first fitted on the subsample, then applied to all images to ensure consistent transformations.

### Clustering

#### Self-Organizing Maps

Self-organizing maps (SOMs) provide dimensionality reduction and improved representation of rare signals:

- The `dimension.som` class implements GPU-accelerated SOM training and scales to large SOM sizes
- Training occurs iteratively (typically 25-50 iterations) on the transformed subsample

#### Graph Construction and Clustering

Clustering identifies pixel groups with similar protein co-expression patterns:

- The `neighbor.knn_graph` class constructs a k-nearest neighbors graph of SOM nodes
  - Optionally supports batch integration for multi-plate experiments
- The `cluster.leiden` class applies Leiden community detection to identify clusters

#### Label Transfer

Once SOM nodes are clustered:

- Cluster assignments are transferred to all pixels in all images
- Each pixel is assigned to its closest SOM node's cluster
- This enables spatial projection of clusters for biological interpretation

### Interpretation

#### Cluster Contributors

To understand what each cluster represents:

- The `tool.get_stats` function identifies statistically significant markers for each cluster
- Markers are ranked by mean intensity and log2 fold change relative to other clusters
- High contributors are markers with high mean intensity, significant p-values, and large fold changes

#### Spatial Projection and Interpretation

Clusters are interpreted through a combination of:

- Spatial projection of clusters onto original images using `plot.cluster_image`
- Assessment of cluster-specific markers using `plot.cluster_contributors`
- Visualization of marker distribution within clusters using `plot.cluster_contributor_histogram`
- Expert biological knowledge integrating marker expression with spatial context

#### Differential Abundance Analysis

To identify differences between experimental conditions:

- The `tool.count_clusters` function quantifies cluster abundance across samples
- The `tool.get_stats` function calculates statistical significance of abundance differences
- Results can be visualized with `plot.volcano`
- This identifies which protein co-expression patterns are altered between conditions

#### Spatial Statistics

To investigate spatial relationships between clusters:

- The `spatial` submodule provides tools for spatial analysis, including:
  - Vicinity composition analysis using `spatial.vicinity_composition`
  - Spatial graph construction with `spatial.vicinity_graph`
  - Visualization of spatial relationships using `plot.spatial_graph`
  - Integration with PySAL for advanced spatial statistics

### Results

The insights gained from these analyses enable:

- Formulation of biological hypotheses about cellular and molecular processes
- Design of confirmatory experiments to validate findings
- Understanding of complex tissue organization and disease mechanisms

For a detailed implementation of this workflow with real data, see the [full example notebook](./tutorials/full_example).
