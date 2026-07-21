# spatiomic v1.0 — design & scaffolding

> Status: **design agreed, implementation not started.** This document is the hand-off so v1 work can
> continue from a cold start. It records the diagnosis, the finalized design decisions, the target
> architecture, the migration plan, and a concrete phased task list.
>
> Prerequisite already shipped: the **v0.9.3 correctness patch (PR #24, branch `fix/v0.9.3-correctness`)**.
> v1 builds on top of that (assume those bug fixes are merged before/at the start of v1).

---

## 1. Why v1 — the core problem

`spatiomic` today has **no central data object**. The de-facto working structure is a bare channel-last
`numpy` array (image, `Y×X×C`) threaded alongside **four hand-maintained parallel Python lists**
(`images`, `samples`, `is_disease`, `channels`), with intermediate results round-tripped through `.npy`
files whose **filenames encode the metadata** (e.g. sample id recovered via
`file_path.split("mouse_cgn/")[-1].split(" ")[0]`, condition via `is_disease[samples.index(sample)]`).
`use_gpu` is re-declared at ~123 call sites. AnnData appears only as a terminal export.

This is the single largest source of silent, plausible-but-wrong results (mislabeled channel, wrong
condition flag) and of ergonomic pain (the canonical pipeline is ~130 notebook cells for what is
conceptually load → normalize → SOM → cluster → label → count → test).

### The one architectural insight that shapes everything

**Pixels-as-obs is the wrong central container.** The pipeline is deliberately built to *never* materialize a
pixel table or a pixel graph:

```
subsample ~40k pixels  →  fit ~900-node SOM  →  kNN graph + Leiden on the 900 NODES  →  project back to pixels via som.label
```

Tabling every pixel (an AnnData with `n_obs = H*W`) re-introduces exactly the cost the design avoids
(a 10k×10k×40 FOV ≈ 16 GB dense `X` + a huge `obsp`; a dozen FOVs = hundreds of GB — which is *why* the
tutorial downscales 16×). Foreground masking (`data[binary_mask]`) also breaks any `n_obs == H*W`
reconstruction contract.

So the real axis is **"pixel raster as an IMAGE vs as a TABLE."** The scverse-correct answer:

- **Image world** — the marker stack and the pixel-cluster raster stay **arrays** (numpy now; SpatialData
  `Images(c,y,x)` + `Labels(y,x)` as an optional export). One-FOV-at-a-time streaming is preserved.
- **Table world** — **AnnData** is used *only* where `obs` rows are genuine observations: the **cell** table
  (cells × markers) and node/aggregate summaries.
- **SpatialData** = optional export target, not a core dependency.

---

## 2. Finalized design decisions (decision log)

All decided with the maintainer. Do not relitigate without new information.

| # | Decision | Choice |
|---|---|---|
| 1 | **Central data model** | Image/table split. Lightweight `Fov` + `FovDataset` dataclasses in `spatiomic.core`. **Not** subclasses of AnnData/SpatialData. `to_anndata(level="pixel")` is **per-FOV only** (squidpy interop), never a whole-dataset concat. |
| 2 | **Scale target** | One-FOV-at-a-time streaming + subsample is the contract. **Defer** dask/zarr out-of-core to a later version. |
| 3 | **Namespaces** | scverse `so.io / so.pp / so.tl / so.pl / so.settings`. Old `data/process/dimension/cluster/neighbor/spatial/segment/tool` kept as **deprecation-warning shims** for one cycle (removed in v2.0). |
| 4 | **API style** | **Functions only (scanpy-style).** `FovDataset` is passive data; functions operate on it. In-place mutation returns `None`; standalone results return values. **Exception:** genuinely fitted models stay PascalCase estimators (`so.tl.Som`, `so.pp.Clip`, `so.pp.Normalize`, `so.pp.Pipeline`) with sklearn `fit`/`transform`. |
| 5 | **Backend** | Default `so.settings.backend = "auto"` (GPU if `cupy`/`cuml` import, else CPU) + `with so.backend(...)` context. Two host↔device helpers at edges. `use_gpu=` accepted-but-deprecated (defaults to settings). **Defer** full array-api-compat + GPU CI. |
| 6 | **Cell / segmentation** | `so.tl.segment` / quantify produce a **cell-level AnnData** (obs=cells, X=marker quant, `obsm["spatial"]`=centroids, `obsm["cluster_composition"]`, `obs["community"]`). Whole-dataset variant = concatenated AnnData with `obs["fov_id"]`. Unifies pixel + cell paths. |
| 7 | **Rollout** | v0.9.3 correctness patch (done, PR #24) → v1.0 additive scverse spine + deprecation shims → v2.0 removes shims. Published PathoPlex/Nature env pinned `<1.0`. |

**Non-goals for v1 (explicitly out of scope):** dask/zarr out-of-core; SpatialData as a core/runtime
dependency; full array-api-compat migration; standing up GPU CI; pixels-as-obs as the working object.

---

## 3. Target architecture

### 3.1 Containers — `spatiomic.core`

Lightweight dataclasses reusing AnnData's `obs`/`var`/`uns` vocabulary (so export is trivial and
recognizable), but **not** subclassing AnnData/SpatialData — this preserves the numpy fast path and per-FOV
streaming and adds no heavy deps.

```python
@dataclass
class Fov:                    # one field of view
    image: NDArray           # (Y, X, C) channel-last; dtype preserved (no blind cast)
    var:   pd.DataFrame      # channel names bound to axis C (index = channel name); cannot desync on reorder
    obs:   dict              # per-FOV scalars: fov_id, sample, is_disease, plate, ...
    clusters: NDArray | None # (Y, X) int raster (som.label output)
    masks:    NDArray | None # (Y, X) int segmentation labels (cellpose)
    graph:    "W | None"     # spatial adjacency (libpysal W); neighborhood order recorded in uns
    uns:  dict               # image_shape, neighborhood params, seed, backend, schema_version
    # __post_init__: assert len(var) == C; clusters/masks match (Y, X)

@dataclass
class FovDataset:            # the missing "dataset" object
    fovs: list[Fov]         # shared channel axis (asserted equal); keeps per-FOV identity
    obs:  pd.DataFrame      # ONE row per FOV, index = fov_id — the real join keys (sample, is_disease, ...)
    var:  pd.DataFrame      # single source of truth for channels
    uns:  dict
```

Invariants (enforced in `__post_init__`): channel-axis length matches `len(var)`; rasters match image `Y,X`;
all FOVs in a dataset share the channel axis. `subset_channels(...)` reorders `image` **and** `var`
together (structurally kills the historical `sorted()` channel-name desync bug).

### 3.2 Namespace map

Old namespaces become lazy deprecation-warning shims re-exporting into the new spine (one minor cycle,
removed v2.0).

| new | contents |
|---|---|
| `so.io` | `read_tiff`, `read_qptiff`, `read_czi`, `read_lif` (plain funcs → `Fov` with channels bound; **ends the `read().read_tiff()` empty-instance dance**), `load_dataset(glob, channels_csv=, obs=)`, `to_anndata`, `to_spatialdata`, `read_spatialdata` |
| `so.pp` | `Clip`, `Normalize`, `ZScore`, `Log1p`, `Arcsinh` (PascalCase estimators), `Pipeline`, `subsample`, `register`, `drop_channels`/`subset_channels` |
| `so.tl` | `Som` (single home), `pca`, `umap`, `tsne`, `knn_graph`, `snn_graph`, `leiden`, `kmeans`, `agglomerative`, `label`, `count_clusters`, `count_clusters_per_region`, `segment`, `quantify_markers`, `assign_communities`, `differential_abundance`, `mean_cluster_intensity`, `get_stats` (→ rename param `channel_names` → `feature_names`), `vicinity`, `vicinity_graph`, `spatial_weights`, `autocorrelation` |
| `so.pl` | all plotting (accept `Fov`/`FovDataset`/`AnnData`) |
| `so.settings` | `backend` (`auto`/`cpu`/`gpu`), `seed`, `so.backend(...)` context manager |

### 3.3 Estimator-vs-function rule (resolves the class-as-callable mess)

- **Stateless ops** → plain functions taking the object first: `so.tl.leiden(graph, ...)`,
  `so.tl.count_clusters(ds, ...)`, `so.tl.vicinity(ds, ...)`, `so.io.read_tiff(path)`.
- **Genuinely fitted models** → PascalCase sklearn-style estimators (fit on a subsample, reapply per FOV):
  `so.tl.Som`, `so.pp.Clip`, `so.pp.Normalize`, `so.pp.Pipeline`.
- **Return contract:** functions that annotate the object mutate **in place and return `None`** (write
  `fov.clusters`, add `ds.obs` columns); functions that compute a standalone result **return the value**
  (`differential_abundance` → table; `knn_graph` → graph; `leiden` on a graph → communities array).

### 3.4 Clustering cost model (must be preserved — do not re-table pixels)

`SOM → knn_graph → leiden` runs on the **~900 SOM nodes**, then `so.tl.label(ds, som, comm)` projects node
communities onto pixels in place. Keep this. Any design that builds a pixel-level graph or pixel `obsp` at
full resolution is wrong (see §1).

### 3.5 GPU / backend

- `so.settings.backend ∈ {"auto","cpu","gpu"}` (default `"auto"`); `so.settings.seed`; `with so.backend("cpu"):`.
- Two centralized host↔device helpers at pipeline edges (e.g. `to_backend(x)`, `to_host(x)`); estimators
  that must pick a cuml-vs-sklearn class read the resolved backend, not a per-call bool.
- `use_gpu=` kwargs retained but deprecated (default `None` → settings), so the ~12 tutorial `use_gpu=False`
  calls collapse to one settings line.
- **Do NOT remove `cuml.set_global_output_type("numpy")` from `__init__.py` in v1** until the two-helper
  conversion boundary is implemented **and** validated on GPU. Removing it prematurely would broadly change
  GPU return types (kmeans/pca/umap/nearest-neighbors) — the exact cross-function hazard flagged during the
  v0.9.3 review. Full array-api-compat + GPU CI is a later effort.

### 3.6 Cell / segmentation → AnnData

`so.tl.segment(fov_or_ds, model="cellpose")` returns a **cell-level AnnData**:
`X` = `quantify_markers` (cells × markers), `obsm["spatial"]` = centroids (x, y),
`obsm["cluster_composition"]` = per-cell pixel-cluster histogram, `obs["community"]` = leiden-on-cells,
`obs["instance_id"]` = mask region id. This is the AnnData "Table annotating a Labels region" that
SpatialData formalizes via `region`/`region_key`/`instance_key`, so it promotes losslessly later.

---

## 4. Canonical pipeline — target (functions-only)

```python
import spatiomic as so
# backend defaults to "auto" (gpu if cupy/cuml import, else cpu) — no per-call flags

ds = so.io.load_dataset("data/mouse_cgn/*.tif", input_dimension_order="CYX",
                        channels_csv="channel_names.csv",
                        obs={"is_disease": is_disease})   # channels + metadata bound; no parallel lists
so.pp.drop_channels(ds, "DAPI")                           # in-place, returns None

sub  = so.pp.subsample(ds, n_per_fov=40_000)
pipe = so.pp.Pipeline([so.pp.Clip("minmax", ...), so.pp.Normalize(0, 1)]).fit(sub)
so.pp.transform(ds, pipe)                                 # fit on subsample, apply per-FOV

som  = so.tl.Som((30, 30)).fit(sub)
comm = so.tl.leiden(so.tl.knn_graph(som, neighbor_count=12), resolution=2.5)   # on ~900 nodes
so.tl.label(ds, som, comm)                                # writes fov.clusters (vectorized)

res  = so.tl.differential_abundance(ds, group="is_disease", aggregate_by="sample")   # in-memory, metadata joined
so.pl.volcano(res)

so.tl.vicinity(ds, order=2)
so.pl.spatial_graph(so.tl.vicinity_graph(ds, group="is_disease"))

cells = so.tl.segment(ds["265 g2"], model="cellpose")     # cell-level AnnData (scanpy/squidpy-ready)
```

~12 meaningful lines vs ~130 notebook cells: no parallel lists, no `.npy` message-bus, no reshape juggling,
one backend line.

---

## 5. Migration & deprecation

- **v0.9.3** — correctness patch, no API change. **Done (PR #24).**
- **v1.0** — additive: ship `so.core`, `so.io/pp/tl/pl`, `so.settings`. Old namespaces re-export into the
  new spine via lazy shims emitting `DeprecationWarning` with the exact new path. Estimators accept
  `Fov`/`FovDataset` additively (unwrap `.image` by isinstance-dispatch); bare-ndarray path unchanged.
  `use_gpu=` retained (default `None` → settings).
- **v2.0** — remove the deprecation shims and the empty-instance classes (`Read`/`KnnGraph`/`Subsample`/
  `Register` become function modules); optionally collapse `Som.predict/fit_predict/label`.

Old → new mapping (build the shim table from this):

| old | new |
|---|---|
| `so.data.read().read_tiff(path)` | `so.io.read_tiff(path)` |
| `so.data.subsample().fit_transform(x, ...)` | `so.pp.subsample(x_or_ds, ...)` |
| `so.data.subset(...)` | `so.pp.subset_channels(...)` / `Fov.subset_channels` |
| `so.data.anndata_from_array(...)` | `so.io.to_anndata(fov, level="pixel")` |
| `so.data.array_from_sdata(...)` | `so.io.read_spatialdata(...)` |
| `so.process.clip(...)` | `so.pp.Clip(...)` |
| `so.process.normalize/zscore/log1p/arcsinh` | `so.pp.Normalize/ZScore/Log1p/Arcsinh` |
| `so.process.register(...)` | `so.pp.register(...)` |
| `so.dimension.som` / `so.cluster.som` | `so.tl.Som` (single home) |
| `so.dimension.pca/umap/tsne` | `so.tl.pca/umap/tsne` |
| `so.neighbor.knn_graph().create(...)` | `so.tl.knn_graph(...)` |
| `so.cluster.leiden().predict(graph, ...)` | `so.tl.leiden(graph, ...)` |
| `so.tool.count_clusters(file_paths, ...)` | `so.tl.count_clusters(ds, ...)` (in-memory) |
| `so.segment.count_clusters(img, masks, ...)` | `so.tl.count_clusters_per_region(...)` |
| `so.tool.get_stats(..., channel_names=)` | `so.tl.get_stats(..., feature_names=)` |
| `so.spatial.vicinity_composition/vicinity_graph` | `so.tl.vicinity/vicinity_graph` |

---

## 6. Open sub-questions (maintainer to confirm during implementation)

1. **Container names** — `Fov`/`FovDataset`, or `Image`/`Dataset`, `Field`/`FieldCollection`, or PathoPlex
   vocabulary?
2. **Standardized keys** — proposal: namespace under `uns["spatiomic"]` (`image_shape`, `neighborhood`,
   `seed`, `schema_version`); `obs["clusters"]` for pixel/cell community labels. Any legacy key names to keep
   for continuity?
3. **`differential_abundance` scope** — keep `so.tl.get_stats` as the general engine (~5 modes today:
   cluster-vs-cluster marker stats, disease-vs-control abundance, custom permutation statistics like
   `compare_means`) and make `differential_abundance` a thin convenience over it? (Recommended.)
4. **`subsample` return** — bare array (simplest) vs a lightweight `Fov`-like sample that remembers
   provenance.

---

## 7. Correctness carried in from the v0.9.3 review (context)

Fixed in PR #24 (must be merged before v1): subsample without-replacement; vicinity permutation null
(all-pixel randomization); `assign_communities` fixed `+1` offset; `count_clusters` raise-on-out-of-range;
`register.apply_shift` returns numpy (dead `"cp" in globals()`/`cp.NDArray` guard); `Som.label` vectorized;
phantom `biclustering` removed (+ `__all__` guard test); CI `-m` filter; `spatial_weights` docstring.

Deferred to v1 GPU work (intentionally NOT in v0.9.3): removing `cuml.set_global_output_type("numpy")` — see §3.5.

Not a bug (verified, do not "fix"): `data_method` preserves float64 (only casts when dtype ∉ {f32,f64});
`obsm["spatial"]` meshgrid ordering is internally consistent with the C-order flatten (it is a squidpy
(x,y)-convention mismatch, worth aligning on export, **not** data corruption).

---

## 8. Scaffolding task list (start here)

Phased so each step is independently reviewable. Check off as you go.

### Phase A — containers + settings (foundation, no behavior change to existing API)
- [ ] `spatiomic/core/__init__.py`, `_fov.py`, `_fov_dataset.py` — `Fov`, `FovDataset` dataclasses + invariants.
- [ ] `Fov.subset_channels`, `FovDataset.subset_channels`/`drop_channels` (reorder image + var together).
- [ ] `spatiomic/settings.py` — `settings` singleton (`backend`, `seed`, `strict_gpu`) + `backend(...)` context manager + `resolve()` returning `xp` and the cuml-vs-sklearn selector.
- [ ] Two host↔device helpers (`to_backend`, `to_host`) in `_internal`.
- [ ] Unit tests for container invariants, channel-subset-without-desync, settings/context.

### Phase B — io returning containers
- [ ] `so.io.read_tiff/read_qptiff/read_czi/read_lif` as plain functions → `Fov` (channels bound).
- [ ] `so.io.load_dataset(glob, channels_csv=, obs=)` → `FovDataset`.
- [ ] `so.io.to_anndata(fov, level="pixel"|"cell")` (per-FOV), `to_spatialdata`, `read_spatialdata`.
- [ ] Fix on export: `obsm["spatial"]` in scverse (x, y) order; record neighborhood order in `uns`.

### Phase C — pp / tl accept containers, mutate in place
- [ ] Estimators (`Clip`/`Normalize`/`ZScore`/`Log1p`/`Arcsinh`/`Som`) accept `Fov`/`FovDataset` (unwrap `.image`); `so.pp.Pipeline`.
- [ ] `so.pp.subsample`, `so.pp.transform(ds, pipe)`, `so.pp.register`.
- [ ] `so.tl.knn_graph`, `so.tl.leiden`, `so.tl.label(ds, som, comm)` (writes `fov.clusters`).
- [ ] `so.tl.count_clusters(ds, aggregate_by=)`, `so.tl.differential_abundance(ds, group=, aggregate_by=)` — in-memory, metadata joined off `ds.obs` (no filename parsing).
- [ ] `so.tl.vicinity(ds, order=)`, `so.tl.vicinity_graph(ds, group=)`.

### Phase D — segmentation → cell AnnData
- [ ] `so.tl.segment(fov_or_ds, model="cellpose")` → cell-level AnnData; `quantify_markers`, `assign_communities`, `count_clusters_per_region` feed it.

### Phase E — namespace shims + polish
- [ ] `so.io/pp/tl/pl/settings` public namespaces; PascalCase estimators exported.
- [ ] Deprecation-warning shims for all old namespaces (mapping table in §5); CI test asserting `__all__` == exports.
- [ ] Rename `get_stats` `channel_names` → `feature_names`; single `Som` home.
- [ ] Rewrite tutorials to the §4 target; migration guide; `py.typed` autocomplete preserved through decorators.

### Cross-cutting
- [ ] Keep the SOM-on-nodes cost model (§3.4).
- [ ] Backend default `"auto"`; `use_gpu=` deprecated → settings.
- [ ] Do not remove `set_global_output_type` (§3.5) until GPU boundary is validated.

---

## 9. References

- v0.9.3 correctness PR: https://github.com/complextissue/spatiomic/pull/24
- Full diagnosis + adversarial design analysis lives in the assistant's project memory
  (`scverse-redesign-analysis`): the 6-angle diagnosis, 3 redesign proposals, and 3 adversarial critiques
  that produced the decisions in §2.
