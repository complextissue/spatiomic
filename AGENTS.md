# AGENTS.md

Guidance for AI coding agents working on `spatiomic`, the scverse-ecosystem library
behind PathoPlex (Kuehl et al., *Nature* 2025) for spatial-omics analysis: pixel-level
clustering, spatial statistics, dimensionality reduction and segmentation.

## Golden rules

- **Never perform git actions.** Do not stage, commit, branch, push, or amend. Leave all
  version control to the user. Edit files only.
- **Everything in English.** Code, docstrings, comments, commit-worthy messages.
- **KISS & DRY.** Prefer the simplest correct solution. Do not duplicate logic — factor it out.
- **Reuse before you build.** Before writing new numerical/algorithmic code or adding a
  dependency, check whether an already-present library solves it: `numpy`, `scipy`,
  `scikit-learn`, `scikit-image`, `anndata`, `pandas`, `networkx`, `esda`/`libpysal`
  (PySAL). Read that dependency's own API/docs first. Adding a new dependency is a last
  resort and must be justified to the user.
- **Spell names out in full.** `disease_variant`, not `dis_var`; `neighbor_count`, not `n`.
  Follow existing domain vocabulary in the codebase.

## Commands (via `uv`)

```bash
make check     # ruff check --fix, ruff format --check, mypy -p spatiomic, bandit
make unittest  # uv run coverage run -m pytest --maxfail=10 -m "not gpu and not no_github_ci"
```

**Before handing work back, run `make check` and the CPU test command above; both must
pass.** GPU tests (`-m gpu`) require hardware and are not run locally — do not delete or
weaken them to make a run green.

## Toolchain (do not fight the config)

- **Build/env:** `uv` + `hatchling`. Python **>=3.11** (targets 3.11/3.12/3.13).
- **Ruff:** line length 120, double quotes, Google docstring convention, mccabe
  max-complexity 7. Let ruff format the code — never hand-format around it.
- **mypy:** `disallow_untyped_defs` — every function/method needs full type hints.
- **bandit:** `-ll --recursive spatiomic`.

## Code style

- **Type hints (mandatory).** Use modern syntax in new code: `list[int]`, `tuple[int, int]`,
  `X | None` (not `List`/`Tuple`/`Optional`/`Union`). Use `numpy.typing.NDArray` for arrays
  and `typing.Literal[...]` for enumerated string options.
- **Docstrings (mandatory, Google style)** on every public module, class and function,
  with `Args:`/`Returns:`/`Raises:`. Match the existing convention of repeating the type and
  default in `Args` (e.g. `node_count (tuple[int, int], optional): ... Defaults to (50, 50).`).
- **Comments minimal.** Docstrings carry the documentation. Add an inline comment only where
  logic is genuinely non-obvious — never to restate what the code says.
- **Validation via exceptions**, not `assert`. Raise `ValueError`/`TypeError` with a clear
  message (asserts are stripped under `python -O`):
  ```python
  if data.ndim != 3:
      raise ValueError(f"data must be 3D (y, x, c), got {data.ndim}D")
  ```
- **GPU/CPU dual path.** New array code supports both via the existing pattern: a
  `use_gpu: bool = True` flag and `xp = import_package("cupy", alternative=np)`, then use
  `xp.*`. Reuse `_internal` helpers (`import_package`, `@data_method`, `@anndata_method`).

## Public API pattern

- **Stateful features → an sklearn-style class.** Define a `PascalCase` class in a private
  module `_name.py`, re-export it lowercase in the submodule `__init__.py`
  (`from ._name import Name as name`), and declare it in `spatiomic/__init__.pyi` (the
  package is lazy-loaded). So the public surface reads `so.dimension.som(...)`.
- **Follow the scikit-learn estimator conventions** the codebase already uses:
  - Implement the relevant verbs: `fit(data, ...)`, `transform(data, ...)`,
    `fit_transform(data, ...)`, and `predict(...)` where meaningful. `fit`/`fit_transform`
    take the data as the first positional argument.
  - **All hyperparameters are constructor keyword arguments with sensible defaults** and are
    stored unchanged on `self` (e.g. `self.use_gpu = use_gpu`). Do not mutate them in `fit`.
  - **Learned state is set in `fit`**, conventionally on attributes distinct from the
    hyperparameters (e.g. `self.estimator`, `self.nodes`). `transform`/`predict` rely on it.
  - For persistable estimators, provide `save(save_path)` / `load(load_path)` (pickle-based,
    matching existing classes), plus config accessors like `get_config`/`set_config` where a
    class already establishes that shape.
- **Conform to the submodule's `_base.py` interface.** Each submodule defines ABCs
  (`DimensionReducer`, `LoadableDimensionReducer`, `Processer`) via `abc.ABCMeta` +
  `__subclasshook__` that check for the required methods. A new class must implement those
  methods so it registers as a subclass; extend the interface only if the whole family changes.
- **Pure, stateless helpers → a function**, still exported lowercase
  (e.g. `so.data.subsample(...)`), not a class.

## Testing (high bar — read carefully)

- **A test must prove real correctness or integration.** Do NOT write tests for trivial
  implementation details: existence of attributes that obviously exist, one-liners asserting
  a constant, shape checks with no logic behind them. If a test does not increase confidence
  that the behavior is correct, it should not exist.
- **New behavior in an existing feature → extend that feature's existing tests.** Only create
  a new test module for a genuinely new feature, modeled on the closest existing one.
- Tests live in `test/` mirroring the package (`test/dimension/test_som.py`), functions named
  `test_<name>`. Import the package as `import spatiomic as so` and exercise the public
  lowercase API.
- **Reuse the seeded fixtures in `conftest.py`** (`example_data`, `cycle_pixels`, …) for
  determinism. Add a shared fixture there if broadly useful rather than inlining random arrays.
- **Use `@pytest.mark.parametrize`** for parameter combinations (each case reported/fails
  independently). Use markers `cpu`/`slow`/`gpu`/`no_github_ci` as appropriate.
