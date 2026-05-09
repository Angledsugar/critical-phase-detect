# T3: LIBERO install notes

## Status

`uv sync --extra libero` reports success but the resulting `libero==0.1.0`
wheel installed into `.venv/lib/python3.11/site-packages/` is **empty**:
only `libero-0.1.0.dist-info/` is present, no `libero/` Python package
directory. Importing fails:

```
$ uv run python -c "import libero"
ModuleNotFoundError: No module named 'libero'
```

## Why it fails

LIBERO upstream uses a legacy `setup.py` at the repo root with
`find_packages()` that expects to be invoked from the repo root. The PEP-517
build (sdist → wheel) used by `uv pip install` from `git+...` runs in an
isolated build env where `find_packages()` discovers no packages because the
real source (`libero/libero/`) is one level deeper than `find_packages`'
default search root, and the repo provides no `pyproject.toml` to override.

In addition, runtime requires:
- `robosuite` + `mujoco` (heavy native deps)
- BDDL parser
- Downloaded hdf5 demo datasets (not bundled)

## Fallback (manual install, headed machine only)

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /opt/LIBERO
cd /opt/LIBERO
pip install -r requirements.txt
pip install -e .
# Download demos (~10 GB):
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

Set `LIBERO_CONFIG_PATH` if you don't want config under `~/.libero`.

## Effect on this codebase

Tests in `tests/envs/test_libero.py` open with
`pytest.importorskip("libero.libero")`, so the rest of the test suite passes
unchanged on installs where LIBERO is missing or broken. When LIBERO is
properly installed, the env smoke test exercises `reset()` + `step()`; the
demo smoke test additionally requires the hdf5 datasets and skips if the
default `datasets/` folder is empty.

## Smoke check, end-to-end (after manual install + dataset download)

```python
from cpd.envs.libero import LiberoEnv
env = LiberoEnv(suite="libero_long", task_id=0, image_size=128)
obs = env.reset()
next_obs, r, done, info = env.step([0.0] * 7)
env.close()
```
