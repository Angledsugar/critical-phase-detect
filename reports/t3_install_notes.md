# T3: LIBERO install notes

## Status (2026-05-09 update)

`import libero.libero` 가 venv 안에서 동작합니다. 다만 실제 env.reset() /
env.step() 까지 가려면 여전히 robosuite + mujoco + bddl + 데이터셋이 필요합니다.

### Working setup (main venv, import-only)

`scripts/setup_libero_venv.sh` 가 별도 venv (`.venv-libero`) 에서 import + 환경
실행까지 한번에 셋업. 메인 venv 는 import 만 동작 (env 객체는 robosuite/mujoco
미설치라 `LiberoEnv.reset()` 시점에 깨짐).

### Dataset

`/media/engineer/DATA/datasets/libero/libero_10/` 에 LIBERO-Long 10 task
hdf5 (각 50 demos, 13 GB) 가 위치. `~/.libero/config.yaml` 의 `datasets`
경로가 거기로 가리키도록 설정되어 있음. UTexas Box 원본 링크는 만료되어 (404)
HF mirror `yifengzhu-hf/LIBERO-datasets` 에서 다음 코드로 받음:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
    local_dir="/media/engineer/DATA/datasets/libero",
    allow_patterns=["libero_10/*"],  # libero_spatial / object / goal 추가 시 패턴 확장
)
```

ablation 추가 시 다른 suite 도 같은 방식으로 받기. libero_90 (60 GB) 은
lifelong baseline 용이라 우리 paper 에는 불필요.

수동 셋업 (메인 venv 에서 import 만 필요한 경우):


1. `.pth` 파일로 namespace package 등록:

   ```
   .venv/lib/python3.11/site-packages/libero.pth:
     /home/engineer/openpi/third_party/libero
   ```

   (openpi submodule으로 이미 클론된 LIBERO 소스를 가리킴)

2. `~/.libero/config.yaml` 미리 생성 — 첫 import 시 input() prompt 우회:

   ```yaml
   benchmark_root: /home/engineer/openpi/third_party/libero/libero/libero
   bddl_files:     /home/engineer/openpi/third_party/libero/libero/libero/bddl_files
   init_states:    /home/engineer/openpi/third_party/libero/libero/libero/init_files
   datasets:       /home/engineer/openpi/third_party/libero/libero/libero/../datasets
   assets:         /home/engineer/openpi/third_party/libero/libero/libero/assets
   ```

3. 검증:
   ```
   $ uv run python -c "import libero.libero; print(libero.libero.__file__)"
   /home/engineer/openpi/third_party/libero/libero/libero/__init__.py
   ```

### Why `uv sync --extra libero` 빈 wheel을 만들었는가

`uv pip install --no-deps -e .../libero` 를 직접 호출해도 결과는 같음.
`__editable___libero_0_1_0_finder.py` 안의 `MAPPING: dict[str, str] = {}` 가
빈 채로 등록되고 `top_level.txt` 도 1바이트(개행만). 이유:

LIBERO upstream uses a legacy `setup.py` at the repo root with
`find_packages()`. PEP-517 build runs `find_packages()` from a search root
where the actual `libero/libero/__init__.py` is at depth 2, and the outer
`libero/` directory has no `__init__.py` (it is a PEP 420 namespace package).
The build's `find_packages()` invocation registers nothing into top_level.txt,
so the resulting wheel installs metadata but no path mapping. The `.pth`
fallback bypasses this entirely by adding the repo root to `sys.path` so
Python's import machinery resolves `libero` as a namespace package and
`libero.libero` as a regular package.

Still missing for actual env runtime:
- `robosuite` + `mujoco` (heavy native deps, GLFW/EGL)
- `bddl` parser
- Downloaded hdf5 demo datasets (~10 GB, not bundled)

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
