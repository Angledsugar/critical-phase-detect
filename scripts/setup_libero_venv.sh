#!/usr/bin/env bash
# LIBERO 전용 별도 venv 셋업.
#
# 메인 venv (.venv) 는 cpd + torch 2.3 + openpi (numpy 1.26 / hydra 1.3 stack) 으로
# 유지하고, LIBERO 의 deprecated stack (numpy 1.22 / hydra 1.2 / gym 0.25 /
# robosuite + mujoco) 은 .venv-libero 에 격리한다. cpd 본체는 main venv 에서 돌고,
# LIBERO 환경 호출 (rollout 수집, env smoke test) 은 .venv-libero 의 python 으로
# subprocess 또는 standalone script 형태로 실행한다.
#
# Usage:
#   bash scripts/setup_libero_venv.sh                  # env-only (default)
#   bash scripts/setup_libero_venv.sh --rebuild        # remove .venv-libero first
#   bash scripts/setup_libero_venv.sh --skip-mujoco    # skip mujoco install
#   bash scripts/setup_libero_venv.sh --with-lifelong  # + transformers/wandb/hydra (학습 baseline 용)
#
# Note:
#   기본 모드는 LIBERO env 만 동작하는 minimum set.
#   LIBERO 의 requirements.txt 통째로 설치 시 transformers==4.21.1 + tokenizers==0.12.1
#   prebuilt wheel 부재로 Rust 컴파일러 빌드 실패 → 학습 baseline (lifelong/) 이
#   필요한 경우에만 --with-lifelong 으로 활성화.
#
# Env overrides:
#   LIBERO_SRC   default: /home/engineer/openpi/third_party/libero
#   VENV_DIR     default: .venv-libero (project-local)
#   PYTHON_VER   default: 3.11

set -euo pipefail

LIBERO_SRC="${LIBERO_SRC:-/home/engineer/openpi/third_party/libero}"
VENV_DIR="${VENV_DIR:-.venv-libero}"
PYTHON_VER="${PYTHON_VER:-3.11}"

REBUILD=0
SKIP_MUJOCO=0
WITH_LIFELONG=0
for arg in "$@"; do
    case "$arg" in
        --rebuild)        REBUILD=1 ;;
        --skip-mujoco)    SKIP_MUJOCO=1 ;;
        --with-lifelong)  WITH_LIFELONG=1 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '\033[1;36m[libero-venv]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[libero-venv]\033[0m %s\n' "$*" >&2; }

# --- 1. preflight ---------------------------------------------------------
if [[ ! -d "$LIBERO_SRC/libero/libero" ]]; then
    err "LIBERO source not found at: $LIBERO_SRC"
    err "  expected layout: \$LIBERO_SRC/libero/libero/__init__.py"
    err "  override with LIBERO_SRC=/path/to/LIBERO bash $0"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    err "uv not found in PATH (install: https://docs.astral.sh/uv/)"
    exit 1
fi

if [[ "$REBUILD" == "1" && -d "$VENV_DIR" ]]; then
    log "removing existing $VENV_DIR (--rebuild)"
    rm -rf "$VENV_DIR"
fi

# --- 2. create venv -------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    log "creating venv at $VENV_DIR (python $PYTHON_VER)"
    uv venv "$VENV_DIR" --python "$PYTHON_VER" --prompt libero
else
    log "reusing existing venv: $VENV_DIR"
fi

VPY="$VENV_DIR/bin/python"
PIP_INSTALL=(uv pip install --python "$VPY")

# --- 3. install LIBERO env-only deps -------------------------------------
# requirements.txt 통째로는 안 쓰는 이유: transformers==4.21.1 / thop 등 학습 전용
# deps 가 prebuilt wheel 없어서 source build 실패. env smoke 에는 불필요.
log "installing LIBERO env-only deps (numpy / gym / bddl / robomimic / robosuite)"
"${PIP_INSTALL[@]}" \
    "numpy==1.22.4" \
    "gym==0.25.2" \
    "bddl==1.0.1" \
    "robomimic==0.2.0" \
    "robosuite==1.4.0" \
    "easydict==1.9" \
    "cloudpickle==2.1.0" \
    "opencv-python==4.6.0.66" \
    "future==0.18.2" \
    "einops==0.4.1" \
    "matplotlib<3.10" \
    "pyyaml"

if [[ "$WITH_LIFELONG" == "1" ]]; then
    log "installing lifelong-baseline deps (transformers / hydra / wandb / matplotlib)"
    # transformers 4.30+ 는 python 3.11 prebuilt wheel 있음. LIBERO/lifelong 코드와
    # API 차이 발생 시 사용자가 직접 코드 패치 필요.
    "${PIP_INSTALL[@]}" \
        "transformers>=4.30,<4.40" \
        "hydra-core==1.2.0" \
        "wandb==0.13.1" \
        "matplotlib==3.5.3" \
        "thop==0.1.1.post2209072238"
fi

# mujoco native binary (heavy, optional via --skip-mujoco for CI)
if [[ "$SKIP_MUJOCO" == "0" ]]; then
    log "installing mujoco (native, EGL/GLFW required at runtime)"
    "${PIP_INSTALL[@]}" "mujoco>=2.3,<3.0"
else
    log "skipping mujoco (--skip-mujoco)"
fi

# --- 4. register LIBERO via .pth (PEP-517 빈 wheel 우회) -----------------
SITE_PKG="$($VPY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
log "writing $SITE_PKG/libero.pth -> $LIBERO_SRC"
echo "$LIBERO_SRC" > "$SITE_PKG/libero.pth"

# --- 5. ~/.libero/config.yaml (skip input() prompt on first import) -------
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
mkdir -p "$LIBERO_CONFIG_DIR"
PKG_ROOT="$LIBERO_SRC/libero/libero"
if [[ ! -f "$LIBERO_CONFIG_DIR/config.yaml" ]]; then
    log "writing $LIBERO_CONFIG_DIR/config.yaml"
    cat > "$LIBERO_CONFIG_DIR/config.yaml" <<EOF
benchmark_root: $PKG_ROOT
bddl_files: $PKG_ROOT/bddl_files
init_states: $PKG_ROOT/init_files
datasets: $PKG_ROOT/../datasets
assets: $PKG_ROOT/assets
EOF
else
    log "config.yaml already exists at $LIBERO_CONFIG_DIR/config.yaml (kept)"
fi

# --- 6. smoke test --------------------------------------------------------
log "smoke test: import + benchmark registry"
"$VPY" - <<'PY'
import libero.libero
from libero.libero.benchmark import BENCHMARK_MAPPING, get_benchmark
suites = sorted(BENCHMARK_MAPPING.keys())
print(f"libero.libero  -> {libero.libero.__file__}")
print(f"benchmark suites ({len(suites)}): {suites}")
# instantiate one to make sure task list loads
bm = get_benchmark("libero_10")()
print(f"libero_10 tasks: {bm.n_tasks}")
PY

cat <<EOF

[done] LIBERO venv ready at: $VENV_DIR

Next steps:
  1. dataset download (~10 GB):
       $VPY $LIBERO_SRC/benchmark_scripts/download_libero_datasets.py --use-huggingface

  2. env smoke (headless, EGL backend):
       MUJOCO_GL=egl $VPY -c "
       from libero.libero.envs import OffScreenRenderEnv
       from libero.libero.benchmark import get_benchmark
       bm = get_benchmark('libero_long')()
       task = bm.get_task(0)
       env = OffScreenRenderEnv(bddl_file_name=bm.get_task_bddl_file_path(0), camera_heights=128, camera_widths=128)
       obs = env.reset(); print('obs keys:', list(obs.keys())); env.close()
       "

  3. main venv 에서 LIBERO 호출:
       cpd.envs.libero.LiberoEnv 가 LIBERO 모듈을 lazy import 하도록 짜여 있으므로,
       LIBERO 가 필요한 워크플로우 (rollout 수집 등) 는 .venv-libero 의 python 으로 실행.
       예) $VPY scripts/<your_rollout_script>.py

EOF
