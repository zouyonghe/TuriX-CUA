#!/bin/bash
# TuriX-CUA Helper Script for OpenClaw (macOS fast mode / mac_legacy)

set -euo pipefail

# ---------- Configuration ----------
PROJECT_DIR="your_dir/TuriX-CUA"
CONFIG_FILE="$PROJECT_DIR/examples/config.json"
ENV_NAME="turix_env"
REQUIRED_BRANCH="mac_legacy"
DEFAULT_CONDA_PATH="/opt/anaconda3/bin/conda"

export PATH="/usr/sbin:/usr/bin:/bin:/sbin:/usr/local/bin:/opt/homebrew/bin:$PATH"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
Usage: run_turix.sh [OPTIONS] [TASK]

OPTIONS:
    -r, --resume ID       Resume task with agent_id
    -c, --config FILE     Use custom config
    --enable-planner      Enable planner (default: off for fast mode)
    --no-planner          Disable planner
    --dry-run             Show command without running
    -h, --help            Show help

EXAMPLES:
    run_turix.sh "Open Safari and go to github.com"
    run_turix.sh --enable-planner "Do a complex multi-step workflow"
    run_turix.sh --resume my-task-001
EOF
}

RESUME_ID=""
CUSTOM_CONFIG=""
DRY_RUN=false
USE_PLANNER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--resume)
            if [[ $# -lt 2 ]]; then
                log_error "Missing value for $1"
                exit 1
            fi
            RESUME_ID="$2"
            shift 2
            ;;
        -c|--config)
            if [[ $# -lt 2 ]]; then
                log_error "Missing value for $1"
                exit 1
            fi
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        --enable-planner)
            USE_PLANNER=true
            shift
            ;;
        --no-planner)
            USE_PLANNER=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

TASK_TEXT="${*:-}"

if [[ -z "$RESUME_ID" && -z "$TASK_TEXT" ]]; then
    log_error "Task or --resume required"
    show_help
    exit 1
fi

if [[ -n "$CUSTOM_CONFIG" ]]; then
    if [[ ! -f "$CUSTOM_CONFIG" ]]; then
        log_error "Config not found: $CUSTOM_CONFIG"
        exit 1
    fi
    CONFIG_FILE="$CUSTOM_CONFIG"
fi

if [[ "$PROJECT_DIR" == "your_dir/TuriX-CUA" ]]; then
    log_error "PROJECT_DIR still uses placeholder path. Set it to your real local TuriX-CUA path first."
    exit 1
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    log_error "TuriX project not found: $PROJECT_DIR"
    exit 1
fi

CONDA_PATH="${CONDA_PATH:-}"
if [[ -z "$CONDA_PATH" ]]; then
    if [[ -x "$DEFAULT_CONDA_PATH" ]]; then
        CONDA_PATH="$DEFAULT_CONDA_PATH"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_PATH="$(command -v conda)"
    else
        log_error "Unable to find conda. Set CONDA_PATH or install/initialize conda first."
        exit 1
    fi
fi

update_config() {
    local task_arg="$1"
    TASK_ARG="$task_arg" USE_PLANNER="$USE_PLANNER" RESUME_ID="$RESUME_ID" CONFIG_FILE="$CONFIG_FILE" \
        python3 << 'PYEOF'
import json
import os

config_path = os.environ["CONFIG_FILE"]

with open(config_path, "r", encoding="utf-8") as f:
    data = json.load(f)

agent = data.setdefault("agent", {})
task_arg = os.environ.get("TASK_ARG", "").strip()
resume_id = os.environ.get("RESUME_ID", "").strip()
use_planner = os.environ.get("USE_PLANNER", "false").lower() == "true"

if task_arg:
    agent["task"] = task_arg

if resume_id:
    agent["resume"] = True
    agent["agent_id"] = resume_id
else:
    # Prevent accidental resume from stale config values.
    agent["resume"] = False

agent["use_planner"] = use_planner

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Config updated successfully")
PYEOF
}

preflight_checks() {
    log_info "Running pre-flight checks..."

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config not found: $CONFIG_FILE"
        exit 1
    fi

    if ! CONFIG_FILE="$CONFIG_FILE" python3 << 'PYEOF'
import json
import os
import sys

config_path = os.environ["CONFIG_FILE"]

with open(config_path, "r", encoding="utf-8") as f:
    data = json.load(f)

placeholder_keys = {
    "your_api_key_here",
    "your_key_here",
    "replace_me",
}
placeholder_models = {
    "turix-model",
    "your_model_name_here",
    "model_name_here",
}

errors = []
api_key_env = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""

for section_name in ("llm", "planner_llm", "memory_llm"):
    section = data.get(section_name)
    if not isinstance(section, dict):
        errors.append(f"{section_name} section missing")
        continue

    raw_provider = section.get("provider")
    provider = raw_provider.strip().lower() if isinstance(raw_provider, str) else ""

    raw_model = section.get("model_name")
    model = raw_model.strip() if isinstance(raw_model, str) else ""

    raw_key = section.get("api_key")
    cfg_key = raw_key.strip() if isinstance(raw_key, str) else ""
    effective_key = cfg_key or api_key_env

    if provider in {"turix", "ollama"}:
        if not model or model.lower() in placeholder_models:
            errors.append(f"{section_name}.model_name is empty or placeholder")

    if provider in {"turix", "gpt", "google_flash", "google_pro", "anthropic"}:
        if not effective_key:
            errors.append(
                f"{section_name}.api_key is missing (and API_KEY/OPENAI_API_KEY env is empty)"
            )
        elif effective_key.lower() in placeholder_keys:
            errors.append(f"{section_name}.api_key is still a placeholder")

if errors:
    print("CONFIG_VALIDATION_FAILED")
    for err in errors:
        print(f"- {err}")
    sys.exit(1)
PYEOF
    then
        log_error "Before first startup, fill real models and keys in examples/config.json."
        log_error "Required fields: llm/planner_llm/memory_llm model_name + api_key."
        exit 1
    fi
    log_info "Model/key config check OK"

    if ! "$CONDA_PATH" env list >/tmp/turix_conda_envs.log 2>&1; then
        log_warn "Unable to inspect conda environments using: $CONDA_PATH"
    elif ! grep -qE "(^|[[:space:]])${ENV_NAME}([[:space:]]|$)" /tmp/turix_conda_envs.log; then
        log_warn "Conda env '$ENV_NAME' not found in 'conda env list' output"
    fi
    rm -f /tmp/turix_conda_envs.log

    if command -v git >/dev/null 2>&1; then
        if [[ -d "$PROJECT_DIR/.git" ]]; then
            local current_branch
            current_branch="$(git -C "$PROJECT_DIR" branch --show-current 2>/dev/null | tr -d '[:space:]')"
            if [[ -z "$current_branch" ]]; then
                log_warn "Unable to detect current git branch under '$PROJECT_DIR'"
            elif [[ "$current_branch" != "$REQUIRED_BRANCH" ]]; then
                log_error "Fast mode requires branch '$REQUIRED_BRANCH', current branch is '$current_branch'"
                log_error "Run: git -C \"$PROJECT_DIR\" checkout $REQUIRED_BRANCH"
                exit 1
            else
                log_info "Branch check OK: $current_branch"
            fi
        else
            log_warn "No .git directory detected under '$PROJECT_DIR'; cannot verify branch."
        fi
    else
        log_warn "git not found in PATH; cannot verify required branch '$REQUIRED_BRANCH'"
    fi

    if ! python3 -c "
import ctypes
CoreGraphics = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
exit(0 if CoreGraphics.CGPreflightScreenCaptureAccess() else 1)
" 2>/dev/null; then
        log_warn "Screen recording permission may be missing"
        log_warn "Grant in System Settings -> Privacy & Security -> Screen Recording"
    fi

    log_info "Pre-flight complete"
}

main() {
    cd "$PROJECT_DIR"
    log_info "TuriX CUA (macOS fast mode)"
    log_info "Project: $PROJECT_DIR"
    log_info "Conda: $CONDA_PATH"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN]"
        if [[ -n "$TASK_TEXT" ]]; then
            echo "  Task: $TASK_TEXT"
        else
            echo "  Task: (resume) $RESUME_ID"
        fi
        echo "  Planner: $USE_PLANNER"
        echo "  Command: $CONDA_PATH run -n $ENV_NAME python examples/main.py"
        exit 0
    fi

    update_config "$TASK_TEXT"
    preflight_checks

    log_info "Starting TuriX mac_legacy..."
    log_info "Press Cmd+Shift+2 to force stop"
    "$CONDA_PATH" run -n "$ENV_NAME" python examples/main.py
}

main
