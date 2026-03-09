# 🐾 TuriX-Mac OpenClaw Skill (Fast Mode / mac_legacy)

This package integrates OpenClaw with TuriX on branch `mac_legacy` for a faster, lighter macOS desktop automation mode.

## Why this fast mode

- Uses the legacy single-model pipeline (`llm`) in `examples/config.json`
- Fewer model settings than `main` multi-model setup
- Defaults to `use_planner=false` for higher speed

## Files

- `SKILL.md`
- `scripts/run_turix.sh`

## Install

Put this folder into your OpenClaw local skills path:

```bash
your_dir/clawd/skills/local/turix-mac-fast/
├── README.md
├── SKILL.md
└── scripts/
    └── run_turix.sh
```

## Required setup

1. Edit `scripts/run_turix.sh` and set `PROJECT_DIR` to your local TuriX path.
2. Ensure TuriX is on branch `mac_legacy`:
```bash
git -C your_dir/TuriX-CUA branch --show-current
```
3. Before first launch, verify `examples/config.json` already has real model names and keys (do not keep placeholders):
   - `llm.model_name` + `llm.api_key`
   - `planner_llm.model_name` + `planner_llm.api_key`
   - `memory_llm.model_name` + `memory_llm.api_key`
4. Keep Screen Recording and Accessibility permissions enabled for your terminal/IDE and Python runtime.

## Usage

```bash
scripts/run_turix.sh "Open Safari and search for turix"
```

Resume:

```bash
scripts/run_turix.sh --resume my-task-001
```

Enable planner:

```bash
scripts/run_turix.sh --enable-planner "Do a complex workflow"
```

Dry run:

```bash
scripts/run_turix.sh --dry-run "Open Finder"
```

Force stop hotkey:

- Press `Cmd+Shift+2` to immediately terminate the running TuriX agent.
