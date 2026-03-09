---
name: turix-mac-fast
description: Fast-mode Computer Use Agent (CUA) for macOS automation using TuriX mac_legacy. Use when users ask for quick mode, faster execution, or lighter model setup for desktop tasks.
---

# TuriX-Mac Fast Skill (mac_legacy)

This skill allows OpenClaw to control the macOS desktop visually using TuriX on branch `mac_legacy`.

## Required Branch

- This skill must run with TuriX repo branch `mac_legacy` (underscore).
- Do not run this fast skill on branch `main`.
- The helper script `scripts/run_turix.sh` validates the active branch before launch when `.git` metadata is available.

## First-Launch Reminder (Required)

- Before first startup, remind the user to verify `examples/config.json` has real model names and API keys (not placeholders).
- At minimum, confirm:
  - `llm.model_name` and `llm.api_key`
  - `planner_llm.model_name` and `planner_llm.api_key`
  - `memory_llm.model_name` and `memory_llm.api_key`
- If placeholders like `your_api_key_here` or `turix-model` remain, stop and ask the user to fill them first.

## When to Use

- Users ask for desktop actions on macOS (open apps, click buttons, navigate UI).
- Users explicitly ask for `quick mode`, `fast mode`, or `mac-legacy`.
- You want faster startup and fewer model config requirements than multi-model `main`.

## Running TuriX (Fast Mode)

### Basic Task
```bash
skills/local/turix-mac-fast/scripts/run_turix.sh "Open Safari and go to github.com"
```

### Resume
```bash
skills/local/turix-mac-fast/scripts/run_turix.sh --resume my-task-001
```

### Enable Planner (optional, for harder tasks)
```bash
skills/local/turix-mac-fast/scripts/run_turix.sh --enable-planner "Finish a longer multi-step UI task"
```

### Dry Run
```bash
skills/local/turix-mac-fast/scripts/run_turix.sh --dry-run "Open Finder"
```

## Notes

- Fast mode defaults to `use_planner: false`.
- Script updates `examples/config.json` safely for:
  - `agent.task`
  - `agent.resume` / `agent_id`
  - `agent.use_planner`
- Set `PROJECT_DIR` in the script to your real local TuriX path before first use.
