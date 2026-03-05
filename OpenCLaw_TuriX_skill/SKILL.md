---
name: turix-win
description: Computer Use Agent (CUA) for Windows 11 automation using TuriX. Use when you need to perform visual tasks on the desktop, such as opening apps, clicking buttons, or navigating UIs that don't have a CLI or API.
---

# TuriX-Win Skill

This skill allows Clawdbot to control the Windows desktop visually using the TuriX Computer Use Agent.

## When to Use

- When asked to perform actions on the Windows desktop (e.g., "Open Spotify and play my liked songs").
- When navigating applications that lack command-line interfaces.
- For multi-step visual workflows (e.g., "Find the latest invoice in my email and upload it to the company portal").
- When you need the agent to plan, reason, and execute complex tasks autonomously.

## Key Features

### 🤖 Multi-Model Architecture
TuriX uses a sophisticated multi-model system:
- **Brain**: Understands the task and generates step-by-step plans
- **Actor**: Executes precise UI actions based on visual understanding
- **Planner**: Coordinates high-level task decomposition (when `use_plan: true`)
- **Memory**: Maintains context across task steps

### 📋 Skills System
Skills are markdown playbooks that guide the agent for specific domains:
- `browser-tasks`: General web browser operations (Chrome/Edge/Firefox)
- Custom skills can be added to the `src/agent/skills/` directory

### 🔄 Resume Capability
The agent can resume interrupted tasks by setting a stable `agent_id`.

## Running TuriX

### Basic Task
```powershell
./scripts/run_turix.ps1 "Open Chrome and go to github.com"
```

### Resume Interrupted Task
```powershell
./scripts/run_turix.ps1 --resume my-task-001
```

> ✅ **Note**: `run_turix.ps1` updates `examples/config.json` for you. If you want to keep a hand-edited config, skip passing a task and edit `examples/config.json` directly.

### Tips for Effective Tasks

**✅ Good Examples:**
- "Open Edge, go to google.com, search for 'TuriX AI', and click the first result"
- "Open Settings, click on Personalization, then change to Dark mode"
- "Open File Explorer, navigate to Documents, and create a new folder named 'Project X'"

**❌ Avoid:**
- Vague instructions: "Help me" or "Fix this"
- Impossible actions: "Format C: drive"
- Tasks requiring system-level permissions without warning

**💡 Best Practices:**
1. Be specific about the target application
2. Break complex tasks into clear steps, but do not mention the precise coordinates on the screen.

## Hotkeys

- **Force Stop**: `Ctrl+Shift+2` - Immediately stops the agent

## Monitoring & Logs

Logs are saved to `AgentHistory.json` or `.turix_tmp/logging.log` in the project directory. Check this for:
- Step-by-step execution details
- LLM interactions and reasoning
- Errors and recovery attempts

## Important Notes

### How TuriX Runs
- TuriX can be started via clawdbot `exec`
- The first launch takes 1-2 minutes to load models
- Background output is buffered - you won't see live progress until task completes or stops

### Before Running
**Ensure Conda environment is ready:**
```powershell
conda activate turix_env
python examples/main.py
```

### Troubleshooting

| Error                                                       | Solution                                                                                                    |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `AttributeError: 'NoneType' object has no attribute 'save'` | Screen recording failed. Ensure VS Code/Terminal has necessary permissions.                                 |
| `uiautomation error`                                        | Ensure `uiautomation` is installed and the target app is not running as Administrator if the script is not. |
| `Conda environment not found`                               | Ensure `turix_env` exists: `conda create -n turix_env python=3.12`                                          |
| Module import errors                                        | Activate environment: `conda activate turix_env` then `pip install -r requirements.txt`                     |

## Architecture

```
User Request
     ↓
[Clawdbot] → [TuriX Skill] → [run_turix.ps1] → [TuriX Agent]
                                               ↓
                    ┌─────────────────────────┼─────────────────────────┐
                    ↓                         ↓                         ↓
                [Planner]                 [Brain]                  [Memory]
                     ↓                         ↓                         ↓
                                          [Actor] ───→ [Controller] ───→ [Windows UI]
```

## Skill System Details

Skills are markdown files with YAML frontmatter in the `skills/` directory:

```md
---
name: skill-name
description: When to use this skill
---
# Skill Instructions
High-level workflow like: Open Edge, then go to Google.
```

## TuriX Skills System

TuriX supports **Skills**: markdown playbooks that help the agent behave more reliably in specific domains.

### 1. Create a Custom Skill

Create a `.md` file in the TuriX project's `skills/` directory:

```md
---
name: my-custom-skill
description: When performing X specific task
---
# Custom Skill

## Guidelines
- Step 1: Do this first
- Step 2: Then do that
- Step 3: Verify the result
```

### 2. Enable Skills

In `examples/config.json`:

```json
{
  "agent": {
    "use_plan": true,
    "use_skills": true,
    "skills_dir": "skills",
    "skills_max_chars": 4000
  }
}
```

### 3. Run a Task with Skills

```powershell
./scripts/run_turix.ps1 "Search for turix-cua on GitHub and star it"
```

### 4. Chinese Text Support

The `run_turix.ps1` script handles UTF-8 correctly by default in PowerShell.

### 5. Document Creation Best Practices

**Recommended approach:** create the document yourself and let TuriX only send it
1. Create the Word document with `python-docx`
2. Let TuriX only send the file

```python
from docx import Document
doc = Document()
doc.add_heading('Title')
doc.save('C:/path/to/file.docx')
```

### 6. Debugging Tips

1. **Inspect Brain reasoning**: check logic interaction logs for `analysis` and `next_goal`
2. **Inspect Actor actions**: check actor interaction logs for actions
3. **Check screenshots**: TuriX captures a screenshot each step
4. **Read record files**: the agent uses `record_info` to save key info to `.txt` files
