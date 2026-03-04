# 🐾 TuriX-Win Clawdbot Skill

This skill allows Clawdbot to control your Windows desktop visually by integrating with the **TuriX Computer Use Agent (CUA)**.

## 🚀 Overview
TuriX acts as the "eyes and hands" for Clawdbot. While Clawdbot is great at terminal and file operations, TuriX allows it to:
- Open and navigate GUI applications (Spotify, Chrome, Settings, etc.)
- Click buttons and interact with complex UIs.
- Perform multi-step visual workflows.

It helps clawdbot complete the task automatically, makes clawdbot the real digital labour!

## 📦 Installation & Setup

### 1. TuriX Core Setup
Set up TuriX following the official repository (Windows branch):
`https://github.com/TurixAI/TuriX-CUA`

```powershell
conda activate turix_env
pip install -r requirements.txt
```

### 2. Skill Configuration
The skill uses a helper script to bridge Clawdbot and TuriX.
- **Helper Script:** `scripts/run_turix.ps1`
- **Skill Definition:** `SKILL.md`

## 🛠 Usage

In your 
```powershell
your_dir/clawd/skills/local/turix-win
```
put the files in this structure:
```
your_dir/clawd/skills/local/turix-win/
├── README.md
├── SKILL.md
└── scripts/
    └── run_turix.ps1
```

You can trigger this skill by asking Clawdbot to perform visual tasks:
> "Use Edge to go to turix.ai, and sign up with Google account."

Clawdbot will execute the following in the background:
```powershell
powershell ./scripts/run_turix.ps1 "Your task here"
```

## 🔍 Troubleshooting
- **`AttributeError: 'NoneType' object has no attribute 'save'`**: This means the screen capture failed. Ensure your Terminal/IDE has screen capture permissions.
- **`uiautomation error`**: Ensure the `uiautomation` library is correctly installed and that the target application is accessible.
