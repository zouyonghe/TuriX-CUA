import asyncio
import configparser
import logging
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from src.linux.screen import get_primary_screen_rect

logger = logging.getLogger(__name__)

_APP_INDEX: Optional[list[dict]] = None

COMMON_APP_ALIASES: dict[str, list[str]] = {
    "firefox": ["firefox", "mozilla firefox"],
    "google chrome": ["google chrome", "google-chrome", "chrome"],
    "chrome": ["google chrome", "google-chrome", "chrome"],
    "chromium": ["chromium", "chromium browser", "chromium-browser"],
    "edge": ["microsoft edge", "microsoft-edge", "edge"],
    "vscode": ["visual studio code", "vscode", "code"],
    "visual studio code": ["visual studio code", "vscode", "code"],
    "terminal": [
        "terminal",
        "terminal emulator",
        "gnome terminal",
        "konsole",
        "xfce terminal",
        "kitty",
        "alacritty",
        "wezterm",
    ],
    "files": [
        "files",
        "file manager",
        "nautilus",
        "dolphin",
        "thunar",
        "nemo",
        "pcmanfm",
    ],
}

DIRECT_COMMAND_HINTS: dict[str, list[str]] = {
    "firefox": ["firefox"],
    "google chrome": ["google-chrome", "google-chrome-stable", "chrome"],
    "chrome": ["google-chrome", "google-chrome-stable", "chrome"],
    "chromium": ["chromium", "chromium-browser"],
    "edge": ["microsoft-edge", "microsoft-edge-stable"],
    "vscode": ["code", "codium"],
    "visual studio code": ["code", "codium"],
    "terminal": [
        "x-terminal-emulator",
        "gnome-terminal",
        "konsole",
        "xfce4-terminal",
        "kitty",
        "alacritty",
        "wezterm",
        "tilix",
        "mate-terminal",
        "lxterminal",
    ],
    "files": ["nautilus", "dolphin", "thunar", "nemo", "pcmanfm"],
}

TERMINAL_WRAPPERS: list[tuple[str, list[str], bool]] = [
    ("x-terminal-emulator", ["-e"], False),
    ("gnome-terminal", ["--"], False),
    ("konsole", ["-e"], False),
    ("xfce4-terminal", ["--command"], True),
    ("kitty", ["--"], False),
    ("alacritty", ["-e"], False),
    ("wezterm", ["start", "--"], False),
    ("tilix", ["-e"], True),
    ("mate-terminal", ["-e"], True),
    ("lxterminal", ["-e"], True),
]


def _parse_bool(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_text(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("_", " ").replace("-", " ").replace(".", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _unique_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _xdg_application_dirs() -> list[Path]:
    xdg_data_home = Path(
        os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))
    )
    xdg_data_dirs = [
        Path(p)
        for p in os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
        if p
    ]

    candidates = [xdg_data_home / "applications"]
    candidates.extend(base / "applications" for base in xdg_data_dirs)

    candidates.extend(
        [
            Path("/var/lib/snapd/desktop/applications"),
            Path("/var/lib/flatpak/exports/share/applications"),
            Path.home() / ".local/share/flatpak/exports/share/applications",
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


APPLICATION_DIRS = _xdg_application_dirs()


def _locale_candidates() -> list[str]:
    raw_values: list[str] = []
    for env_name in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        raw = os.environ.get(env_name, "")
        if raw:
            raw_values.extend(part for part in raw.split(":") if part)

    candidates: list[str] = []
    for raw in raw_values:
        value = raw.split(".", 1)[0]
        value = value.split("@", 1)[0]
        if not value:
            continue
        candidates.append(value)
        if "_" in value:
            candidates.append(value.split("_", 1)[0])

    return _unique_keep_order(candidates)


def _get_localized_value(section: configparser.SectionProxy, key: str) -> str:
    for locale_key in _locale_candidates():
        candidate = f"{key}[{locale_key}]"
        if candidate in section and section.get(candidate, "").strip():
            return section.get(candidate, "").strip()

    if key in section and section.get(key, "").strip():
        return section.get(key, "").strip()

    for candidate in section.keys():
        if candidate.startswith(f"{key}[") and section.get(candidate, "").strip():
            return section.get(candidate, "").strip()

    return ""


def _split_desktop_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(";") if item.strip()]


def _current_desktops() -> set[str]:
    raw = os.environ.get("XDG_CURRENT_DESKTOP", "")
    return {
        part.strip().lower()
        for part in re.split(r"[:;]", raw)
        if part and part.strip()
    }


def _should_show_in_current_desktop(section: configparser.SectionProxy) -> bool:
    current = _current_desktops()
    if not current:
        return True

    only_show = {item.lower() for item in _split_desktop_list(section.get("OnlyShowIn", ""))}
    not_show = {item.lower() for item in _split_desktop_list(section.get("NotShowIn", ""))}

    if only_show and current.isdisjoint(only_show):
        return False
    if not_show and not current.isdisjoint(not_show):
        return False
    return True


def _desktop_id_from_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    return relative.replace("/", "-")


def _is_executable_available(command: str) -> bool:
    command = command.strip()
    if not command:
        return False

    expanded = os.path.expanduser(os.path.expandvars(command))
    if os.path.isabs(expanded):
        return os.path.isfile(expanded) and os.access(expanded, os.X_OK)
    return shutil.which(expanded) is not None


def _strip_exec_field_codes(exec_cmd: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(exec_cmd):
        ch = exec_cmd[i]
        if ch == "%" and i + 1 < len(exec_cmd):
            nxt = exec_cmd[i + 1]
            if nxt == "%":
                out.append("%")
                i += 2
                continue
            if nxt.isalpha():
                i += 2
                continue
        out.append(ch)
        i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _split_exec_args(exec_cmd: str) -> list[str]:
    cleaned = _strip_exec_field_codes(exec_cmd)
    if not cleaned:
        return []
    try:
        return shlex.split(cleaned)
    except Exception:
        return []


def _exec_basename(exec_cmd: str) -> str:
    args = _split_exec_args(exec_cmd)
    if not args:
        return ""
    return Path(args[0]).name.lower().replace(".desktop", "").replace(".exe", "")


def _parse_desktop_file(path: Path, root: Path) -> Optional[dict]:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            parser.read_file(f)
    except Exception as e:
        logger.debug("Failed to parse desktop file %s: %s", path, e)
        return None

    if "Desktop Entry" not in parser:
        return None

    section = parser["Desktop Entry"]

    if section.get("Type", "Application").strip() != "Application":
        return None
    if _parse_bool(section.get("Hidden")):
        return None
    if _parse_bool(section.get("NoDisplay")):
        return None
    if not _should_show_in_current_desktop(section):
        return None

    name = _get_localized_value(section, "Name")
    if not name:
        return None

    try_exec = section.get("TryExec", "").strip()
    if try_exec and not _is_executable_available(try_exec):
        return None

    dbus_activatable = _parse_bool(section.get("DBusActivatable"))
    exec_cmd = section.get("Exec", "").strip()
    exec_cmd = _strip_exec_field_codes(exec_cmd)

    if not exec_cmd and not dbus_activatable:
        return None

    working_dir = os.path.expanduser(os.path.expandvars(section.get("Path", "").strip()))
    if not working_dir:
        working_dir = ""

    keywords = _split_desktop_list(_get_localized_value(section, "Keywords"))

    return {
        "name": name,
        "generic_name": _get_localized_value(section, "GenericName"),
        "comment": _get_localized_value(section, "Comment"),
        "keywords": keywords,
        "exec": exec_cmd,
        "try_exec": try_exec,
        "path": working_dir,
        "terminal": _parse_bool(section.get("Terminal")),
        "startup_wm_class": section.get("StartupWMClass", "").strip(),
        "dbus_activatable": dbus_activatable,
        "desktop_path": str(path),
        "desktop_id": _desktop_id_from_path(path, root),
        "source_dir": str(root),
    }


def _scan_applications() -> list[dict]:
    apps: list[dict] = []
    seen_desktop_ids: set[str] = set()

    for directory in APPLICATION_DIRS:
        if not directory.exists():
            continue

        try:
            desktop_files = directory.rglob("*.desktop")
        except Exception as e:
            logger.debug("Failed to iterate application directory %s: %s", directory, e)
            continue

        for desktop_file in desktop_files:
            if not desktop_file.is_file():
                continue

            app = _parse_desktop_file(desktop_file, directory)
            if not app:
                continue

            desktop_id = app["desktop_id"]
            if desktop_id in seen_desktop_ids:
                continue

            seen_desktop_ids.add(desktop_id)
            apps.append(app)

    return apps


def build_index(force: bool = True) -> None:
    global _APP_INDEX
    if _APP_INDEX is None or force:
        _APP_INDEX = _scan_applications()


def _load_index(force: bool = False) -> list[dict]:
    if _APP_INDEX is None or force:
        build_index(force=True)
    return _APP_INDEX or []


def list_applications() -> list[str]:
    names = sorted({app["name"] for app in _load_index(force=False)}, key=str.lower)
    return names


def _record_search_fields(app: dict) -> list[str]:
    values: list[str] = [
        app.get("name", ""),
        app.get("generic_name", ""),
        app.get("comment", ""),
        app.get("startup_wm_class", ""),
        app.get("desktop_id", "").replace(".desktop", ""),
        _exec_basename(app.get("exec", "")),
    ]
    values.extend(app.get("keywords", []))
    return [v for v in (_normalize_text(x) for x in values) if v]


def _expanded_queries(app_name: str) -> list[str]:
    normalized_name = _normalize_text(app_name)
    queries = [normalized_name]

    queries.extend(_normalize_text(alias) for alias in COMMON_APP_ALIASES.get(normalized_name, []))
    queries.extend(_normalize_text(cmd) for cmd in DIRECT_COMMAND_HINTS.get(normalized_name, []))

    return _unique_keep_order([q for q in queries if q])


def _score_record(query: str, app: dict) -> int:
    fields = _record_search_fields(app)
    if not fields:
        return 0

    score = 0
    normalized_name = _normalize_text(app.get("name", ""))
    exec_name = _normalize_text(_exec_basename(app.get("exec", "")))
    startup_wm_class = _normalize_text(app.get("startup_wm_class", ""))

    if query == normalized_name:
        score = max(score, 140)
    if exec_name and query == exec_name:
        score = max(score, 132)
    if startup_wm_class and query == startup_wm_class:
        score = max(score, 130)

    for field in fields:
        if query == field:
            score = max(score, 125)
        elif field.startswith(query) or query.startswith(field):
            score = max(score, 108)
        elif query in field:
            score = max(score, 96)
        else:
            fuzzy_score = int(fuzz.WRatio(query, field))
            score = max(score, fuzzy_score)

    return score


def resolve_app(name: str) -> Optional[dict]:
    query = name.strip()
    if not query:
        return None

    queries = _expanded_queries(query)

    def choose_best(index: list[dict]) -> Optional[dict]:
        best_app: Optional[dict] = None
        best_score = 0

        for app in index:
            score = max((_score_record(q, app) for q in queries), default=0)
            if score > best_score:
                best_app = app
                best_score = score

        if best_app and best_score >= 72:
            return best_app
        return None

    index = _load_index(force=False)
    best = choose_best(index)
    if best:
        return best

    index = _load_index(force=True)
    return choose_best(index)


def _build_match_tokens(app_name: str, rec: Optional[dict]) -> list[str]:
    tokens: list[str] = [_normalize_text(app_name)]
    tokens.extend(_expanded_queries(app_name))

    if rec:
        tokens.extend(
            [
                _normalize_text(rec.get("name", "")),
                _normalize_text(rec.get("generic_name", "")),
                _normalize_text(rec.get("startup_wm_class", "")),
                _normalize_text(rec.get("desktop_id", "").replace(".desktop", "")),
                _normalize_text(_exec_basename(rec.get("exec", ""))),
            ]
        )
        tokens.extend(
            _normalize_text(keyword)
            for keyword in rec.get("keywords", [])
            if len(keyword.strip()) >= 4
        )

    if "firefox" in tokens:
        tokens.extend(
            [
                "mozilla firefox",
                "navigator firefox",
                "navigator.firefox",
                "firefox firefox",
            ]
        )
    if "google chrome" in tokens or "chrome" in tokens:
        tokens.extend(["google chrome", "google-chrome", "chrome"])
    if "visual studio code" in tokens or "vscode" in tokens or "code" in tokens:
        tokens.extend(["visual studio code", "vscode", "code"])

    deduped = _unique_keep_order([t for t in tokens if t])
    deduped.sort(key=lambda item: (-len(item), item))
    return deduped


def _can_use_wmctrl() -> bool:
    return shutil.which("wmctrl") is not None


def _can_use_xdotool() -> bool:
    return shutil.which("xdotool") is not None


def _has_window_control_backend() -> bool:
    return _can_use_wmctrl() or _can_use_xdotool()


def _prepare_primary_screen_launch_context() -> None:
    """Best-effort: move pointer to primary monitor so new windows open there."""
    try:
        import pyautogui

        x, y, w, h = get_primary_screen_rect()
        target_x = int(x + (w * 0.5))
        target_y = int(y + (h * 0.5))
        pyautogui.moveTo(target_x, target_y, duration=0.05)
        pyautogui.click(target_x, target_y)
    except Exception:
        # Launch should continue even if this hint cannot be applied.
        return


async def _request_running_app_activation(rec: Optional[dict], query: str) -> tuple[bool, str]:
    """
    Best-effort activation request when no window-control backend is available.
    Uses desktop launchers which may route to an existing single-instance app.
    """
    if rec:
        ok, msg = await _launch_desktop_entry(rec)
        if ok:
            return True, f"activation request sent via desktop entry ({msg})"
        return False, msg

    for candidate in _direct_command_candidates(query):
        ok, msg = await _launch_exec(candidate)
        if ok:
            return True, f"activation request sent via command ({candidate})"
    return False, "no activation request path available"


def _list_windows() -> list[dict]:
    windows: list[dict] = []

    if _can_use_wmctrl():
        try:
            proc = subprocess.run(
                ["wmctrl", "-lxp"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    parts = line.split(None, 5)
                    if len(parts) < 5:
                        continue
                    pid = ""
                    if len(parts) >= 5:
                        pid = parts[2]
                    wm_class = parts[4] if len(parts) >= 5 else ""
                    title = parts[5] if len(parts) > 5 else ""
                    windows.append(
                        {
                            "id": parts[0],
                            "pid": pid,
                            "wm_class": wm_class,
                            "title": title,
                        }
                    )
                return windows
        except Exception:
            pass

    if not _can_use_xdotool():
        return []

    try:
        proc = subprocess.run(
            ["xdotool", "search", "--all", "--name", ".*"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    for raw_id in proc.stdout.splitlines():
        raw_id = raw_id.strip()
        if not raw_id:
            continue

        try:
            win_id = hex(int(raw_id))
        except Exception:
            continue

        title = ""
        wm_class = ""

        try:
            name_proc = subprocess.run(
                ["xdotool", "getwindowname", raw_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if name_proc.returncode == 0:
                title = (name_proc.stdout or "").strip()
        except Exception:
            pass

        if shutil.which("xprop") is not None:
            try:
                class_proc = subprocess.run(
                    ["xprop", "-id", str(int(raw_id)), "WM_CLASS"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if class_proc.returncode == 0:
                    # Example: WM_CLASS(STRING) = "Navigator", "firefox"
                    classes = re.findall(r'"([^"]+)"', class_proc.stdout or "")
                    if classes:
                        wm_class = ".".join(classes)
            except Exception:
                pass

        windows.append(
            {
                "id": win_id,
                "pid": "",
                "wm_class": wm_class,
                "title": title,
            }
        )

    return windows


def _normalize_window_id(value: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    if not raw.startswith("0x"):
        try:
            raw = hex(int(raw, 16))
        except Exception:
            return raw
    return raw


def _window_stacking_order() -> dict[str, int]:
    if shutil.which("xprop") is None:
        return {}

    try:
        proc = subprocess.run(
            ["xprop", "-root", "_NET_CLIENT_LIST_STACKING"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {}

    if proc.returncode != 0:
        return {}

    # xprop returns ids from bottom -> top, we use larger index as more recent/top.
    ids = re.findall(r"0x[0-9a-fA-F]+", proc.stdout or "")
    return {_normalize_window_id(win_id): idx for idx, win_id in enumerate(ids)}


def _window_score(window: dict, tokens: list[str]) -> int:
    wm_class = _normalize_text(window.get("wm_class", ""))
    title = _normalize_text(window.get("title", ""))

    best = 0
    for token in tokens:
        if not token:
            continue

        if wm_class and token == wm_class:
            best = max(best, 140)
        elif wm_class and token in wm_class:
            best = max(best, 118)
        elif title and token == title:
            best = max(best, 102)
        elif title and token in title:
            best = max(best, 82)
        elif len(token) >= 4:
            if wm_class:
                best = max(best, int(fuzz.WRatio(token, wm_class) * 0.82))
            if title:
                best = max(best, int(fuzz.WRatio(token, title) * 0.60))

    return best


def _find_matching_window(tokens: list[str]) -> Optional[str]:
    windows = _list_windows()
    if not windows or not tokens:
        return None

    stacking = _window_stacking_order()
    best_id: Optional[str] = None
    best_score = 0
    best_stack = -1
    normalized_tokens = [_normalize_text(t) for t in tokens if t]

    for window in windows:
        score = _window_score(window, normalized_tokens)
        if score < 90:
            continue

        window_id = _normalize_window_id(window.get("id", ""))
        stack_rank = stacking.get(window_id, -1)

        if score > best_score or (score == best_score and stack_rank > best_stack):
            best_score = score
            best_stack = stack_rank
            best_id = window_id or window["id"]

    if best_id:
        return best_id

    # Fallback: relaxed containment matching to avoid false "not running" cases.
    # This prefers the most recently used/top window when multiple windows match.
    for window in windows:
        wm_class = _normalize_text(window.get("wm_class", ""))
        title = _normalize_text(window.get("title", ""))
        joined = f"{wm_class} {title}".strip()
        if not joined:
            continue

        matched = any(token and len(token) >= 2 and token in joined for token in normalized_tokens)
        if not matched:
            continue

        window_id = _normalize_window_id(window.get("id", ""))
        stack_rank = stacking.get(window_id, -1)
        if stack_rank > best_stack:
            best_stack = stack_rank
            best_id = window_id or window["id"]

    if best_id:
        return best_id
    return None


def _activate_window_by_id(window_id: str) -> bool:
    raw = _normalize_window_id(window_id)
    numeric = str(int(raw, 16)) if raw.startswith("0x") else raw

    if _can_use_wmctrl():
        try:
            subprocess.run(
                ["wmctrl", "-ir", window_id, "-b", "remove,hidden"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            result = subprocess.run(
                ["wmctrl", "-ia", window_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    if not _can_use_xdotool():
        return False

    try:
        subprocess.run(
            ["xdotool", "windowmap", numeric],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        result = subprocess.run(
            ["xdotool", "windowactivate", "--sync", numeric],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def _move_window_to_primary_screen(window_id: str) -> bool:
    try:
        x, y, _, _ = get_primary_screen_rect()
    except Exception:
        return False

    if _can_use_wmctrl():
        try:
            result = subprocess.run(
                ["wmctrl", "-ir", window_id, "-e", f"0,{x},{y},-1,-1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    if not _can_use_xdotool():
        return False

    raw = _normalize_window_id(window_id)
    try:
        numeric = str(int(raw, 16)) if raw.startswith("0x") else raw
        result = subprocess.run(
            ["xdotool", "windowmove", numeric, str(x), str(y)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


async def _wait_for_window(
    tokens: list[str],
    attempts: int = 40,
    delay: float = 0.1,
    move_to_primary: bool = False,
) -> bool:
    for _ in range(attempts):
        win_id = _find_matching_window(tokens)
        if win_id:
            activated = _activate_window_by_id(win_id)
            moved = _move_window_to_primary_screen(win_id) if move_to_primary else False
            if activated or moved:
                return True
        await asyncio.sleep(delay)
    return False


def _launch_subprocess(
    args: list[str],
    cwd: Optional[str] = None,
) -> tuple[bool, str]:
    if not args:
        return False, "empty command"

    try:
        subprocess.Popen(
            args,
            cwd=cwd or None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, "launched"
    except Exception as e:
        return False, f"launch failed: {e}"


async def _launch_exec(exec_cmd: str, cwd: Optional[str] = None) -> tuple[bool, str]:
    args = _split_exec_args(exec_cmd)
    return _launch_subprocess(args, cwd=cwd)


async def _launch_in_terminal(exec_cmd: str, cwd: Optional[str] = None) -> tuple[bool, str]:
    args = _split_exec_args(exec_cmd)
    if not args:
        return False, "empty command"

    last_error = "no terminal wrapper found"
    cmd_str = " ".join(shlex.quote(arg) for arg in args)

    for terminal_bin, prefix, expects_single_string in TERMINAL_WRAPPERS:
        if not shutil.which(terminal_bin):
            continue

        if expects_single_string:
            launch_cmd = [terminal_bin, *prefix, cmd_str]
        else:
            launch_cmd = [terminal_bin, *prefix, *args]

        ok, msg = _launch_subprocess(launch_cmd, cwd=cwd)
        if ok:
            return True, f"launched via {terminal_bin}"
        last_error = msg

    return False, last_error


async def _launch_desktop_entry(rec: dict) -> tuple[bool, str]:
    desktop_path = rec.get("desktop_path", "")
    desktop_id = rec.get("desktop_id", "")
    cwd = rec.get("path") or None

    if desktop_path and shutil.which("gio"):
        ok, msg = _launch_subprocess(["gio", "launch", desktop_path], cwd=cwd)
        if ok:
            return True, "gio launch ok"
        logger.debug("gio launch failed for %s: %s", desktop_path, msg)

    if desktop_id and shutil.which("gtk-launch"):
        ok, msg = _launch_subprocess(["gtk-launch", desktop_id], cwd=cwd)
        if ok:
            return True, "gtk-launch ok"
        logger.debug("gtk-launch failed for %s: %s", desktop_id, msg)

    exec_cmd = rec.get("exec", "")
    if not exec_cmd:
        return False, "no exec command"

    if rec.get("terminal"):
        ok, msg = await _launch_in_terminal(exec_cmd, cwd=cwd)
        if ok:
            return True, msg

    return await _launch_exec(exec_cmd, cwd=cwd)


def _looks_like_resource(target: str) -> bool:
    value = target.strip()
    lower = value.lower()
    if lower.startswith(("http://", "https://", "file://", "mailto:")):
        return True
    if value.startswith("/") or value.startswith("~/"):
        return True

    expanded = os.path.expanduser(os.path.expandvars(value))
    return Path(expanded).exists()


def _direct_command_candidates(app_name: str) -> list[str]:
    normalized_name = _normalize_text(app_name)
    candidates: list[str] = []

    for candidate in DIRECT_COMMAND_HINTS.get(normalized_name, []):
        if _is_executable_available(candidate):
            candidates.append(candidate)

    raw = app_name.strip()
    if raw:
        candidates.append(raw)

    return _unique_keep_order(candidates)


def _build_process_tokens(app_name: str, rec: Optional[dict]) -> list[str]:
    tokens: list[str] = []

    def add_token(raw: str) -> None:
        normalized = _normalize_text(raw)
        if normalized:
            tokens.append(normalized)

    if rec:
        exec_name = _exec_basename(rec.get("exec", ""))
        if exec_name:
            add_token(exec_name)
        try_exec_name = Path(rec.get("try_exec", "")).name.strip().lower()
        if try_exec_name:
            add_token(try_exec_name)
        startup_name = rec.get("startup_wm_class", "").strip().lower()
        if startup_name and " " not in startup_name:
            add_token(startup_name)

    for cmd in _direct_command_candidates(app_name):
        args = _split_exec_args(cmd)
        if args:
            add_token(Path(args[0]).name.strip().lower())
        else:
            add_token(Path(cmd).name.strip().lower())

    return _unique_keep_order(
        [
            token
            for token in tokens
            if len(token) >= 3
            and " " not in token
            and token not in {"python", "bash", "sh", "env", "gio", "gtk-launch", "xdg-open"}
        ]
    )


def _list_processes() -> list[tuple[str, str]]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "comm=,args="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    rows: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        comm = _normalize_text(parts[0])
        args = _normalize_text(parts[1] if len(parts) > 1 else "")
        rows.append((comm, args))
    return rows


def _is_existing_instance_running(tokens: list[str]) -> bool:
    if not tokens:
        return False

    processes = _list_processes()
    if not processes:
        return False

    for comm, _args in processes:
        if not comm:
            continue
        for token in tokens:
            # Strict process-name match to avoid false positives.
            if token == comm:
                return True
            if comm.startswith(token + "-"):
                return True
    return False


def _running_pids_for_tokens(tokens: list[str]) -> set[str]:
    if not tokens:
        return set()
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,comm="],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return set()
    if proc.returncode != 0:
        return set()

    pids: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid, comm = parts[0], _normalize_text(parts[1])
        if not comm:
            continue
        for token in tokens:
            if token == comm or comm.startswith(token + "-"):
                pids.add(pid)
                break
    return pids


def _find_window_by_pids(pids: set[str]) -> Optional[str]:
    if not pids:
        return None
    windows = _list_windows()
    if not windows:
        return None

    stacking = _window_stacking_order()
    best_id: Optional[str] = None
    best_stack = -1

    for window in windows:
        win_pid = str(window.get("pid", "")).strip()
        if not win_pid or win_pid not in pids:
            continue
        window_id = _normalize_window_id(window.get("id", ""))
        stack_rank = stacking.get(window_id, -1)
        if stack_rank > best_stack:
            best_stack = stack_rank
            best_id = window_id or window.get("id")

    return best_id


def _activate_existing_window_by_class_hint(tokens: list[str]) -> bool:
    hints = _unique_keep_order([_normalize_text(t) for t in tokens if t])
    if not hints:
        return False

    stacking = _window_stacking_order()
    windows = _list_windows()
    if windows:
        best_id: Optional[str] = None
        best_stack = -1
        for window in windows:
            wm_class = _normalize_text(window.get("wm_class", ""))
            if not wm_class:
                continue
            if not any(len(hint) >= 2 and hint in wm_class for hint in hints):
                continue
            window_id = _normalize_window_id(window.get("id", ""))
            stack_rank = stacking.get(window_id, -1)
            if stack_rank > best_stack:
                best_stack = stack_rank
                best_id = window_id or window.get("id")
        if best_id and _activate_window_by_id(best_id):
            return True

    if _can_use_xdotool():
        for hint in hints:
            if len(hint) < 2:
                continue
            try:
                search = subprocess.run(
                    ["xdotool", "search", "--all", "--class", hint],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if search.returncode != 0:
                    continue
                best_raw_id = ""
                best_stack = -1
                for raw_id in search.stdout.splitlines():
                    raw_id = raw_id.strip()
                    if not raw_id:
                        continue
                    normalized = _normalize_window_id(hex(int(raw_id)))
                    stack_rank = stacking.get(normalized, -1)
                    if stack_rank > best_stack:
                        best_stack = stack_rank
                        best_raw_id = raw_id
                if best_raw_id:
                    subprocess.run(
                        ["xdotool", "windowmap", best_raw_id],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    act = subprocess.run(
                        ["xdotool", "windowactivate", "--sync", best_raw_id],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    if act.returncode == 0:
                        return True
            except Exception:
                continue

    return False


async def _finalize_launch_result(
    launch_label: str,
    bring_to_front: bool,
    tokens: list[str],
) -> tuple[bool, str]:
    if not bring_to_front:
        return True, f"launched {launch_label}"
    if await _wait_for_window(tokens, move_to_primary=True):
        return True, f"launched and activated {launch_label}"
    return True, f"launched {launch_label} (window activation not confirmed)"


async def open_application_by_name(
    app_name: str,
    bring_to_front: bool = True,
) -> tuple[bool, str]:
    query = app_name.strip()
    if not query:
        return False, "empty app name"

    rec = resolve_app(query)
    tokens = _build_match_tokens(query, rec)
    window_control_available = _has_window_control_backend()
    process_tokens = _build_process_tokens(query, rec)

    if bring_to_front:
        win_id = _find_matching_window(tokens)
        if not win_id and _can_use_wmctrl():
            win_id = _find_window_by_pids(_running_pids_for_tokens(process_tokens))
        if win_id:
            target_name = rec["name"] if rec else query
            if _activate_window_by_id(win_id):
                return True, f"activated existing window for {target_name}"
            # Existing window found; avoid opening a new instance.
            return True, f"found existing window for {target_name} but activation failed"
        if _activate_existing_window_by_class_hint(tokens):
            target_name = rec["name"] if rec else query
            return True, f"activated existing window by class hint for {target_name}"

    if bring_to_front and not window_control_available:
        if _is_existing_instance_running(process_tokens):
            # No window backend: best effort to ask app to focus existing instance.
            target_name = rec["name"] if rec else query
            ok, detail = await _request_running_app_activation(rec, query)
            if ok:
                return True, f"requested existing window for {target_name}: {detail}"
            return True, f"{target_name} already running; skipped launching new instance"
        # Bias new app launch onto primary screen in multi-monitor setups.
        _prepare_primary_screen_launch_context()

    launch_errors: list[str] = []

    if rec:
        ok, msg = await _launch_desktop_entry(rec)
        if ok:
            return await _finalize_launch_result(rec["name"], bring_to_front, tokens)

        logger.warning("Desktop entry launch failed for '%s': %s", query, msg)
        launch_errors.append(f"desktop entry: {msg}")

    for candidate in _direct_command_candidates(query):
        ok, msg = await _launch_exec(candidate)
        if ok:
            return await _finalize_launch_result(candidate, bring_to_front, tokens)

        launch_errors.append(f"{candidate}: {msg}")

    if _looks_like_resource(query):
        ok, msg = _launch_subprocess(["xdg-open", query])
        if ok:
            return True, f"xdg-open {query}"
        launch_errors.append(f"xdg-open: {msg}")

    if launch_errors:
        return False, "; ".join(launch_errors[-3:])

    return False, f"failed to launch app '{query}'"
