import os
import re
import subprocess
from typing import Optional


def _parse_target_screen(raw: Optional[str]) -> int:
    """Parse 1-based screen selector; defaults to 1."""
    if not raw:
        return 1
    value = raw.strip().lower()
    m = re.match(r"screen\s*(\d+)$", value)
    if m:
        return max(1, int(m.group(1)))
    if value.isdigit():
        return max(1, int(value))
    return 1


def list_screens() -> list[tuple[int, int, int, int]]:
    """
    Return monitor rectangles as (x, y, width, height), ordered by x then y.
    Linux implementation uses xrandr.
    """
    try:
        proc = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return []

        screens: list[tuple[int, int, int, int]] = []
        pattern = re.compile(r" connected(?: primary)? (\d+)x(\d+)\+(-?\d+)\+(-?\d+)")
        for line in proc.stdout.splitlines():
            m = pattern.search(line)
            if not m:
                continue
            w, h, x, y = map(int, m.groups())
            if w > 0 and h > 0:
                screens.append((x, y, w, h))
        screens.sort(key=lambda s: (s[0], s[1]))
        return screens
    except Exception:
        return []


def get_primary_screen_rect() -> tuple[int, int, int, int]:
    """
    Return primary monitor rect as (x, y, width, height).
    Falls back to first detected monitor, then full-screen fallback.
    """
    try:
        proc = subprocess.run(
            ["xrandr", "--query"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            pattern = re.compile(r" connected primary (\d+)x(\d+)\+(-?\d+)\+(-?\d+)")
            for line in proc.stdout.splitlines():
                m = pattern.search(line)
                if not m:
                    continue
                w, h, x, y = map(int, m.groups())
                if w > 0 and h > 0:
                    return (x, y, w, h)
    except Exception:
        pass

    screens = list_screens()
    if screens:
        return screens[0]

    try:
        import pyautogui

        w, h = pyautogui.size()
        return (0, 0, int(w), int(h))
    except Exception:
        return (0, 0, 1920, 1080)


def get_target_screen_rect() -> tuple[int, int, int, int]:
    """
    Return target screen region as (x, y, width, height).
    Uses env TURIX_TARGET_SCREEN as 1-based index: 1, 2, ... or screen1/screen2.
    """
    screens = list_screens()
    if not screens:
        # Fallback to full screen region from X root geometry.
        try:
            import pyautogui

            w, h = pyautogui.size()
            return (0, 0, int(w), int(h))
        except Exception:
            return (0, 0, 1920, 1080)

    target_1_based = _parse_target_screen(os.getenv("TURIX_TARGET_SCREEN"))
    idx = min(max(target_1_based - 1, 0), len(screens) - 1)
    return screens[idx]
