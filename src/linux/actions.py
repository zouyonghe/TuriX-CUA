import asyncio
import logging
import shutil
import subprocess
from typing import Optional, Tuple

import pyautogui
import pyperclip

from src.linux.screen import get_target_screen_rect

logger = logging.getLogger(__name__)


class LinuxActions:
    """Linux UI actions implemented with PyAutoGUI."""

    def getscreen_size(self) -> Tuple[int, int]:
        try:
            _, _, screen_width, screen_height = get_target_screen_rect()
            return int(screen_width), int(screen_height)
        except Exception as e:
            logger.error(f"Error getting screen size: {e}")
            return 1920, 1080

    def _to_pixel(self, x: float, y: float) -> tuple[float, float]:
        offset_x, offset_y, screen_w, screen_h = get_target_screen_rect()
        if 0 <= x <= 1 and 0 <= y <= 1:
            return offset_x + (screen_w * x), offset_y + (screen_h * y)
        return offset_x + (screen_w * (x / 1000)), offset_y + (screen_h * (y / 1000))

    @staticmethod
    def _is_ascii_text(text: str) -> bool:
        try:
            text.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    async def _type_with_xdotool(self, text: str) -> bool:
        if not text:
            return True
        if shutil.which("xdotool") is None:
            return False
        try:
            proc = await asyncio.to_thread(
                subprocess.run,
                ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                return True
            logger.warning("xdotool type failed (code=%s): %s", proc.returncode, proc.stderr.strip())
            return False
        except Exception as e:
            logger.warning("xdotool type error: %s", e)
            return False

    async def _send_paste_hotkey(self) -> bool:
        if shutil.which("xdotool") is not None:
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    ["xdotool", "key", "--clearmodifiers", "ctrl+v"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0:
                    return True
                logger.warning("xdotool paste failed (code=%s): %s", proc.returncode, proc.stderr.strip())
            except Exception as e:
                logger.warning("xdotool paste error: %s", e)

        try:
            await asyncio.to_thread(pyautogui.hotkey, "ctrl", "v")
            return True
        except Exception as e:
            logger.error("PyAutoGUI paste error: %s", e)
            return False

    async def click(self, x: float, y: float, button: str = "left") -> bool:
        pos_x, pos_y = self._to_pixel(x, y)
        try:
            if button == "left":
                pyautogui.click(pos_x, pos_y)
            elif button == "right":
                pyautogui.rightClick(pos_x, pos_y)
            elif button == "double":
                pyautogui.doubleClick(pos_x, pos_y)
            else:
                logger.error("Unknown click button '%s'", button)
                return False

            await asyncio.sleep(0.1)
            return True
        except Exception as e:
            logger.error(f"Error clicking at ({x}, {y}): {e}")
            return False

    async def drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration: float = 0.5,
    ) -> bool:
        pos_start_x, pos_start_y = self._to_pixel(start_x, start_y)
        pos_end_x, pos_end_y = self._to_pixel(end_x, end_y)
        try:
            pyautogui.moveTo(pos_start_x, pos_start_y, duration=0.1)
            pyautogui.dragTo(pos_end_x, pos_end_y, duration=duration, button="left")
            return True
        except Exception as e:
            logger.error(
                "Error dragging from (%s, %s) to (%s, %s): %s",
                start_x,
                start_y,
                end_x,
                end_y,
                e,
            )
            return False

    async def scroll(self, x: float, y: float, clicks: int) -> bool:
        pos_x, pos_y = self._to_pixel(x, y)
        try:
            pyautogui.scroll(clicks, x=pos_x, y=pos_y)
            return True
        except Exception as e:
            logger.error(f"Error scrolling at ({x}, {y}): {e}")
            return False

    async def type_text(self, text: str) -> bool:
        original_clipboard = None
        try:
            if not text:
                return True

            # xdotool key injection is often unreliable for CJK/IME text.
            # Keep xdotool only for pure ASCII to preserve fast English typing.
            if self._is_ascii_text(text) and await self._type_with_xdotool(text):
                return True

            for attempt in range(3):
                try:
                    original_clipboard = await asyncio.to_thread(pyperclip.paste)
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: Failed to read clipboard: {e}")
                    await asyncio.sleep(0.2 * (attempt + 1))

            for attempt in range(3):
                try:
                    await asyncio.to_thread(pyperclip.copy, text)
                    await asyncio.sleep(0.05)
                    if await asyncio.to_thread(pyperclip.paste) != text:
                        raise ValueError("Clipboard verification failed")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}: Failed to set clipboard: {e}")
                    await asyncio.sleep(0.2 * (attempt + 1))
            else:
                return False

            await asyncio.sleep(0.1)
            if not await self._send_paste_hotkey():
                return False
            await asyncio.sleep(0.2)

            if original_clipboard is not None:
                for attempt in range(5):
                    try:
                        await asyncio.to_thread(pyperclip.copy, original_clipboard)
                        await asyncio.sleep(0.05)
                        if await asyncio.to_thread(pyperclip.paste) == original_clipboard:
                            break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1}: Failed to restore clipboard: {e}")
                        await asyncio.sleep(0.2 * (attempt + 1))
            return True
        except Exception as e:
            logger.error(f"Error typing text '{text}': {e}")
            if original_clipboard is not None:
                try:
                    await asyncio.to_thread(pyperclip.copy, original_clipboard)
                except Exception:
                    pass
            return False

    async def press_key(self, key: str) -> bool:
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            logger.error(f"Error pressing key '{key}': {e}")
            return False

    async def press_hotkey(self, key1: str, key2: str, key3: Optional[str] = None) -> bool:
        try:
            if key3 is not None:
                pyautogui.hotkey(key1, key2, key3)
            else:
                pyautogui.hotkey(key1, key2)
            return True
        except Exception as e:
            logger.error(f"Error pressing hotkey {key1, key2, key3}: {e}")
            return False

    async def take_screenshot(self, save_path: Optional[str] = None) -> bool:
        try:
            x, y, w, h = get_target_screen_rect()
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
            if save_path:
                screenshot.save(save_path)
            return True
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return False

    async def get_mouse_position(self) -> Tuple[int, int]:
        try:
            return pyautogui.position()
        except Exception as e:
            logger.error(f"Error getting mouse position: {e}")
            return (0, 0)

    async def move_mouse(self, x: float, y: float, duration: float = 0.0) -> bool:
        pos_x, pos_y = self._to_pixel(x, y)
        try:
            pyautogui.moveTo(pos_x, pos_y, duration=duration)
            return True
        except Exception as e:
            logger.error(f"Error moving mouse to ({x}, {y}): {e}")
            return False
