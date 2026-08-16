"""
屏幕截图模块
"""

import cv2
import numpy as np
from PIL import Image
import pygetwindow as gw
import mss
from config import GAME_WINDOW_TITLE, BACKGROUND_MODE, GAME_CROP


def get_game_window():
    """获取游戏窗口"""
    windows = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)
    if not windows:
        raise RuntimeError(f"找不到窗口: {GAME_WINDOW_TITLE}")
    return windows[0]


def capture_screenshot(save_path: str = None) -> Image.Image:
    """
    截取游戏窗口并返回 PIL Image。
    如果 save_path 不为 None，同时保存到文件。

    优先通过窗口句柄截图，如果窗口位置异常（全屏/无边框模式），
    则回退到截取整个主显示器。
    """
    try:
        win = get_game_window()

        # 检测窗口位置是否有效（全屏/无边框模式可能返回负坐标）
        if win.left < -1000 or win.top < -1000 or win.width < 100 or win.height < 100:
            raise ValueError("Window position invalid, falling back to monitor capture")

        # 激活窗口（后台模式下跳过，不抢焦点）
        try:
            if not BACKGROUND_MODE and not win.isActive:
                win.activate()
        except Exception:
            pass

        monitor = {
            "left": win.left,
            "top": win.top,
            "width": win.width,
            "height": win.height,
        }
    except Exception:
        # 回退：截取整个主显示器
        with mss.mss() as sct:
            mon = sct.monitors[1]  # 主显示器
        monitor = {
            "left": mon["left"],
            "top": mon["top"],
            "width": mon["width"],
            "height": mon["height"],
        }

    with mss.mss() as sct:
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    # ── 裁剪到游戏实际显示区域 ──────────────────────
    if GAME_CROP:
        try:
            parts = [int(x.strip()) for x in GAME_CROP.split(",")]
            if len(parts) == 4:
                left, top, cw, ch = parts
                img = img.crop((left, top, left + cw, top + ch))
        except (ValueError, IndexError):
            pass
    else:
        # 自动检测：找游戏暗色区域的左边界
        img = _auto_crop_game(img)

    if save_path:
        img.save(save_path)

    return img


def _auto_crop_game(img: Image.Image) -> Image.Image:
    """
    自动检测并裁剪到游戏区域。

    在全屏截图中，游戏通常渲染在桌面的一部分。
    通过检测游戏窗口的暗色背景来定位游戏区域。
    只裁剪水平方向（去掉 VS Code 等窗口），保持完整高度。
    """
    import numpy as np
    arr = np.array(img)
    h, w = arr.shape[:2]
    gray = np.mean(arr, axis=2)

    # 中上部分采样（避开顶部UI和底部技能栏）
    sample_top = h // 4
    sample_bot = h * 3 // 4
    sample = gray[sample_top:sample_bot, :]
    col_means = sample.mean(axis=0)

    # 找暗色区域（游戏通常在 10-60 亮度范围）
    # VS Code / 桌面通常更亮 (>80)
    dark_cols = col_means < 55

    dark_runs = []
    in_dark = False
    start = 0
    for i in range(len(dark_cols)):
        if dark_cols[i] and not in_dark:
            start = i
            in_dark = True
        elif not dark_cols[i] and in_dark:
            run_w = i - start
            if run_w > 200:  # 至少 200px 宽才可能是游戏
                dark_runs.append((start, i))
            in_dark = False
    if in_dark:
        run_w = w - start
        if run_w > 200:
            dark_runs.append((start, w))

    if not dark_runs:
        return img  # 无法检测，返回原图

    # 取最宽的暗色区域
    best = max(dark_runs, key=lambda r: r[1] - r[0])
    left, right = best

    if right - left > 300:
        # 只裁剪水平方向，保持完整垂直高度
        return img.crop((left, 0, right, h))

    return img


def get_window_rect() -> dict:
    """获取窗口位置和尺寸"""
    win = get_game_window()
    return {
        "left": win.left,
        "top": win.top,
        "width": win.width,
        "height": win.height,
    }


def screen_to_game_coords(screen_x: int, screen_y: int) -> tuple[int, int]:
    """将屏幕坐标转换为相对于游戏窗口的坐标"""
    rect = get_window_rect()
    return (screen_x - rect["left"], screen_y - rect["top"])


def game_to_screen_coords(game_x: int, game_y: int) -> tuple[int, int]:
    """将游戏窗口内坐标转换为屏幕绝对坐标"""
    rect = get_window_rect()
    return (game_x + rect["left"], game_y + rect["top"])
