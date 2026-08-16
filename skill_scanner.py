"""
技能就绪扫描器

原理：光标落到就绪技能上时图标变大，冷却技能图标不变。
通过对比光标前后的图标尺寸来判断冷却状态。

用法：
  scan_skill_cooldowns() → [True, False, True, False, True, False]
  True = 就绪，False = 冷却/空
"""

import time
import numpy as np
from PIL import Image
from capture import capture_screenshot
from executor import press_key, reset_skill_cursor


def _measure_icon_y_center(img: np.ndarray, tile_index: int) -> float:
    """
    测量技能栏中第 tile_index 个位置图标的垂直中心。
    通过高饱和度像素的加权平均 Y 坐标。
    """
    h, w = img.shape[:2]
    x1 = int(tile_index * w / 6)
    x2 = int((tile_index + 1) * w / 6)
    y1 = int(h * 0.75)
    y2 = h

    region = img[y1:y2, x1:x2]
    if region.size < 100:
        return 0.0

    r = region[:, :, 0].astype(float)
    g = region[:, :, 1].astype(float)
    b = region[:, :, 2].astype(float)
    max_v = np.maximum(np.maximum(r, g), b)
    min_v = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_v > 10, (max_v - min_v) / (max_v + 1e-6), 0)

    colored = sat > 0.25
    if colored.sum() < 10:
        return 0.0

    # Y中心 = 有色像素的加权平均Y坐标
    y_indices = np.arange(region.shape[0])[:, None]
    weights = colored.astype(float)
    y_center = float((y_indices * weights).sum() / max(weights.sum(), 1))

    return y_center


def _icon_jumped_up(before: np.ndarray, after: np.ndarray,
                    tile_index: int) -> bool:
    """
    光标落到技能上后，图标是否向上跳动了。
    就绪技能：Y中心上移 >= 1.5 像素
    冷却技能：Y中心基本不变
    """
    y_before = _measure_icon_y_center(before, tile_index)
    y_after = _measure_icon_y_center(after, tile_index)
    if y_before == 0 or y_after == 0:
        return False
    # 上移（Y变小）=正值
    jump = y_before - y_after
    return jump > 1.5


def scan_skill_cooldowns() -> list[bool]:
    """
    扫描所有6个技能的就绪状态。

    流程：
      1. 截图（光标在当前位置）
      2. 将光标移到最左(0)
      3. 对每个槽位：
         a. 截图（光标在该位置）
         b. 测量图标面积
         c. 将光标右移一位
      4. 比较相邻槽位的面积变化 = 判断就绪

    Returns: [bool × 6]，True=就绪
    """
    from executor import _tap_key

    ready = [False] * 6

    try:
        # ── 重置光标到最左 ──────────────────────────
        reset_skill_cursor()
        time.sleep(0.05)

        prev_img = None
        for i in range(6):
            # 截图（光标在位置 i）
            img = np.array(capture_screenshot())

            if prev_img is not None:
                # 检查上一个技能：光标离开后，就绪图标会掉下来（Y变大）
                # 在 prev_img 中光标在 i-1，在 img 中光标在 i
                ready[i-1] = _icon_dropped_after_cursor_left(prev_img, img, i-1)

            # 最后一个技能：需要额外比较
            if i == 5:
                # 光标移到最左再比
                reset_skill_cursor()
                time.sleep(0.05)
                img_after = np.array(capture_screenshot())
                ready[i] = _icon_dropped_after_cursor_left(img, img_after, i)
            else:
                # 光标右移一位（使用 _tap_key 支持后台模式）
                _tap_key("right", hold=0.02)
                time.sleep(0.05)

            prev_img = img

        reset_skill_cursor()

    except Exception as e:
        pass

    return ready


def _icon_dropped_after_cursor_left(img_cursor_on, img_cursor_off,
                                     tile_index: int) -> bool:
    """光标在技能上时图标高，离开后掉下来=就绪"""
    y_on = _measure_icon_y_center(img_cursor_on, tile_index)
    y_off = _measure_icon_y_center(img_cursor_off, tile_index)
    if y_on == 0 or y_off == 0:
        return False
    # 掉下来 = Y变大（图标回到原位）
    drop = y_off - y_on
    return drop > 1.5
