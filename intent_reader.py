"""
意图识别模块 v2 — 基于实际截图校准

关键发现（来自 7 张截图的像素分析）：
  - 意图标识位于敌人头顶 grid_y - 70 到 grid_y - 130 处
  - 标识特征：饱和度(saturation)极高（0.50-0.67），是背景(0.22-0.30)的 2 倍
  - 攻击意图：琥珀/橙色，R和G双高，B明显偏低 (R≈80, G≈90, B≈35)
  - 防御意图：蓝色调，B偏高
  - 移动意图：绿色/黄色调，G偏高

检测方法：
  1. 在每个敌人上方垂直扫描 y_offset 70-130 范围
  2. 计算每层的平均饱和度，找饱和度尖峰 (>0.40)
  3. 在尖峰位置分析 RGB 比例，分类意图类型
  4. OCR 读取伤害数字（仅攻击意图）
"""

import numpy as np
from PIL import Image
from typing import Optional

# ── 意图区域参数（已根据实际截图校准） ───────────────────

# 搜索范围（相对地面线向上的像素偏移）
INTENT_SEARCH_TOP = 130    # grid_y - 130
INTENT_SEARCH_BOT = 70     # grid_y - 70
INTENT_SEARCH_HALF_W = 55  # 水平半宽

# 饱和度阈值：超过此值认为找到意图标识
# 真实意图饱和度 0.56-0.67，背景杂讯 0.22-0.40
SATURATION_SPIKE = 0.50

# 攻击意图的 RGB 特征（琥珀/橙色）
# R 和 G 双高，B 明显低于 R
ATTACK_R_B_RATIO = 1.25    # R > B * 1.25
ATTACK_G_B_RATIO = 1.20    # G > B * 1.20
ATTACK_R_MIN = 55          # R 最低值

# 防御意图的 RGB 特征（蓝色/青色）
# B 高于 R 和 G
DEFEND_B_R_RATIO = 1.10    # B > R * 1.10
DEFEND_B_MIN = 70

# 增益/治疗的 RGB 特征（绿色）
BUFF_G_R_RATIO = 1.10      # G > R * 1.10
BUFF_G_B_RATIO = 1.05      # G > B * 1.05
BUFF_G_MIN = 70

# 移动/位移的 RGB 特征（黄绿/亮绿）
MOVE_G_R_RATIO = 1.05      # G > R * 1.05
MOVE_G_MIN = 80
MOVE_SAT_MIN = 0.30

# 特殊/紫色的 RGB 特征
SPECIAL_B_G_RATIO = 0.90   # B 和 R 都高 (品红)
SPECIAL_R_B_DIFF = 15      # |R-B| < 15 且两者都高


def _compute_saturation(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算 HSV 中的饱和度 S = (max - min) / max"""
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    return np.where(max_rgb > 5, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0.0)


def _scan_intent_region(img: np.ndarray,
                        cx: int, grid_y: int,
                        half_w: int = INTENT_SEARCH_HALF_W) -> dict:
    """
    在敌人上方垂直扫描，找到意图标识的确切位置和颜色特征。

    Returns: {
        "found": bool,
        "y_offset": int,      # 意图标识中心的 y 偏移（相对 grid_y）
        "sat_peak": float,    # 峰值饱和度
        "r_mean": float, "g_mean": float, "b_mean": float,
        "intent_type": str,
        "confidence": float,
    }
    """
    h, w = img.shape[:2]

    best = {
        "found": False,
        "y_offset": 0,
        "sat_peak": 0.0,
        "r_mean": 0.0, "g_mean": 0.0, "b_mean": 0.0,
        "intent_type": "unknown",
        "confidence": 0.0,
    }

    saturation_profile = []  # (y_offset, avg_sat, r_mean, g_mean, b_mean)

    for y_off in range(INTENT_SEARCH_BOT, INTENT_SEARCH_TOP + 1, 3):
        y1 = max(0, grid_y - y_off - 3)
        y2 = max(y1 + 2, grid_y - y_off + 3)
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w)

        region = img[y1:y2, x1:x2]
        if region.size < 100:
            continue

        r = region[:, :, 0].astype(float)
        g = region[:, :, 1].astype(float)
        b = region[:, :, 2].astype(float)

        sat = _compute_saturation(r, g, b)
        avg_sat = float(sat.mean())

        r_mean = float(r.mean())
        g_mean = float(g.mean())
        b_mean = float(b.mean())

        saturation_profile.append((y_off, avg_sat, r_mean, g_mean, b_mean))

        if avg_sat > best["sat_peak"]:
            best["sat_peak"] = avg_sat
            best["y_offset"] = y_off
            best["r_mean"] = r_mean
            best["g_mean"] = g_mean
            best["b_mean"] = b_mean

    # 没有显著的饱和度尖峰 → 没有意图标识
    if best["sat_peak"] < SATURATION_SPIKE:
        return best

    best["found"] = True

    # ── 基于 RGB 比例分类意图类型 ──────────────────────
    r_m = best["r_mean"]
    g_m = best["g_mean"]
    b_m = best["b_mean"]

    # 计算颜色得分
    attack_score = 0.0
    defend_score = 0.0
    buff_score = 0.0
    move_score = 0.0
    special_score = 0.0

    # 攻击：R 和 G 都明显高于 B，呈琥珀色
    if r_m > ATTACK_R_MIN and r_m > b_m * ATTACK_R_B_RATIO and g_m > b_m * ATTACK_G_B_RATIO:
        attack_score = best["sat_peak"] * 1.0  # 攻击是最常见的意图

    # 防御：B 高于 R
    if b_m > DEFEND_B_MIN and b_m > r_m * DEFEND_B_R_RATIO:
        defend_score = best["sat_peak"] * 0.9

    # 增益：G 明显高于 R 和 B
    if g_m > BUFF_G_MIN and g_m > r_m * BUFF_G_R_RATIO and g_m > b_m * BUFF_G_B_RATIO:
        buff_score = best["sat_peak"] * 0.8

    # 移动：G 偏高且饱和度中等
    if g_m > MOVE_G_MIN and g_m > r_m * MOVE_G_R_RATIO and best["sat_peak"] > MOVE_SAT_MIN:
        move_score = best["sat_peak"] * 0.7

    # 特殊：R 和 B 都高（品红/紫色），且差距小
    if r_m > 70 and b_m > 60 and abs(r_m - b_m) < SPECIAL_R_B_DIFF:
        special_score = best["sat_peak"] * 0.6

    scores = {
        "attack": attack_score,
        "defend": defend_score,
        "buff": buff_score,
        "move": move_score,
        "special": special_score,
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score > 0:
        best["intent_type"] = best_type
        best["confidence"] = round(min(1.0, best_score), 3)
    else:
        # 有高饱和度但无法归类 → 默认攻击（最常见）
        best["intent_type"] = "attack"
        best["confidence"] = round(best["sat_peak"] * 0.5, 3)

    return best


def _try_ocr_damage(img: np.ndarray,
                    cx: int, grid_y: int, y_offset: int,
                    half_w: int = INTENT_SEARCH_HALF_W) -> Optional[int]:
    """
    尝试从意图区域 OCR 读取伤害数字。

    使用 pytesseract 进行数字识别。
    如果 tesseract 未安装则静默跳过。
    """
    try:
        import pytesseract
        # 检查 tesseract 是否可用（可能抛出 TesseractNotFoundError）
        pytesseract.get_tesseract_version()
    except Exception:
        return None

    h, w = img.shape[:2]

    # 裁剪意图标识周围的紧凑区域
    y1 = max(0, grid_y - y_offset - 12)
    y2 = max(y1 + 4, grid_y - y_offset + 12)
    x1 = max(0, cx - 35)
    x2 = min(w, cx + 35)

    region = img[y1:y2, x1:x2]
    if region.size < 200:
        return None

    # 转灰度
    gray = np.mean(region, axis=2).astype(np.uint8)

    # 二值化：亮色文字在暗色背景上
    mean_val = gray.mean()
    # 尝试两种极性
    for invert in [False, True]:
        if invert:
            binary = np.where(gray < mean_val * 0.8, 255, 0).astype(np.uint8)
        else:
            binary = np.where(gray > mean_val * 1.2, 255, 0).astype(np.uint8)

        # 检查是否有足够的前景像素
        fg_pct = (binary > 0).sum() / binary.size
        if fg_pct < 0.02 or fg_pct > 0.6:
            continue

        try:
            pil_img = Image.fromarray(binary)
            text = pytesseract.image_to_string(
                pil_img,
                config="--psm 7 -c tessedit_char_whitelist=0123456789"
            ).strip()

            if text and text.isdigit():
                val = int(text)
                if 1 <= val <= 99:
                    return val
        except Exception:
            continue

    return None


def read_enemy_intent(img: np.ndarray,
                      char_x_center: int,
                      grid_y: int) -> dict:
    """
    读取单个敌人上方的意图标识。

    Args:
        img: 完整游戏画面的 numpy 数组 (H, W, 3)
        char_x_center: 敌人在画面中的 x 中心像素
        grid_y: 地面线的 y 坐标

    Returns:
        {
            "action": "attack" | "defend" | "move" | "buff" | "special" | "unknown",
            "damage": int | None,
            "detail": str,
            "confidence": float,
        }
    """
    # ── 饱和度扫描 ──────────────────────────────────────
    scan = _scan_intent_region(img, char_x_center, grid_y)

    if not scan["found"]:
        return {
            "action": "unknown",
            "damage": None,
            "detail": "no intent indicator detected",
            "confidence": 0.0,
        }

    intent_type = scan["intent_type"]
    confidence = scan["confidence"]

    # ── OCR 读取伤害数字 ────────────────────────────────
    damage = None
    if intent_type == "attack":
        damage = _try_ocr_damage(
            img, char_x_center, grid_y, scan["y_offset"]
        )

    # ── 构建描述 ────────────────────────────────────────
    if intent_type == "attack":
        if damage:
            detail = f"attack for {damage} damage"
        else:
            detail = "attack (damage unknown)"
    elif intent_type == "defend":
        detail = "defend / block"
    elif intent_type == "move":
        detail = "move / reposition"
    elif intent_type == "buff":
        detail = "buff / heal"
    elif intent_type == "special":
        detail = "special ability"
    else:
        detail = "unknown action"

    return {
        "action": intent_type,
        "damage": damage,
        "detail": detail,
        "confidence": confidence,
    }


def read_all_enemy_intents(img: np.ndarray,
                           enemies: list[dict],
                           grid_y: int) -> list[dict]:
    """
    批量读取所有敌人的意图。

    Args:
        img: 完整游戏画面 (H, W, 3)
        enemies: 敌人列表，每个包含 x_center
        grid_y: 地面线 y 坐标

    Returns:
        敌人意图列表，与输入顺序一致
    """
    results = []
    for i, enemy in enumerate(enemies):
        cx = enemy.get("x_center", 0)

        intent = read_enemy_intent(img, char_x_center=cx, grid_y=grid_y)
        intent["enemy_index"] = i
        results.append(intent)
    return results
