"""
屏幕解析器 v4 — 针对 Shogun Showdown 优化
使用像素分析可靠地提取关键游戏状态。

核心思路：
- 找到地面线，然后在地面上方窄带区域检测角色
- 用血条颜色判断 HP 状态
- 用技能栏色彩判断技能冷却
- 用精灵左右半区分析检测朝向
- 支持技能数据库查表
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from state import GameState, PlayerState, EnemyState, TileState

# ── 技能数据库 ─────────────────────────────────────────
_skill_db = None


def _load_skill_db() -> dict:
    """加载技能数据库"""
    global _skill_db
    if _skill_db is not None:
        return _skill_db
    db_path = Path(__file__).parent / "skill_database.json"
    if db_path.exists():
        with open(db_path, "r", encoding="utf-8") as f:
            _skill_db = json.load(f)
    else:
        _skill_db = {"skills": {}, "characters": {}}
    return _skill_db


def lookup_skill(name: str) -> dict | None:
    """
    在技能数据库中查找技能。支持英文名和中文名。

    Returns: TileState 所需的字段 dict，如果未找到则返回 None
    """
    if not name or not name.strip():
        return None
    db = _load_skill_db()
    tiles = db.get("tiles", {}) or db.get("skills", {})

    result = None

    # 精确匹配
    if name in tiles:
        result = tiles[name]

    # 模糊匹配：大小写不敏感
    if result is None:
        name_lower = name.lower().strip()
        for tile_name, tile_data in tiles.items():
            if tile_name.lower() == name_lower:
                result = tile_data
                break

    # 中文名匹配
    if result is None:
        for tile_name, tile_data in tiles.items():
            if tile_data.get("name_zh", "") == name:
                result = tile_data
                break

    # 部分匹配
    if result is None:
        name_lower = name.lower().strip()
        for tile_name, tile_data in tiles.items():
            if name_lower in tile_name.lower() or tile_name.lower() in name_lower:
                result = tile_data
                break

    # 解析 _ref 引用（别名条目指向主条目）
    if result and "_ref" in result:
        ref_name = result["_ref"]
        if ref_name in tiles:
            result = tiles[ref_name]

    return result


def analyze_screenshot(img: Image.Image) -> dict:
    """
    分析游戏截图。返回可用于决策的结构化数据。
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    gray = np.mean(arr, axis=2)

    # ── 0. 预处理：仅在全桌面截图中裁剪，窗口截图跳过 ──
    # 全桌面截图宽度通常 > 1920 且包含桌面UI元素
    # 窗口/全屏游戏截图 = 整个画面都是游戏，不需要裁剪
    if w > 2000:
        r_ch = arr[:, :, 0].astype(float)
        g_ch = arr[:, :, 1].astype(float)
        b_ch = arr[:, :, 2].astype(float)
        max_rgb = np.maximum(np.maximum(r_ch, g_ch), b_ch)
        min_rgb = np.minimum(np.minimum(r_ch, g_ch), b_ch)
        sat = np.where(max_rgb > 5, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)

        # 底部 20%：技能栏区域
        skill_search_top = int(h * 0.82)
        skill_search = sat[skill_search_top:h, :]
        skill_col_sat = skill_search.mean(axis=0)

        # 找高饱和度列（>0.04 = 有彩色图标）的最宽连续区域
        has_color = skill_col_sat > 0.04
        runs = []
        in_run = False; start = 0
        for i in range(len(has_color)):
            if has_color[i] and not in_run:
                start = i; in_run = True
            elif not has_color[i] and in_run:
                if i - start > 60:
                    runs.append((start, i))
                in_run = False
        if in_run and w - start > 60:
            runs.append((start, w))

        if runs:
            best = max(runs, key=lambda r: r[1] - r[0])
            bar_left, bar_right = best
            bar_center = (bar_left + bar_right) // 2
            game_width = int((bar_right - bar_left) * 3.0)
            game_left = max(0, bar_center - game_width // 2)
            game_right = min(w, bar_center + game_width // 2)
            game_top = int(h * 0.02)
            game_bottom = h
            arr = arr[game_top:game_bottom, game_left:game_right, :]
            gray = np.mean(arr, axis=2)
            h, w = arr.shape[:2]

    result = {
        'image_size': (w, h),
        'player': {'hp_pct': 100, 'x_center': int(w * 0.25), 'x_range': (int(w * 0.2), int(w * 0.3)),
                   'grid_pos': 1, 'shield': False, 'facing': 'right', 'floor_color': 'unknown',
                   'green_px': 0, 'red_px': 0, 'hp_color': 'unknown', 'width': 40, 'gray_pct': 0},
        'enemies': [],
        'skills': [],
        'grid_size': 7,
        'enemy_count': 0,
        'danger_level': 'low',
    }

    # ── 1. 先找地面线 ────────────────────────────────
    r_ch = arr[:, :, 0].astype(float)
    g_ch = arr[:, :, 1].astype(float)
    b_ch = arr[:, :, 2].astype(float)
    max_rgb = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    min_rgb = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    sat = np.where(max_rgb > 10, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)

    # 用地板颜色定位地面线
    grid_y = _find_floor_rgb(arr)
    result['grid_y'] = grid_y

    # ── 2. 在地面线上方窄带扫描HP条 ───────────────────
    # HP条通常在地面线上方 2-20px，每条HP条约 40-80px 宽
    hp_band_top = max(0, grid_y - 20)
    hp_band_bot = min(h, grid_y - 2)
    band_height = hp_band_bot - hp_band_top + 1

    # 逐列统计绿色 HP 像素数量
    # 级联阈值：先用严格值，不够再降（远处敌人 HP 条较暗）
    def _scan_hp_band(g_min: int) -> tuple:
        counts = np.zeros(w, dtype=int)
        for y in range(hp_band_top, hp_band_bot + 1):
            hi = sat[y, :] > 0.25
            green = (hi & (g_ch[y, :] > g_min) &
                     (g_ch[y, :] > r_ch[y, :] * 1.15) &
                     (g_ch[y, :] > b_ch[y, :] * 1.4))
            red = (hi & (r_ch[y, :] > g_min) &
                   (r_ch[y, :] > g_ch[y, :] * 1.15) &
                   (r_ch[y, :] > b_ch[y, :] * 1.4))
            counts += (green | red).astype(int)
        mask = counts >= 3
        indices = np.where(mask)[0]
        regions = []
        if len(indices) > 5:
            s = indices[0]; p = indices[0]
            for idx in indices[1:]:
                if idx - p > 12:
                    w_c = p - s
                    if 30 < w_c < 140:
                        regions.append({
                            'x_start': int(s), 'x_end': int(p),
                            'x_center': int((s + p) // 2),
                            'width': int(p - s),
                        })
                    s = idx
                p = idx
            w_c = p - s
            if 30 < w_c < 140:
                regions.append({
                    'x_start': int(s), 'x_end': int(p),
                    'x_center': int((s + p) // 2),
                    'width': int(p - s),
                })
        return regions, len(indices)

    char_regions, hp_total = _scan_hp_band(60)
    if len(char_regions) == 0:
        # 严格阈值完全找不到 → 降级重试（远处敌人 HP 条较暗）
        char_regions, hp_total = _scan_hp_band(40)
    # 合并太近的区域（< 60px）
    char_regions.sort(key=lambda r: r['x_center'])
    merged = []
    for r in char_regions:
        if merged and r['x_center'] - merged[-1]['x_center'] < 70:
            merged[-1]['x_end'] = r['x_end']
            merged[-1]['x_center'] = (merged[-1]['x_start'] + merged[-1]['x_end']) // 2
            merged[-1]['width'] = merged[-1]['x_end'] - merged[-1]['x_start']
        else:
            merged.append(r)
    char_regions = merged
    char_regions.sort(key=lambda r: r['x_center'])

    # ── 拆分宽区域：>85px 的区域可能包含 2 个相邻角色 ──
    _split = []
    for r in char_regions:
        if r['width'] > 85:
            mid = (r['x_start'] + r['x_end']) // 2
            _split.append({
                'x_start': r['x_start'], 'x_end': mid,
                'x_center': (r['x_start'] + mid) // 2,
                'width': mid - r['x_start'],
            })
            _split.append({
                'x_start': mid, 'x_end': r['x_end'],
                'x_center': (mid + r['x_end']) // 2,
                'width': r['x_end'] - mid,
            })
        else:
            _split.append(r)
    char_regions = _split

    # ── 3. 分析每个角色的HP ────────────────────────────
    all_chars_temp = []
    for i, region in enumerate(char_regions[:6]):
        cx = region['x_center']
        cw = region['width']
        hp_hw = max(15, int(cw * 0.9))

        # HP条区域：血条在角色头顶上方（地面线以上 30-60px）
        hp_top = max(0, grid_y - 65)
        hp_bottom = min(h, grid_y - 25)
        hp_x1 = max(0, cx - hp_hw)
        hp_x2 = min(w, cx + hp_hw)
        hp_region = arr[hp_top:hp_bottom, hp_x1:hp_x2]

        # ── 检测脚下平台颜色（灰=玩家，橙=敌人）─────
        floor_color = _detect_floor_color(arr, cx, grid_y)
        is_orange = floor_color == "orange"

        hp_info = _analyze_hp(hp_region, is_orange_platform=is_orange)

        # 跳过无效角色：HP 检测不到任何像素的 → 不是角色，是 UI/地面噪声
        if hp_info['pct'] <= 0 or hp_info['color'] == 'invalid':
            continue

        char_info = {
            'x_center': cx,
            'x_range': (region['x_start'], region['x_end']),
            'width': region['width'],
            'hp_pct': hp_info['pct'],
            'hp_color': hp_info['color'],
            'green_px': hp_info.get('green_px', 0),
            'red_px': hp_info.get('red_px', 0),
            'shield': False,
            'is_player': floor_color == "gray",  # 灰色平台 = 玩家
            'floor_color': floor_color,
            'gray_pct': 1.0 if floor_color == "gray" else 0.0,
        }
        all_chars_temp.append(char_info)

    # ── 硬上限：游戏最多 1 玩家 + 3 敌人 = 4 角色 ──
    # 按 HP 像素总量排序，保留最可信的
    all_chars_temp.sort(key=lambda c: c.get('green_px', 0) + c.get('red_px', 0), reverse=True)
    MAX_CHARS = 4
    if len(all_chars_temp) > MAX_CHARS:
        all_chars_temp = all_chars_temp[:MAX_CHARS]

    # ── 识别玩家：灰色平台 = 玩家，橙色平台 = 敌人 ──
    # 这是最可靠的视觉特征——平台颜色由游戏引擎决定，与位置无关
    player_candidates = [c for c in all_chars_temp if c['floor_color'] == "gray"]
    enemy_candidates = [c for c in all_chars_temp if c['floor_color'] != "gray"]

    if player_candidates:
        # 灰色平台上只应该有一个角色（玩家）
        result['player'] = player_candidates[0]
        enemies = enemy_candidates + player_candidates[1:]
    elif all_chars_temp:
        # 平台颜色全部检测不到 → 光线/特效干扰
        # 不做猜测，标记第一个为玩家，其余为敌人
        # parse_to_gamestate 会用 tracked_player_pos 做最终纠正
        result['player'] = all_chars_temp[0]
        result['player']['floor_color'] = 'unknown'
        enemies = all_chars_temp[1:]
    else:
        enemies = []

    for c in enemies:
        c['is_player'] = False
    result['enemies'] = enemies
    result['enemy_count'] = len(enemies)

    # ── 3.5. 检测玩家朝向 (通过精灵分析) ──────────────
    p = result['player']
    px1, px2 = p['x_range']
    # 角色精灵区域: 地面线以上 15-65px（避开地面亮色干扰）
    sprite_top = max(0, grid_y - 65)
    sprite_bottom = max(sprite_top + 10, grid_y - 15)
    sprite_left = max(0, px1 - 2)
    sprite_right = min(w, px2 + 2)
    sprite_region = arr[sprite_top:sprite_bottom, sprite_left:sprite_right]
    detected_facing = _detect_facing(sprite_region)
    result['player']['facing'] = detected_facing
    result['player']['facing_confidence'] = 'cv_gradient'

    # ── 4. 估算网格位置 ────────────────────────────────
    all_chars = [result['player']] + result['enemies']
    all_chars.sort(key=lambda c: c['x_center'])

    if len(all_chars) > 0:
        n = len(all_chars)
        result['grid_size'] = max(5, min(9, n + 3))
        grid_slots = result['grid_size']

        # 居中排列：角色组在网格中间
        start_pos = max(0, (grid_slots - n) // 2)
        for i, char in enumerate(all_chars):
            char['grid_pos'] = min(start_pos + i, grid_slots - 1)

    # ── 4.5. 读取敌人意图 ────────────────────────────────
    try:
        from intent_reader import read_all_enemy_intents
        enemy_intents = read_all_enemy_intents(arr, result['enemies'], grid_y)
        for i, intent in enumerate(enemy_intents):
            if i < len(result['enemies']):
                result['enemies'][i]['intent'] = intent
    except Exception as e:
        # 意图读取失败不影响其他解析
        for i in range(len(result['enemies'])):
            result['enemies'][i]['intent'] = {
                "action": "unknown",
                "damage": None,
                "detail": f"intent read error: {e}",
                "confidence": 0.0,
            }

    # ── 5. 分析技能栏 ──────────────────────────────────
    # 技能卡在底部 15% 区域内，但并非均匀分布在全宽上
    # 先找到实际有内容的列范围，再在其中定位 6 个槽位
    skill_search_top = int(h * 0.85)
    skill_search_bot = int(h * 0.92)
    search_band = sat[skill_search_top:skill_search_bot, :]
    col_sat = search_band.mean(axis=0)

    # 找技能栏的实际左右边界（有饱和度内容的列）
    content_cols = np.where(col_sat > 0.05)[0]
    if len(content_cols) > 80:
        bar_left = content_cols[0]
        bar_right = content_cols[-1]
        bar_width = bar_right - bar_left
        slot_width = bar_width / 6.0

        skill_top = int(h * 0.85)
        skill_region = arr[skill_top:int(h * 0.95), :]
        skill_gray = np.mean(skill_region, axis=2)
    else:
        # 回退：使用底部 80% 区域均分
        bar_left = 0
        bar_right = w
        slot_width = w / 6.0
        skill_top = int(h * 0.85)
        skill_region = arr[skill_top:h, :]
        skill_gray = np.mean(skill_region, axis=2)

    for i in range(6):
        # 在检测到的技能栏范围内均分 6 个槽位
        x1 = int(bar_left + i * slot_width)
        x2 = int(bar_left + (i + 1) * slot_width)
        # 使用中心 60% 避免边缘污染
        margin = int(slot_width * 0.2)
        cx1 = max(0, x1 + margin)
        cx2 = min(w, x2 - margin)

        if cx2 <= cx1:
            cx1, cx2 = x1, x2

        slot_gray = skill_gray[:, cx1:cx2]
        slot_rgb = skill_region[:, cx1:cx2, :]

        if slot_rgb.size < 30:
            result['skills'].append({
                'index': i, 'status': 'empty', 'ready': False,
                'cooldown_remaining': 0, 'cooldown_max': 2,
                'bright_pct': 0.0, 'high_sat_pct': 0.0,
            })
            continue

        r_s = slot_rgb[:, :, 0].astype(float)
        g_s = slot_rgb[:, :, 1].astype(float)
        b_s = slot_rgb[:, :, 2].astype(float)
        max_s = np.maximum(np.maximum(r_s, g_s), b_s)
        min_s = np.minimum(np.minimum(r_s, g_s), b_s)
        saturation = np.where(max_s > 5, (max_s - min_s) / (max_s + 1e-6), 0)

        high_sat_pct = float(np.mean(saturation > 0.25))
        bright_pct = float(np.mean(slot_gray > 100))
        dark_pct = float(np.mean(slot_gray < 25))

        # 空槽：亮背景但无彩色（bright 高 + sat 极低）
        # 就绪：有彩色图标（sat 高）
        # 冷却：灰白图标（中亮度 + 低饱和度）
        bright_high = float(np.mean(slot_gray > 80))
        sat_very_low = float(np.mean(saturation > 0.15))

        if bright_high > 0.40 and sat_very_low < 0.02:
            status = 'empty'
        elif dark_pct > 0.50:
            status = 'empty'
        elif high_sat_pct > 0.04 and bright_pct > 0.02:
            status = 'ready'
        elif high_sat_pct > 0.015:
            status = 'cooldown'
            cooldown_remaining = 1
        else:
            status = 'empty'

        result['skills'].append({
            'index': i,
            'status': status,
            'ready': status == 'ready',
            'cooldown_remaining': cooldown_remaining if status == 'cooldown' else 0,
            'cooldown_max': 2,  # 默认，实际值需从游戏数据读取
            'bright_pct': bright_pct,
            'high_sat_pct': high_sat_pct,
        })

    # ── 6. 威胁评估 ────────────────────────────────────
    player_hp = result['player']['hp_pct']
    enemy_count = result['enemy_count']
    enemy_hp_total = sum(e['hp_pct'] for e in result['enemies'])

    if player_hp < 25:
        result['danger_level'] = 'critical'
    elif player_hp < 50 or enemy_hp_total > 300:
        result['danger_level'] = 'high'
    elif enemy_count >= 3:
        result['danger_level'] = 'medium'
    else:
        result['danger_level'] = 'low'

    return result


def _detect_floor_color(arr: np.ndarray, cx: int, grid_y: int,
                        half_w: int = 25) -> str:
    """
    检测角色脚下的平台颜色。

    玩家 = 灰色平台 (R≈G≈B)
    敌人 = 橙色平台 (R>G, R>B)

    Returns: "gray" | "orange" | "unknown"
    """
    h, w = arr.shape[:2]
    # 采样地面线上下各 8px 的窄带
    y1 = max(0, grid_y - 5)
    y2 = min(h, grid_y + 8)
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w)

    region = arr[y1:y2, x1:x2]
    if region.size < 50:
        return "unknown"

    r = region[:, :, 0].astype(float)
    g = region[:, :, 1].astype(float)
    b_chan = region[:, :, 2].astype(float)
    brightness = (r + g + b_chan) / 3

    # 只分析足够亮的像素（忽略阴影/暗色区域）
    bright = brightness > 30
    if bright.sum() < 20:
        return "unknown"

    r_b = r[bright].mean()
    g_b = g[bright].mean()
    b_b = b_chan[bright].mean()

    # 灰色判定：R、G、B 三者接近 (每对差距 < 20)
    rg_diff = abs(r_b - g_b)
    rb_diff = abs(r_b - b_b)
    gb_diff = abs(g_b - b_b)
    is_gray = rg_diff < 20 and rb_diff < 25 and gb_diff < 25

    # 橙色判定：R 明显高于 G，R 明显高于 B
    is_orange = r_b > g_b * 1.05 and r_b > b_b * 1.15

    if is_gray and not is_orange:
        return "gray"
    elif is_orange:
        return "orange"
    elif rg_diff < 25 and rb_diff < 30:
        return "gray"  # 偏灰
    else:
        return "orange"  # 偏暖色


def _find_floor(gray: np.ndarray, h: int) -> int:
    """
    找地面线的 y 坐标。

    策略：在角色的 RGB 数组上，找灰色和橙色平台同时出现的行，
    平台顶部边缘 = 角色站立的地面线。
    这需要 RGB 数据，所以改为在 analyze_screenshot 中调用。
    """
    # 灰度版本的简化回退：找亮度方差最大的行
    search_top = h // 4
    search_bot = h * 2 // 3
    band = gray[search_top:search_bot, :]
    band_h = band.shape[0]

    best_score = 0
    best_row = h // 2
    for i in range(2, band_h - 2):
        row = band[i, :]
        row_avg = float(row.mean())
        if row_avg < 12 or row_avg > 210:
            continue
        diffs = np.abs(np.diff(row))
        diff_sorted = np.sort(diffs)[::-1]
        top_n = max(10, len(diff_sorted) // 4)
        score = float(diff_sorted[:top_n].mean())
        if score > best_score:
            best_score = score
            best_row = search_top + i

    return best_row


def _find_floor_rgb(arr: np.ndarray) -> int:
    """
    用 RGB 平台颜色定位地面线（比灰度版更准）。

    找灰色和橙色平台同时存在的最上方行 = 地面线。
    避开底部技能栏区域。
    """
    h, w = arr.shape[:2]
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    # 限制搜索范围：h/6 到 h*2/3（避开顶部UI和底部技能栏）
    y1 = max(h // 6, 100)
    y2 = min(h * 2 // 3, h - int(h * 0.25))

    best_y = None
    best_score = 0

    for y in range(y1, y2, 2):
        row_r = r[y, :]
        row_g = g[y, :]
        row_b = b[y, :]
        bright = (row_r + row_g + row_b) / 3 > 25

        if bright.sum() < 100:
            continue

        br_r = row_r[bright]
        br_g = row_g[bright]
        br_b = row_b[bright]

        # 灰色像素：R≈G≈B
        gray_mask = (abs(br_r - br_g) < 12) & (abs(br_r - br_b) < 25)
        gray_pct = float(gray_mask.sum()) / max(bright.sum(), 1)

        # 橙色像素：R>G 且 R>B
        orange_mask = (br_r > br_g * 1.05) & (br_r > br_b * 1.2)
        orange_pct = float(orange_mask.sum()) / max(bright.sum(), 1)

        # 综合得分：灰橙同时出现，且灰占比高更可信
        if gray_pct > 0.10 and orange_pct > 0.03:
            score = gray_pct + orange_pct
            if score > best_score:
                best_score = score
                best_y = y

    if best_y is not None:
        return best_y

    # 回退：h//2
    return h // 2


def _group_columns(bright_mask: np.ndarray, min_gap: int = 30) -> list:
    """将亮列分组为角色区域"""
    if not bright_mask.any():
        return []

    # 找到所有亮列的索引
    indices = np.where(bright_mask)[0]

    # 分组
    regions = []
    start = indices[0]
    prev = indices[0]

    for idx in indices[1:]:
        if idx - prev > min_gap:
            regions.append({
                'x_start': int(start),
                'x_end': int(prev),
                'x_center': int((start + prev) / 2),
                'width': int(prev - start),
            })
            start = idx
        prev = idx

    # 最后一组
    regions.append({
        'x_start': int(start),
        'x_end': int(prev),
        'x_center': int((start + prev) / 2),
        'width': int(prev - start),
    })

    return regions


def _analyze_hp(region: np.ndarray, is_orange_platform: bool = False,
                ref_region: np.ndarray | None = None) -> dict:
    """
    分析血条。

    用背景减法：对比HP条区域和下方纯平台区域，
    只在HP区域有而平台区域没有的彩色像素才算HP。
    """
    if region.size < 10:
        return {'pct': 100, 'color': 'unknown', 'green_px': 0, 'red_px': 0}

    r = region[:, :, 0].astype(float)
    g_ch = region[:, :, 1].astype(float)
    b = region[:, :, 2].astype(float)

    max_rgb = np.maximum(np.maximum(r, g_ch), b)
    min_rgb = np.minimum(np.minimum(r, g_ch), b)
    sat = np.where(max_rgb > 10, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)
    high_sat = sat > 0.25

    # ── 背景减法 ────────────────────────────────────
    if ref_region is not None and ref_region.size > 10:
        ref_r = ref_region[:, :, 0].astype(float)
        ref_g = ref_region[:, :, 1].astype(float)
        ref_b = ref_region[:, :, 2].astype(float)

        # 平均参考颜色（平台本色）
        ref_r_mean = float(ref_r.mean())
        ref_g_mean = float(ref_g.mean())
        ref_b_mean = float(ref_b.mean())

        # 只统计与平台颜色明显不同的像素
        r_diff = np.abs(r - ref_r_mean)
        g_diff = np.abs(g_ch - ref_g_mean)
        b_diff = np.abs(b - ref_b_mean)
        not_platform = (r_diff + g_diff + b_diff) > 25  # 与平台颜色不同
    else:
        not_platform = np.ones_like(r, dtype=bool)

    hp_pixels = high_sat & not_platform

    # 绿HP
    green = int((hp_pixels & (g_ch > r * 1.05) & (g_ch > b * 1.4)).sum())
    # 红HP
    if is_orange_platform:
        # 橙色平台：只信任绿色（红色被平台污染无法区分）
        red = 0
    else:
        # 灰色平台：R>G*1.03即可（平台R≈G，稍有偏差就是HP）
        red = int((hp_pixels & (r > g_ch * 1.03) & (r > b * 1.4)).sum())

    total = green + red
    if total > 4:
        pct = int(green / total * 100)
        if green > red * 2:
            color = 'green'
        elif red > green * 2:
            color = 'red'
        else:
            color = 'yellow'
        return {'pct': max(1, min(100, pct)), 'color': color, 'green_px': green, 'red_px': red}

    # 回退：不用背景减法
    green = int((high_sat & (g_ch > r * 1.05) & (g_ch > b * 1.4)).sum())
    if is_orange_platform:
        red = 0  # 橙色平台红色不可靠
    else:
        red = int((high_sat & (r > g_ch * 1.03) & (r > b * 1.4)).sum())

    total = green + red
    if total > 4:
        pct = int(green / total * 100)
        color = 'green' if green > red else ('red' if red > green else 'yellow')
        return {'pct': max(1, min(100, pct)), 'color': color, 'green_px': green, 'red_px': red}

    # 最终回退：检测不到任何HP像素 → 这不是有效角色
    # 之前默认 85% 导致假角色（地面装饰、UI元素）带着虚假HP进入敌人列表
    return {'pct': 0, 'color': 'invalid', 'green_px': 0, 'red_px': 0}


def _detect_shield(region: np.ndarray) -> bool:
    """检测蓝色护盾"""
    if region.size < 50:
        return False

    b = region[:, :, 2].astype(float)
    r = region[:, :, 0].astype(float)
    g = region[:, :, 1].astype(float)

    blue_pixels = (
        (b > 80) & (b > r * 1.15) & (b > g * 1.1)
    ).sum()

    return blue_pixels > 20


def _analyze_sprite_color(region: np.ndarray) -> dict:
    """分析角色精灵的主色调，用于区分玩家和敌人"""
    if region.size < 100:
        return {'blue_ratio': 0, 'red_ratio': 0, 'dominant': 'unknown'}

    r = region[:, :, 0].astype(float)
    g = region[:, :, 1].astype(float)
    b_chan = region[:, :, 2].astype(float)
    brightness = r + g + b_chan

    # 只分析足够亮的像素（忽略背景暗色）
    bright_mask = brightness > 60
    if bright_mask.sum() < 20:
        return {'blue_ratio': 0, 'red_ratio': 0, 'dominant': 'unknown'}

    r_bright = r[bright_mask]
    g_bright = g[bright_mask]
    b_bright = b_chan[bright_mask]

    total_bright = r_bright.sum() + g_bright.sum() + b_bright.sum()
    if total_bright < 1:
        return {'blue_ratio': 0, 'red_ratio': 0, 'dominant': 'unknown'}

    blue_ratio = float(b_bright.sum() / total_bright)
    red_ratio = float(r_bright.sum() / total_bright)
    green_ratio = float(g_bright.sum() / total_bright)

    # 判断主导颜色
    if blue_ratio > red_ratio and blue_ratio > green_ratio:
        dominant = 'blue'
    elif red_ratio > blue_ratio and red_ratio > green_ratio:
        dominant = 'red'
    else:
        dominant = 'neutral'

    return {
        'blue_ratio': blue_ratio,
        'red_ratio': red_ratio,
        'green_ratio': green_ratio,
        'dominant': dominant,
    }


def _detect_facing(sprite_region: np.ndarray) -> str:
    """
    通过分析角色精灵的水平梯度分布检测朝向。

    原理：角色面朝的方向有更多视觉细节（面部、武器），
    水平梯度（相邻像素列之间的变化）更大。

    Returns: "left" | "right"
    """
    if sprite_region.size < 100:
        return "right"  # 默认

    h_s, w_s = sprite_region.shape[:2]
    if w_s < 10:
        return "right"

    gray = np.mean(sprite_region, axis=2)
    mid = w_s // 2

    # 水平梯度 — 前方（面部+武器）边缘细节更丰富
    h_grad = np.abs(np.diff(gray, axis=1))
    left_grad = h_grad[:, :mid].sum()
    right_grad = h_grad[:, mid - 1:].sum()

    # 如果精灵太窄（<40px），梯度不可靠，回退到默认
    if w_s < 40:
        return "right"

    if right_grad > left_grad * 1.08:
        return "right"    # 右侧细节多 → 面朝右
    elif left_grad > right_grad * 1.08:
        return "left"     # 左侧细节多 → 面朝左

    # 差异不明显，回退到亮度中心分析
    x_weights = np.arange(w_s)
    com = np.sum(gray * x_weights) / max(np.sum(gray), 1)
    if com > mid * 1.02:
        return "right"
    elif com < mid * 0.98:
        return "left"

    return "right"  # 最终默认


def is_main_menu(img: Image.Image) -> bool:
    """
    检测是否为游戏非战斗画面（主菜单/角色选择/商店/过场等）。

    窗口截图策略：游戏窗口内，大部分画面都是战斗或准战斗状态。
    只有明显特征不匹配时才判定为非战斗。
    """
    arr = np.array(img)
    h, w = arr.shape[:2]

    # 1. 底部技能栏区域：检查是否有任何彩色内容
    skill_top = int(h * 0.85)
    skill_region = arr[skill_top:h, :]

    r = skill_region[:, :, 0].astype(float)
    g = skill_region[:, :, 1].astype(float)
    b = skill_region[:, :, 2].astype(float)
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_rgb > 5, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0)
    # 底部有任何彩色内容 → 很可能是战斗中的技能栏
    has_skill_content = float(np.mean(sat > 0.20)) > 0.01

    # 2. 中部区域亮度
    mid = arr[h // 3: h * 2 // 3, :]
    mid_brightness = float(np.mean(mid))
    is_very_dark = mid_brightness < 15

    # 3. 判定：没有技能内容 且 非常暗 → 非战斗
    if not has_skill_content and is_very_dark:
        return True

    return False


def parse_to_gamestate(img: Image.Image, turn_number: int = 0,
                       skill_names: list[str] | None = None,
                       tracked_facing: str | None = None,
                       tracked_player_pos: int | None = None) -> GameState:
    """
    主接口：截图 → GameState

    Args:
        img: 游戏截图
        turn_number: 当前回合数
        skill_names: 可选的技能名称列表（6个，索引0-5）。
        tracked_facing: 外部追踪的朝向（"left"|"right"），优先于CV检测。
        tracked_player_pos: 外部追踪的玩家位置，用于确认CV检测的玩家。
    """
    a = analyze_screenshot(img)

    # 玩家
    from config import PLAYER_MAX_HP, ENEMY_MAX_HP

    p = a['player']

    # ── HP：像素比例 × 配置的最大HP ────────────────────
    player_current_hp = max(1, int(PLAYER_MAX_HP * p['hp_pct'] / 100))
    player_max_hp = PLAYER_MAX_HP

    # ── 用追踪位置校正玩家识别 ──────────────────────────
    # executor 确定性追踪每次 move/turn，比 CV 更可靠
    # 当 CV 平台颜色检测失败时，追踪位置是唯一可靠的识别依据
    player_grid_pos = p.get('grid_pos', 3)  # Wanderer 默认出生在中间位置3
    if tracked_player_pos is not None:
        # 先 clamp tracked_pos 到合理范围 [0, grid_size-1]
        grid_sz = a.get('grid_size', 7)
        clamped_pos = max(0, min(tracked_player_pos, grid_sz - 1))

        # 如果 CV 没检测到灰色平台（floor_color 不确定），
        # 说明 CV 可能在乱猜——用追踪位置强制纠正
        if p.get('floor_color') == 'unknown' and a['enemies']:
            # 找所有检测到的角色中离追踪位置最近的那个 = 玩家
            all_detected = [p] + a['enemies']
            best_dist = 999
            best_char = p
            for char in all_detected:
                dist = abs(char.get('grid_pos', 99) - clamped_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_char = char
            if best_char is not p:
                for e in a['enemies']:
                    if e is best_char:
                        e['is_player'] = True
                        p['is_player'] = False
                        p = best_char
                        break

        # 追踪位置 = 真实位置（优先级最高）
        player_grid_pos = clamped_pos

    # 朝向：直接信任 executor 追踪（CV 朝向检测不可靠）
    # 游戏开始默认朝右，executor 跟踪每次 turn 操作保持准确
    if tracked_facing and tracked_facing in ('left', 'right'):
        facing = tracked_facing
    else:
        # 回退：默认朝右（游戏开始时的朝向）
        facing = "right"

    player = PlayerState(
        character="Wanderer",
        hp=player_current_hp,
        max_hp=player_max_hp,
        shield=10 if p.get('shield') else 0,
        position=player_grid_pos,
        facing=facing,
    )

    # 敌人
    enemies = []
    for i, e in enumerate(a['enemies']):
        # 基于可检测特征命名（位置 + 意图 + HP），不做虚假精确识别
        intent = e.get('intent', {})
        intent_type = intent.get('action', 'unknown')
        hp_pct = e.get('hp_pct', 100)

        # 意图标记
        intent_tag = ""
        if intent_type == 'attack':
            intent_tag = "A"
        elif intent_type == 'defend':
            intent_tag = "D"
        elif intent_type == 'buff':
            intent_tag = "B"
        elif intent_type == 'move':
            intent_tag = "M"

        # HP 等级标记
        if hp_pct <= 25:
            hp_tag = "L"  # Low
        elif hp_pct >= 80:
            hp_tag = "F"  # Full
        else:
            hp_tag = ""

        # 构建名称: Enemy_位置_意图HP标记
        pos = e.get('grid_pos', i)
        name = f"E@{pos}"
        if intent_tag:
            name += f"[{intent_tag}]"
        if hp_tag:
            name += f"({hp_tag})"

        # ── HP：像素比例 × 配置的最大HP ────────────────
        enemy_current_hp = max(1, int(ENEMY_MAX_HP * e['hp_pct'] / 100))
        enemy_max_hp = ENEMY_MAX_HP

        # ── 使用意图识别结果 ──────────────────────────
        intent = e.get('intent', {})
        action_type = intent.get('action', 'unknown')
        intent_detail = intent.get('detail', 'no intent data')
        damage = intent.get('damage')
        intent_conf = intent.get('confidence', 0.0)

        # 构建意图描述
        if action_type == 'attack':
            next_action = "attack"
            if damage:
                next_action_detail = f"{damage} damage (conf: {intent_conf:.0%})"
            else:
                next_action_detail = f"damage incoming (conf: {intent_conf:.0%})"
        elif action_type == 'defend':
            next_action = "defend"
            next_action_detail = f"blocking/defending (conf: {intent_conf:.0%})"
        elif action_type == 'move':
            next_action = "move"
            next_action_detail = f"repositioning (conf: {intent_conf:.0%})"
        elif action_type == 'buff':
            next_action = "buff"
            next_action_detail = f"buffing/healing (conf: {intent_conf:.0%})"
        elif action_type == 'special':
            next_action = "special"
            next_action_detail = f"special ability (conf: {intent_conf:.0%})"
        else:
            next_action = "unknown"
            next_action_detail = f"intent unclear, HP ~{e.get('hp_pct', '?')}%"

        enemies.append(EnemyState(
            name=name,
            hp=enemy_current_hp,
            max_hp=enemy_max_hp,
            position=e.get('grid_pos', i + 2),
            next_action=next_action,
            next_action_detail=next_action_detail,
        ))

    # 技能
    # 如果提供了 skill_names，使用数据库查表获取属性
    if skill_names:
        skill_names_list = list(skill_names)  # 确保是 list
    else:
        skill_names_list = [None] * 6

    # ── 从数据库读取角色默认技能（按CV检测的非空槽位顺序匹配） ──
    # 不再使用硬编码 default_backup[slot_index]，因为不同角色/加载的技能排列不同
    _char_db = _load_skill_db()
    _char_info = _char_db.get("characters", {}).get("Wanderer", {})
    _char_default_names = _char_info.get("default_tiles", ["旋风斩", "箭矢"])
    _char_default_skills = []
    for _cn in _char_default_names:
        _s = lookup_skill(_cn)
        if _s:
            _char_default_skills.append(_s)

    # 如果角色数据库也没有（极端情况），回退到 Wanderer 已知的初始技能
    if not _char_default_skills:
        _char_default_skills = [
            {'name': '旋风斩', 'damage': 2, 'range_min': 1, 'range_max': 1, 'cooldown_max': 3, 'aoe': True, 'effects': []},
            {'name': '箭矢',   'damage': 2, 'range_min': 1, 'range_max': 99, 'cooldown_max': 5, 'aoe': False, 'effects': []},
        ]

    # 建立映射: 按CV检测到的非空槽位顺序，依次分配角色默认技能
    _non_empty_slots = [i for i, s in enumerate(a['skills']) if s.get('status') != 'empty']
    _slot_to_default = {}
    for _idx, _slot_i in enumerate(_non_empty_slots):
        if _idx < len(_char_default_skills):
            _slot_to_default[_slot_i] = _char_default_skills[_idx]

    tiles = []
    for i, s in enumerate(a['skills']):
        name = None
        dmg = 6
        rng_min = 1
        rng_max = 1
        cd_max = 2
        aoe = False
        effects = []

        # 尝试从 skill_names 获取名称并查数据库
        identified_name = None
        if i < len(skill_names_list) and skill_names_list[i]:
            identified_name = skill_names_list[i]

        if identified_name:
            db_skill = lookup_skill(identified_name)
            if db_skill:
                name = db_skill.get("name", identified_name)
                dmg = db_skill.get("damage", 6)
                rng_min = db_skill.get("range_min", 1)
                rng_max = db_skill.get("range_max", 1)
                cd_max = db_skill.get("cooldown_max", 2)
                aoe = db_skill.get("aoe", False)
                effects = db_skill.get("effects", [])
            else:
                # 数据库中未找到，使用识别到的名称
                name = identified_name
        elif s.get('status') == 'empty':
            name = "(empty)"

        # 如果仍然没有名称，使用角色默认技能映射
        if not name or name == "(empty)":
            if i in _slot_to_default:
                default = _slot_to_default[i]
                name = default.get('name', default.get('name_zh', f'Skill{i+1}'))
                if name == "(empty)":
                    name = "(empty)"
                else:
                    dmg = default.get('damage', 6)
                    rng_max = default.get('range_max', 1)
                    rng_min = default.get('range_min', 1)
                    cd_max = default.get('cooldown_max', 2)
                    aoe = default.get('aoe', False)
                    effects = default.get('effects', [])
            else:
                # 超出角色默认技能数量的额外技能槽（游戏中后续拾取的技能）
                # 用CV检测到的信息给一个合理的默认值
                name = "(empty)" if s.get('status') == 'empty' else f"Skill{i+1}"

        status = s.get('status', 'ready' if s.get('ready') else 'cooldown')

        if status == 'empty' or name == "(empty)":
            cooldown_rem = 99
            cooldown_max = 99
        elif status == 'cooldown':
            cooldown_rem = s.get('cooldown_remaining', cd_max)
            cooldown_max = cd_max
        else:
            cooldown_rem = 0
            cooldown_max = cd_max

        tiles.append(TileState(
            name=name,
            cooldown_remaining=cooldown_rem,
            cooldown_max=cooldown_max,
            damage=dmg,
            range_min=rng_min,
            range_max=rng_max if dmg > 0 else 0,
            aoe=aoe,
            effects=effects,
        ))

    # ── 后解析校验 ─────────────────────────────────
    # 游戏实际最多 3 个敌人 + 1 个玩家 = 4 个角色
    # 如果解析出更多，说明 CV 产生了大量误判，标记为非战斗
    enemy_count = len(enemies)
    if enemy_count > 3:
        # 敌人数量不可能超过 3，这是 CV 误判
        # 返回一个 game_over=False 但无敌人的空状态，触发生效的 Space 按键推进
        return GameState(
            turn_number=turn_number,
            grid_size=7,
            player=PlayerState(character="Wanderer", hp=PLAYER_MAX_HP, max_hp=PLAYER_MAX_HP,
                              position=1, facing="right"),
            enemies=[],
            tiles=[TileState(name="(empty)", cooldown_remaining=99, cooldown_max=99, damage=0)
                   for _ in range(6)],
            attack_queue=[],
            consumables=[],
            game_over=False,
            victory=False,
        )

    grid_size = max(5, a['enemy_count'] + 3)
    return GameState(
        turn_number=turn_number,
        grid_size=grid_size,
        player=player,
        enemies=enemies,
        tiles=tiles,
        attack_queue=[],
        consumables=[],
        game_over=False,
        victory=False,
    )


if __name__ == "__main__":
    from pathlib import Path

    for turn_num in [1, 5, 10, 15]:
        test_img = Path(f"screenshots/turn_{turn_num:04d}.png")
        if not test_img.exists():
            continue

        img = Image.open(test_img)
        print(f"\n=== Turn {turn_num} ===")
        print(f"  Size: {img.size}")

        a = analyze_screenshot(img)

        print(f"  Grid Y: {a.get('grid_y', '?')}")
        print(f"  Player: HP~{a['player']['hp_pct']}%, "
              f"grid_pos={a['player'].get('grid_pos', '?')}, "
              f"shield={a['player']['shield']}")

        for i, e in enumerate(a['enemies']):
            print(f"  Enemy {i + 1}: HP~{e['hp_pct']}%, "
                  f"grid_pos={e.get('grid_pos', '?')}")

        skill_status = ' '.join(
            ['R' if s['ready'] else 'C' for s in a['skills']]
        )
        print(f"  Skills: [{skill_status}]")
        print(f"  Danger: {a['danger_level']} (enemies: {a['enemy_count']})")
