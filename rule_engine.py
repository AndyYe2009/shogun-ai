"""
规则引擎 v11 — 意图感知 + 射程匹配 + combo + 智能走位

策略优先级：
  1. 队列有技能 → 释放（如果值得）或继续排队combo
  2. 威胁响应：敌人即将攻击 → 优先击杀 / 走位躲避
  3. 最佳技能排队：射程匹配、AOE优先、高伤害优先
  4. 多技能combo：队列不满3且有多余就绪技能 → 继续排
  5. 无就绪技能 → 移动/转身/等待
"""

from state import GameState


def decide_action(state: GameState, attack_queue_count: int = 0,
                  swap_cooldown: int = 0) -> dict:
    """
    纯函数决策——无全局状态。

    Args:
        state: 当前游戏状态
        attack_queue_count: 攻击队列中已有的技能数 (0-3)
        swap_cooldown: Wanderer换位被动冷却剩余回合数 (0=可用)
    """
    tiles = state.tiles
    enemies = [e for e in state.enemies if e.hp > 0]
    player = state.player
    ready_indices = [
        i for i, t in enumerate(tiles)
        if t.cooldown_remaining == 0 and t.damage > 0
    ]
    can_queue_more = attack_queue_count < 3

    # ─────────────────────────────────────────────────
    # 优先级 1: 队列满 → 必须释放
    # ─────────────────────────────────────────────────
    if attack_queue_count == 3:
        return {
            "reasoning": "Queue full (3/3), releasing all skills",
            "actions": [{"type": "end_turn"}],
        }

    # ─────────────────────────────────────────────────
    # 优先级 2: 队列有技能 → 释放 或 继续combo
    # ─────────────────────────────────────────────────
    if attack_queue_count > 0:
        # 如果还有就绪技能且队列不满，先尝试斩杀 combo
        if can_queue_more and ready_indices and enemies:
            # 优先尝试斩杀威胁敌人，否则打最低 HP 的敌人
            threatening = [e for e in enemies if e.next_action == "attack"]
            target = (threatening[0] if threatening
                      else min(enemies, key=lambda e: e.hp))
            lethal = _find_lethal_combo(tiles, ready_indices, target, player)
            if lethal:
                total_dmg = sum(tiles[i].damage for i in lethal)
                names = "+".join(tiles[i].name for i in lethal)
                return {
                    "reasoning": f"LETHAL: {total_dmg} dmg combo ({names}) kills "
                                 f"{target.name} (HP:{target.hp})",
                    "actions": _build_queue_actions(tiles, lethal),
                }

            # 回退：有AOE技能或多敌人场景 → combo
            has_aoe = any(tiles[i].aoe for i in ready_indices)
            many_enemies = len(enemies) >= 3
            if has_aoe and many_enemies:
                best = _pick_best_skill(tiles, ready_indices, enemies, player)
                if best is not None:
                    skill = tiles[best]
                    return {
                        "reasoning": f"Combo: queue [{best}] {skill.name} before release "
                                     f"(AOE + {len(enemies)} enemies)",
                        "actions": [
                            {"type": "reset_cursor"},
                            {"type": "queue", "tile_index": best, "tile_name": skill.name},
                        ],
                    }

        # 否则直接释放
        return {
            "reasoning": f"Release {attack_queue_count} queued skill(s)",
            "actions": [{"type": "end_turn"}],
        }

    # ─────────────────────────────────────────────────
    # 优先级 3: 队列空 → 排队技能
    # ─────────────────────────────────────────────────
    if ready_indices and enemies:
        # 注意：不再预先检查朝向！先尝试用当前朝向攻击，
        # 攻击不了时后面的 fallthrough 才会考虑转身。
        # 避免因 CV 位置数据不准而"转身背对敌人空放技能"。

        # 3a. 威胁响应：有敌人即将攻击
        threatening = [
            e for e in enemies
            if e.next_action == "attack"
        ]
        if threatening:
            # 找距离最近的威胁敌人
            nearest_threat = min(threatening, key=lambda e: abs(e.position - player.position))
            dist = abs(nearest_threat.position - player.position)

            # 找能打到这个威胁的技能
            in_range = [
                i for i in ready_indices
                if _can_hit(tiles[i], player, nearest_threat)
            ]
            if in_range:
                # 先尝试斩杀 combo — 确保击杀威胁敌人
                lethal = _find_lethal_combo(tiles, in_range, nearest_threat, player)
                if lethal:
                    total_dmg = sum(tiles[i].damage for i in lethal)
                    names = "+".join(tiles[i].name for i in lethal)
                    return {
                        "reasoning": f"LETHAL: {total_dmg} dmg combo ({names}) kills "
                                     f"threat {nearest_threat.name} (HP:{nearest_threat.hp})",
                        "actions": _build_queue_actions(tiles, lethal),
                    }

                # 无法斩杀 + 威胁相邻 → 尝试换位闪避
                if abs(nearest_threat.position - player.position) == 1:
                    swap = _try_swap(state, swap_cooldown,
                                     f"Cannot kill {nearest_threat.name}, ")
                    if swap:
                        return swap

                # 回退：单个最优技能（优先 AOE，然后高伤害）
                best = _pick_best_skill(tiles, in_range, enemies, player)
                skill = tiles[best]
                return {
                    "reasoning": f"Threat: {nearest_threat.name} about to attack! "
                                 f"Queue [{best}] {skill.name} (dmg:{skill.damage})",
                    "actions": [
                        {"type": "reset_cursor"},
                        {"type": "queue", "tile_index": best, "tile_name": skill.name},
                    ],
                }

            # 威胁不在射程内 → 检查是否因为朝向不对
            threat_in_front = (
                (player.facing == "right" and nearest_threat.position >= player.position) or
                (player.facing == "left" and nearest_threat.position <= player.position)
            )
            if threat_in_front:
                # 面朝威胁但距离不够 → 移动靠近
                move = _move_toward(player, nearest_threat, state)
                if move:
                    return move
            # 威胁在身后 → 跳到3b处理（3b会检测无面前敌人时转身）

            # 尝试换位：威胁相邻时 swap 将其移到身后
            swap = _try_swap(state, swap_cooldown,
                             f"Threat {nearest_threat.name} adjacent, ")
            if swap:
                return swap

        # 3b. 无威胁 → 打最近的敌人
        nearest = min(enemies, key=lambda e: abs(e.position - player.position))

        in_range = [
            i for i in ready_indices
            if _can_hit(tiles[i], player, nearest)
        ]

        # 3b-extra: 如果所有 in-range 技能都是远程的，但有近战AOE可用
        # → 先靠近敌人再用近战（节省远程技能的长CD）
        if in_range:
            all_ranged = all(tiles[i].range_max >= 99 for i in in_range)
            if all_ranged:
                melee_ready = [i for i in ready_indices
                              if tiles[i].range_max <= 1 and tiles[i].damage > 0]
                dist = abs(nearest.position - player.position)
                if melee_ready and dist <= 2 and dist > 0:
                    move = _move_toward(player, nearest, state)
                    if move:
                        names = ", ".join(tiles[i].name for i in melee_ready)
                        move["reasoning"] = (
                            f"Closing in to use melee [{names}] instead of ranged "
                            f"(dist={dist}, save long CD)"
                        )
                        return move

        if in_range:
            # 先尝试斩杀 combo — 确保击杀最近敌人
            lethal = _find_lethal_combo(tiles, in_range, nearest, player)
            if lethal:
                total_dmg = sum(tiles[i].damage for i in lethal)
                names = "+".join(tiles[i].name for i in lethal)
                return {
                    "reasoning": f"LETHAL: {total_dmg} dmg combo ({names}) kills "
                                 f"{nearest.name} (HP:{nearest.hp})",
                    "actions": _build_queue_actions(tiles, lethal),
                }

            # 回退：单个最优技能
            best = _pick_best_skill(tiles, in_range, enemies, player)
            skill = tiles[best]

            # Combo: 如果有多余就绪技能且队列空，考虑一次排多个
            combo_indices = [best]
            remaining = [i for i in ready_indices if i != best]
            for extra in remaining[:2]:  # 最多再排2个（总共最多3个）
                if _can_hit(tiles[extra], player, nearest) or tiles[extra].aoe:
                    combo_indices.append(extra)
                if len(combo_indices) >= 3:
                    break

            if len(combo_indices) >= 2:
                actions = [{"type": "reset_cursor"}]
                for idx in combo_indices:
                    actions.append({
                        "type": "queue",
                        "tile_index": idx,
                        "tile_name": tiles[idx].name,
                    })
                names = ", ".join(f"[{i}] {tiles[i].name}" for i in combo_indices)
                return {
                    "reasoning": f"Combo {len(combo_indices)} skills: {names}",
                    "actions": actions,
                }

            return {
                "reasoning": f"Queue [{best}] {skill.name} (dmg:{skill.damage}"
                             f"{' AOE' if skill.aoe else ''})",
                "actions": [
                    {"type": "reset_cursor"},
                    {"type": "queue", "tile_index": best, "tile_name": skill.name},
                ],
            }

        # 打不到最近的敌人 → 检查是否是朝向问题
        # 如果最近敌人在身后，尝试找面前的敌人
        front_enemies = [
            e for e in enemies
            if (player.facing == "right" and e.position >= player.position) or
               (player.facing == "left" and e.position <= player.position)
        ]
        if not front_enemies:
            # 所有敌人都在身后 → 转身
            return {"reasoning": "All enemies behind, turning around",
                    "actions": [{"type": "turn"}]}
        # 面前有敌人但不在射程 → 移动靠近最近的面前敌人
        nearest_front = min(front_enemies, key=lambda e: abs(e.position - player.position))
        dist_to_front = abs(nearest_front.position - player.position)
        if dist_to_front == 0:
            # 敌人在同一格 → 无法攻击，后退一步或换方向
            if player.position > 0:
                return {"reasoning": "Enemy at same position, backing up",
                        "actions": [{"type": "move", "direction": "left"}]}
            elif player.position < state.grid_size - 1:
                return {"reasoning": "Enemy at same position, backing up",
                        "actions": [{"type": "move", "direction": "right"}]}
            else:
                return {"reasoning": "Stuck, waiting",
                        "actions": [{"type": "wait_turn"}]}
        move = _move_toward(player, nearest_front, state)
        if move:
            return move
        # 无法移动（边缘）→ 等待冷却
        return {"reasoning": "Cannot approach, waiting",
                "actions": [{"type": "wait_turn"}]}

    # ─────────────────────────────────────────────────
    # 优先级 4: 有就绪技能但没敌人
    # ─────────────────────────────────────────────────
    if ready_indices:
        if player.position < state.grid_size - 1:
            return {"reasoning": "Advancing to find enemies",
                    "actions": [{"type": "move", "direction": "right"}]}
        return {"reasoning": "Waiting for enemies",
                "actions": [{"type": "wait_turn"}]}

    # ─────────────────────────────────────────────────
    # 优先级 5: 没就绪技能 → 走位/等待
    # ─────────────────────────────────────────────────
    if enemies:
        # 先尝试换位 — 没技能时 swap 可改变站位
        swap = _try_swap(state, swap_cooldown, "No ready skills, ")
        if swap:
            return swap
        return _move_to_safety_or_enemy(player, enemies, state)

    # 没敌人也没技能 → 前进
    if player.position < state.grid_size - 1:
        return {"reasoning": "Advancing",
                "actions": [{"type": "move", "direction": "right"}]}

    return {"reasoning": "Waiting for cooldowns",
            "actions": [{"type": "wait_turn"}]}


# ── 辅助函数 ──────────────────────────────────────────

def _can_hit(tile, player, enemy) -> bool:
    """检查技能是否能打到敌人（考虑朝向和射程）"""
    dist = enemy.position - player.position
    abs_dist = abs(dist)

    # 射程检查（dist=0 表示同一格，视为近战距离）
    effective_range_min = 0 if abs_dist == 0 else tile.range_min
    if abs_dist < effective_range_min or abs_dist > tile.range_max:
        return False

    # ── 双向AOE检测 ────────────────────────────────────
    # 近战AOE技能（range_max<=1 且 aoe=True）通常攻击前后两格
    # 如：旋风斩(Swirl)、阴阳铁扇(Twin Tessen)、急转身(Sharp Turn)
    # 这些技能无视朝向，能同时打到身前和身后的敌人
    is_melee_aoe = (
        tile.aoe and tile.range_max <= 1 and tile.damage > 0
    )
    if is_melee_aoe:
        return True  # 前后都能打

    # 朝向检查：大多数技能只能打面前的敌人
    # 背击(Back Strike)打身后，其他打面前
    has_backstab = any("back" in (e or "").lower() for e in getattr(tile, 'effects', []) or [])
    if has_backstab:
        # 背击：敌人在身后才能打
        if player.facing == "right" and dist >= 0:
            return False
        if player.facing == "left" and dist <= 0:
            return False
    else:
        # 正常技能：敌人在面前才能打
        if player.facing == "right" and dist < 0:
            return False
        if player.facing == "left" and dist > 0:
            return False

    return True


def _pick_best_skill(tiles, indices, enemies, player) -> int | None:
    """从候选技能中选最优：AOE优先 > 高伤害 > 低冷却 > 近战优先"""
    if not indices:
        return None

    def score(i: int) -> tuple:
        t = tiles[i]
        # 计算能打到的敌人数
        hit_count = sum(1 for e in enemies if _can_hit(t, player, e))
        # AOE额外加分
        aoe_bonus = 100 if t.aoe else 0
        # 高伤害加分
        dmg_score = t.damage * 10
        # 能打到多个敌人加分
        multi_hit = hit_count * 50

        # ── 近战/远程平衡 ──────────────────────────────
        # 如果敌人在近战范围(距离<=1)，近战技能应与远程同权
        # 防止AI过度依赖远程技能（手里剑/箭矢）而忽略近战AOE
        has_adjacent_enemy = any(
            abs(e.position - player.position) <= 1
            for e in enemies
        )
        is_melee = t.range_max <= 1
        is_ranged = t.range_max >= 99

        melee_bonus = 0
        if has_adjacent_enemy and is_melee and t.damage > 0:
            # 近战范围有敌人时，近战技能+30（冷却短、不浪费远程CD）
            melee_bonus = 30
        elif has_adjacent_enemy and is_ranged:
            # 有近身敌人时，远程技能轻微扣分（留着远程打远处的敌人）
            melee_bonus = -15

        # ── 斩杀加成 ────────────────────────────────────
        # 伤害足够击杀敌人时额外加分（+10/可能击杀的敌人）
        lethal_bonus = 0
        for e in enemies:
            if _can_hit(t, player, e) and t.damage >= e.hp > 0:
                lethal_bonus += 10

        return (
            aoe_bonus + multi_hit + dmg_score + melee_bonus + lethal_bonus,
            t.damage,
            -t.cooldown_max,
        )

    return max(indices, key=score)


def _check_facing(player, enemies) -> dict | None:
    """检查是否需要转身才能打到敌人"""
    if not enemies:
        return None

    all_pos = [e.position for e in enemies]
    enemies_right = sum(1 for p in all_pos if p > player.position)
    enemies_left = sum(1 for p in all_pos if p < player.position)

    # 所有敌人都在身后 → 必须转身
    if player.facing == "right" and enemies_right == 0 and enemies_left > 0:
        return {"reasoning": "Turn left to face enemies (all behind)",
                "actions": [{"type": "turn"}]}
    if player.facing == "left" and enemies_left == 0 and enemies_right > 0:
        return {"reasoning": "Turn right to face enemies (all behind)",
                "actions": [{"type": "turn"}]}

    # 不再因"最近敌人在身后"就转身 —— 当两面都有敌人时，
    # 转身会导致背对原来的敌人，产生"空放技能"的问题
    return None


def _move_toward(player, target_enemy, state) -> dict | None:
    """向目标移动一步"""
    dist = target_enemy.position - player.position

    if dist < 0 and player.position > 0:
        return {"reasoning": f"Move left toward {target_enemy.name}",
                "actions": [{"type": "move", "direction": "left"}]}
    if dist > 0 and player.position < state.grid_size - 1:
        return {"reasoning": f"Move right toward {target_enemy.name}",
                "actions": [{"type": "move", "direction": "right"}]}
    return None


def _move_to_safety_or_enemy(player, enemies, state) -> dict:
    """没有就绪技能时的站位策略：远离威胁 / 靠近敌人"""
    # 先检查是否有敌人即将攻击
    threatening = [e for e in enemies if e.next_action == "attack"]
    if threatening:
        nearest_threat = min(threatening, key=lambda e: abs(e.position - player.position))
        # 远离威胁
        dist = nearest_threat.position - player.position
        if dist >= 0 and player.position > 0:
            return {"reasoning": "Back away from threat",
                    "actions": [{"type": "move", "direction": "left"}]}
        if dist <= 0 and player.position < state.grid_size - 1:
            return {"reasoning": "Back away from threat",
                    "actions": [{"type": "move", "direction": "right"}]}
        # 无路可退，检查是否需要转身
        face = _check_facing(player, enemies)
        if face:
            return face
        return {"reasoning": "Cornered, waiting",
                "actions": [{"type": "wait_turn"}]}

    # 无威胁 → 靠近最近敌人
    nearest = min(enemies, key=lambda e: abs(e.position - player.position))
    move = _move_toward(player, nearest, state)
    if move:
        return move

    return {"reasoning": "Waiting for cooldowns",
            "actions": [{"type": "wait_turn"}]}


# ── Wanderer 换位被动 ───────────────────────────────

def _can_swap(state: GameState, swap_cooldown: int = 0) -> tuple[bool, str, str]:
    """
    检查 Wanderer 换位是否可用。面朝相邻敌人 + 背后有空位 = 可换位。

    Returns:
        (can_swap, direction, detail)
        direction = "left" 或 "right"; detail = 原因描述
    """
    if swap_cooldown > 0:
        return False, "", f"swap on cooldown ({swap_cooldown}t left)"

    p = state.player
    pos = p.position
    facing = p.facing
    grid = state.grid_size

    # 检查面朝方向的相邻敌人
    target_pos = pos + (1 if facing == "right" else -1)
    if target_pos < 0 or target_pos >= grid:
        return False, "", f"facing grid edge (pos {target_pos})"

    adjacent_enemy = None
    for e in state.enemies:
        if e.position == target_pos and e.hp > 0:
            adjacent_enemy = e
            break
    if not adjacent_enemy:
        return False, "", f"no adjacent enemy at pos {target_pos}"

    # 检查背后空位
    behind_pos = pos + (-1 if facing == "right" else 1)
    if behind_pos < 0 or behind_pos >= grid:
        return False, "", f"behind pos {behind_pos} OOB (cornered)"

    for e in state.enemies:
        if e.position == behind_pos and e.hp > 0:
            return False, "", f"behind occupied by {e.name} at pos {behind_pos}"

    return True, facing, (
        f"swap with {adjacent_enemy.name}(HP:{adjacent_enemy.hp}) → "
        f"enemy to pos {behind_pos}, player to pos {target_pos}"
    )


def _try_swap(state: GameState, swap_cooldown: int = 0,
              reason_prefix: str = "") -> dict | None:
    """尝试生成换位决策。不可用则返回 None。"""
    can, direction, detail = _can_swap(state, swap_cooldown)
    if can:
        return {
            "reasoning": f"{reason_prefix}SWAP {direction}: {detail}",
            "actions": [{"type": "move", "direction": direction, "swap": True}],
        }
    return None


# ── 斩杀组合计算 ──────────────────────────────────────

def _find_lethal_combo(tiles, ready_indices, target_enemy, player,
                        max_skills: int = 3) -> list[int] | None:
    """
    找到能正好斩杀目标敌人的最小技能组合。

    算法：
      1. 只考虑能打到目标的就绪技能
      2. 依次尝试 1/2/3 技能组合
      3. 偏好：技能数最少 > 溢出伤害最低 > 总冷却最短

    Returns:
        技能索引列表 [idx, ...] 或 None（无可行组合）
    """
    target_hp = target_enemy.hp
    if target_hp <= 0:
        return None

    # 过滤：只保留能命中目标的技能
    hittable = [
        i for i in ready_indices
        if _can_hit(tiles[i], player, target_enemy)
    ]
    if not hittable:
        return None

    # 按 伤害/冷却比 排序（高伤害低冷却优先）
    def _efficiency(idx: int) -> float:
        t = tiles[idx]
        return t.damage / max(t.cooldown_max, 1)

    hittable.sort(key=_efficiency, reverse=True)

    # ── 尝试单技能斩杀 ──────────────────────────────
    for i in hittable:
        if tiles[i].damage >= target_hp:
            return [i]

    if max_skills < 2 or len(hittable) < 2:
        return None

    # ── 尝试双技能 combo ────────────────────────────
    best: list[int] | None = None
    best_overkill = 999
    best_cd = 999
    n = len(hittable)
    for a in range(n):
        for b in range(a + 1, n):
            total = tiles[hittable[a]].damage + tiles[hittable[b]].damage
            if total >= target_hp:
                overkill = total - target_hp
                cd_sum = tiles[hittable[a]].cooldown_max + tiles[hittable[b]].cooldown_max
                if (overkill < best_overkill or
                        (overkill == best_overkill and cd_sum < best_cd)):
                    best_overkill = overkill
                    best_cd = cd_sum
                    best = [hittable[a], hittable[b]]

    if best:
        return best

    if max_skills < 3 or len(hittable) < 3:
        return None

    # ── 尝试三技能 combo ────────────────────────────
    best_overkill = 999
    best_cd = 999
    best = None
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                total = (tiles[hittable[a]].damage +
                         tiles[hittable[b]].damage +
                         tiles[hittable[c]].damage)
                if total >= target_hp:
                    overkill = total - target_hp
                    cd_sum = (tiles[hittable[a]].cooldown_max +
                              tiles[hittable[b]].cooldown_max +
                              tiles[hittable[c]].cooldown_max)
                    if (overkill < best_overkill or
                            (overkill == best_overkill and cd_sum < best_cd)):
                        best_overkill = overkill
                        best_cd = cd_sum
                        best = [hittable[a], hittable[b], hittable[c]]

    return best


def _build_queue_actions(tiles, combo_indices: list[int]) -> list[dict]:
    """将技能索引列表转换为排队动作序列 (含 reset_cursor)"""
    actions = [{"type": "reset_cursor"}]
    for idx in combo_indices:
        actions.append({
            "type": "queue",
            "tile_index": idx,
            "tile_name": tiles[idx].name,
        })
    return actions


# ── 兼容旧接口 ──────────────────────────────────────────
# main.py 可能不带 attack_queue_count 调用，默认 0

def reset_parity():
    """兼容旧代码 —— 现在无需重置，保留空函数避免 import 报错"""
    pass
