"""
Shogun AI - 全自动游戏循环
截图 → CV解析画面 → Ollama决策 → 执行 → 循环
"""

import time
import sys
from pathlib import Path

from capture import capture_screenshot, get_window_rect
from executor import (
    execute_action, validate_action, get_tracked_facing,
    get_tracked_player_pos, reset_facing, get_attack_queue_count,
    get_skill_cooldown_tracker, tick_cooldowns,
    flush_queued_skills, get_queued_skill_indices,
    clear_queued_skills, get_swap_cooldown,
    hold_key, press_key,
)
from config import (
    AI_BACKEND,
    WAIT_AFTER_ENEMY_TURN,
    WAIT_BETWEEN_CYCLES,
    BACKGROUND_MODE,
    RESTART_HOLD_W,
    RESTART_WAIT,
)
from state import state_to_text
from scoring import get_scorer, reset_scorer
from playbook import get_playbook, extract_fingerprint


def auto_loop(max_turns: int = 500, save_screenshots: bool = True,
              scan_skills: bool = False, quiet: bool = False):
    """
    全自动游戏循环。

    核心流程：
    1. CV 解析截图 → GameState（不依赖 AI 视觉模型）
    2. Ollama 文本模型做决策
    3. 执行键盘操作
    """
    def _log(*args, level="info"):
        """条件输出：quiet模式下只输出重要信息"""
        if quiet and level == "info":
            return
        print(*args)

    _log("\n" + "=" * 60)
    _log(f"  Shogun AI - 全自动模式")
    _log(f"  画面解析: OpenCV 像素分析")
    _log(f"  决策引擎: {AI_BACKEND}")
    _log(f"  安静模式: {'ON' if quiet else 'OFF'}")

    # Playbook 统计
    pb = get_playbook()
    stats = pb.stats()
    _log(f"  战术手册: {stats['total']} 条经验 | 平均得分: {stats['avg_score']}")
    _log("=" * 60)

    # ── 初始化 ──────────────────────────────────────────
    from screen_parser import parse_to_gamestate, is_main_menu

    if AI_BACKEND == "rule":
        from rule_engine import decide_action as decide_rule
        print("Decision engine: Rule-based (fast, no LLM)")
        use_rule_engine = True
        decide_fn = None

    elif AI_BACKEND == "ollama":
        from ollama_client import (
            decide_action_ollama, check_ollama_available,
            identify_skills_ollama,
        )
        from rule_engine import decide_action as decide_rule

        print("Checking Ollama...")
        ollama_ok = check_ollama_available()
        if ollama_ok:
            print("Ollama connected")
            use_rule_engine = False
            decide_fn = decide_action_ollama
        else:
            print("WARNING: Ollama not available, using rule engine")
            use_rule_engine = True
            decide_fn = None

    elif AI_BACKEND == "anthropic":
        from vision import parse_screenshot as parse_anthropic
        from decision import decide_action as decide_anthropic

        parse_fn = parse_anthropic
        decide_fn = decide_anthropic
    else:
        print(f"ERROR: Unknown backend: {AI_BACKEND}")
        return

    # ── 确认游戏窗口 ────────────────────────────────────
    try:
        rect = get_window_rect()
        print(f"Game window: {rect['width']}x{rect['height']} "
              f"at ({rect['left']}, {rect['top']})")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Please start Shogun Showdown first!")
        print("Window title should contain 'ShogunShowdown'")
        return

    # ── 准备截图目录 ────────────────────────────────────
    screenshot_dir = Path("screenshots") if save_screenshots else None
    if screenshot_dir:
        screenshot_dir.mkdir(exist_ok=True)

    # ── 技能扫描（可选，仅 --scan-skills 时启用） ──────
    skill_names: list[str] = []
    if scan_skills and AI_BACKEND == "ollama":
        print("\nScanning skill bar (vision model)...")
        try:
            scan_screenshot = capture_screenshot()
            identified = identify_skills_ollama(scan_screenshot)
            if identified:
                skill_names = [
                    s.get("name", "") if not s.get("empty", False) else ""
                    for s in identified[:6]
                ]
                while len(skill_names) < 6:
                    skill_names.append("")
                print(f"  Raw result: {skill_names}")
                from screen_parser import lookup_skill
                found_count = sum(1 for name in skill_names if name and lookup_skill(name))
                if found_count >= 1:
                    print(f"  Verified: {found_count}/6 skills matched database")
                else:
                    print(f"  WARNING: 0 skills matched, using CV-only defaults")
                    skill_names = []
            else:
                print("  Skill scan returned empty, using CV-only detection")
        except Exception as e:
            print(f"  Skill scan error: {e}")
            print(f"  Using CV-only defaults from database")
    else:
        print("\nSkill scan: SKIPPED (CV-only, using database defaults)")

    # 初始化朝向追踪、评分、规则引擎
    reset_facing()
    reset_scorer()
    try:
        from rule_engine import reset_parity
        reset_parity()
    except: pass

    # ── 后台模式：静默启动，不抢焦点 ────────────────────
    if BACKGROUND_MODE:
        print("  Background mode ON — window stays unfocused")
    else:
        print("\nActivating game window...")
        try:
            from capture import get_game_window
            win = get_game_window()
            win.activate()
            print(f"  Window activated: {win.title}")
        except Exception as e:
            print(f"  WARNING: Could not activate window: {e}")
    print("Starting in 2 seconds...")
    time.sleep(2)

    turn_count = 0
    _stuck_history = []  # 卡死检测：记录最近3回合状态
    _pending_snapshot = None  # 延迟评分：上回合的快照数据
    _pending_playbook = get_playbook()  # playbook引用

    try:
        while turn_count < max_turns:
            turn_count += 1
            _log(f"\n{'─' * 40}")
            _log(f"Turn {turn_count}")

            # ── 1. 截图 ──────────────────────────────────
            _log("  Capturing...")
            save_path = str(screenshot_dir / f"turn_{turn_count:04d}.png") if screenshot_dir else None
            screenshot = capture_screenshot(save_path=save_path)

            # ── 每回合冷却 -1 ──────────────────────────
            tick_cooldowns()

            # ── 1.5. 主菜单/非战斗画面检测 ────────────────
            # 如果不是在战斗中（主菜单、角色选择、商店等），
            # 尝试按 Space 确认进入游戏
            if is_main_menu(screenshot):
                print("  Non-combat screen detected — pressing Space to advance ...")
                press_key("space", hold=0.1)
                time.sleep(3.0)
                # 有时候需要再按一次 Enter
                press_key("enter", hold=0.1)
                time.sleep(2.0)
                _stuck_history.clear()
                continue

            # ── 2. CV 解析画面 ────────────────────────────
            _log("  Analyzing (CV)...")
            try:
                if AI_BACKEND in ("ollama", "rule"):
                    # 使用 CV 解析器，不依赖 AI 视觉模型
                    game_state = parse_to_gamestate(
                        screenshot, turn_number=turn_count,
                        skill_names=skill_names if skill_names else None,
                        tracked_facing=get_tracked_facing(),
                        tracked_player_pos=get_tracked_player_pos(),
                    )
                else:
                    # Anthropic 后端仍可用视觉 API
                    game_state = parse_fn(screenshot)
                    # 用追踪朝向覆盖
                    game_state.player.facing = get_tracked_facing()
                    # 如果识别了技能名称，更新 game_state
                    if skill_names:
                        from screen_parser import lookup_skill
                        for i, tile in enumerate(game_state.tiles):
                            if i < len(skill_names) and skill_names[i]:
                                db = lookup_skill(skill_names[i])
                                if db:
                                    tile.name = db.get("name", skill_names[i])
                                    tile.damage = db.get("damage", tile.damage)
                                    tile.range_min = db.get("range_min", tile.range_min)
                                    tile.range_max = db.get("range_max", tile.range_max)
                                    tile.cooldown_max = db.get("cooldown_max", tile.cooldown_max)
                                    tile.aoe = db.get("aoe", tile.aoe)
                                    tile.effects = db.get("effects", [])
                                else:
                                    tile.name = skill_names[i]
            except Exception as e:
                print(f"  WARNING: Parse failed: {e}")
                if screenshot_dir:
                    screenshot.save(str(screenshot_dir / f"error_turn_{turn_count:04d}.png"))
                time.sleep(2)
                continue

            # ── 检查游戏结束 ──────────────────────────────
            if game_state.game_over:
                # 结算最后一回合的分数
                scorer = get_scorer()
                player_hp = game_state.player.hp
                enemy_list = [
                    {"id": f"e{i}", "hp": e.hp}
                    for i, e in enumerate(game_state.enemies)
                ]
                final_score = scorer.compute_delta(
                    player_hp, enemy_list,
                    victory=game_state.victory,
                    game_over=not game_state.victory,
                )
                if game_state.victory:
                    # 胜利 → 结算并停止
                    print(f"\n{'=' * 60}")
                    print(f"  VICTORY! Final Score: {scorer.total}")
                    print(f"  Total Kills: {scorer.kills} | Damage: {scorer.damage_dealt} | Turns: {turn_count}")
                    print(f"{'=' * 60}")
                    break
                else:
                    # 失败 → 结算，然后长按 W 重新开始
                    print(f"\n{'─' * 40}")
                    print(f"  ☠ DEFEATED (turn {turn_count}) — Score: {scorer.total}")
                    print(f"  Restarting: holding W for {RESTART_HOLD_W}s ...")
                    hold_key("w", RESTART_HOLD_W)
                    print(f"  Waiting {RESTART_WAIT}s for reload ...")
                    time.sleep(RESTART_WAIT)
                    # 确认新游戏（Space 或 Enter）
                    print(f"  Pressing Space/Enter to start new run ...")
                    press_key("space", hold=0.15)
                    time.sleep(1.5)
                    press_key("enter", hold=0.15)
                    time.sleep(3.0)  # 等待游戏加载

                    # ── 重置所有追踪状态 ──────────────────
                    reset_facing()
                    reset_scorer()
                    try:
                        from rule_engine import reset_parity
                        reset_parity()
                    except: pass
                    _stuck_history.clear()
                    _pending_snapshot = None

                    # 激活窗口（仅前台模式）
                    if not BACKGROUND_MODE:
                        try:
                            from capture import get_game_window
                            win = get_game_window()
                            win.activate()
                        except Exception:
                            pass

                    print(f"  ✅ New game started! Continuing ...")
                    print(f"{'─' * 40}")
                    continue

            # ── 评分结算（延迟一回合） ──────────────────────
            # 用当前解析到的HP作为"上一回合执行后"的结果
            scorer = get_scorer()
            current_hp = game_state.player.hp
            current_enemies = [
                {"id": f"e{i}", "hp": e.hp}
                for i, e in enumerate(game_state.enemies)
            ]

            # 如果有上回合的快照，计算上回合得分
            if _pending_snapshot is not None:
                try:
                    # 检测上回合是否释放了技能（用于空放惩罚）
                    had_release = any(
                        a.get("type") == "end_turn"
                        for a in _pending_snapshot.get("actions", [])
                    )
                    result = scorer.compute_delta(
                        current_hp, current_enemies, skills_released=had_release
                    )
                    if result["delta"] != 0:
                        sign = "+" if result["delta"] > 0 else ""
                        parts = []
                        if result.get("dealt_this_turn", 0):
                            parts.append(f"{result['dealt_this_turn']} dmg dealt (+{result['dealt_this_turn'] * 5})")
                        if result.get("kills_this_turn", 0):
                            parts.append(f"{result['kills_this_turn']} kills (+{result['kills_this_turn'] * 10})")
                        if result.get("taken_this_turn", 0):
                            parts.append(f"{result['taken_this_turn']} dmg taken ({result['taken_this_turn'] * -30})")
                        if result.get("miss_penalty", 0):
                            parts.append(f"miss penalty ({result['miss_penalty']})")
                        if result.get("action_reward", 0):
                            parts.append(f"{result['action_reward']} action bonus")
                        print(f"  Score (turn {turn_count - 1}): {sign}{result['delta']} ({', '.join(parts)}) → Total: {result['total']}")
                    else:
                        print(f"  Score (turn {turn_count - 1}): no change → Total: {result['total']}")

                    # 记录成功经验到战术手册
                    if result["delta"] >= 1:
                        sn = _pending_snapshot
                        save_actions = [
                            a for a in sn.get("actions", [])
                            if a.get("type") not in ("reset_cursor",)
                        ]
                        if save_actions and sn.get("fingerprint"):
                            _pending_playbook.add(
                                fingerprint=sn["fingerprint"],
                                actions=save_actions,
                                reasoning=sn.get("reasoning", ""),
                                score_delta=result["delta"],
                            )
                except Exception:
                    pass

            # 保存本回合快照（下回合结算）
            scorer.snapshot(current_hp, current_enemies)

            # ── 打印状态 ──────────────────────────────────
            p = game_state.player
            enemies_desc = ", ".join(
                f"{e.name}(HP:{e.hp})" for e in game_state.enemies
            )
            _log(f"  Player {p.character} | HP:{p.hp}/{p.max_hp} | Pos:{p.position}")
            _log(f"  Enemies: {enemies_desc or 'none'}")
            _log(f"  Skills: {len([t for t in game_state.tiles if t.cooldown_remaining == 0])}/{len(game_state.tiles)} ready")
            print(f"  Score: {scorer.total} | Kills: {scorer.kills}")

            # ── 卡死检测 ────────────────────────────────
            # 连续3回合 HP、敌人数量、位置都没变化 → 触发恢复
            _stuck_history.append({
                "hp": game_state.player.hp,
                "enemy_count": len(game_state.enemies),
                "pos": game_state.player.position,
                "queue_count": get_attack_queue_count(),
            })
            if len(_stuck_history) > 3:
                _stuck_history.pop(0)
            if len(_stuck_history) == 3:
                all_same = all(
                    h["hp"] == _stuck_history[0]["hp"] and
                    h["enemy_count"] == _stuck_history[0]["enemy_count"] and
                    h["pos"] == _stuck_history[0]["pos"]
                    for h in _stuck_history
                )
                queue_stuck = all(h["queue_count"] == 0 for h in _stuck_history)
                if all_same and queue_stuck:
                    print("  STUCK DETECTED! Trying recovery: move right + reset cursor")
                    try:
                        from executor import reset_skill_cursor, move_right
                        reset_skill_cursor()
                        move_right()
                        _stuck_history.clear()
                    except Exception:
                        pass
                    time.sleep(WAIT_AFTER_ENEMY_TURN)
                    continue

            # ── 2.5. 提取场景指纹（用于 playbook） ──────────
            fp = extract_fingerprint(game_state)

            # ── 2.6. 技能就绪扫描（仅确认，不覆盖CV判断） ──
            # 扫描器通过光标图标位移来判断冷却，存在误判风险
            # 策略：只用于确认CV已判定就绪的技能，不覆盖CV的冷却判定
            # （CV像素饱和度分析对冷却的判定更可靠）
            try:
                from skill_scanner import scan_skill_cooldowns
                _log("  Scanning skills...")
                ready_list = scan_skill_cooldowns()
                ready_count = sum(ready_list)
                _log(f"  Skills ready: {ready_count}/6 ({ready_list})")
                if any(ready_list):
                    overrides = []
                    for i, t in enumerate(game_state.tiles):
                        if i < len(ready_list):
                            if ready_list[i] and t.cooldown_remaining > 0:
                                # 扫描器说就绪但CV说冷却 → 不信任扫描器
                                # CV的像素饱和度分析对冷却态（灰白图标）更可靠
                                overrides.append(f"[{i}]{t.name}")
                            elif ready_list[i] and t.cooldown_remaining == 0:
                                # 扫描器确认CV的就绪判定 → 保持，无需操作
                                pass
                    if overrides:
                        _log(f"  Scanner→ready but CV→CD, ignoring: {', '.join(overrides)}")
            except Exception as e:
                _log(f"  Skill scan skipped: {e}")

            # ── 2.7. 冷却追踪器兜底 ──────────────────────
            # 确定性追踪：刚才用过的技能一定在冷却中
            # 当CV误判就绪时，tracker作为安全底线强制修正
            cooldown_tracker = get_skill_cooldown_tracker()
            tracker_overrides = []
            for i, t in enumerate(game_state.tiles):
                tracked_cd = cooldown_tracker[i]
                if tracked_cd > t.cooldown_remaining:
                    tracker_overrides.append(
                        f"[{i}]{t.name}(CV:{t.cooldown_remaining}→tracker:{tracked_cd})"
                    )
                    t.cooldown_remaining = tracked_cd
            if tracker_overrides:
                _log(f"  CD tracker overrides: {', '.join(tracker_overrides)}")

            # ── 3. 决策 ──────────────────────────────────
            _log("  Deciding...")
            try:
                if use_rule_engine:
                    # 规则引擎：直接用 game_state 决策
                    from rule_engine import decide_action as decide_rule
                    decision = decide_rule(game_state,
                                          attack_queue_count=get_attack_queue_count(),
                                          swap_cooldown=get_swap_cooldown())
                else:
                    # LLM：构建状态文本
                    score_info = {
                        "total": scorer.total,
                        "kills": scorer.kills,
                        "damage_dealt": scorer.damage_dealt,
                        "damage_taken": scorer.damage_taken,
                        "action_count": scorer.action_count,
                        "action_bonus": scorer.action_bonus,
                        "idle_penalty": scorer.idle_penalty,
                        "last_delta": scorer.turn_deltas[-1] if scorer.turn_deltas else 0,
                        "last_kills": 0,
                        "last_dealt": 0,
                        "last_taken": 0,
                    }
                    state_text = state_to_text(game_state, score_info=score_info)

                    # ── 战术手册检索 ──────────────────────────
                    playbook = get_playbook()
                    matches = playbook.search(fp)
                    if matches:
                        print(f"  Playbook: {len(matches)} similar case(s) found "
                              f"(best score +{matches[0].score_delta})")
                    else:
                        print(f"  Playbook: no match (db size: {playbook.stats()['total']})")

                    decision = decide_fn(state_text, playbook_examples=matches if matches else None)
            except Exception as e:
                print(f"  WARNING: Decision failed: {e}")
                # 兜底：不能卡住不动，至少wait一回合
                from executor import execute_action as _fallback_exec
                print(f"  FALLBACK: executing wait_turn")
                _fallback_exec({"type": "wait_turn"})
                time.sleep(WAIT_AFTER_ENEMY_TURN)
                continue

            reasoning = decision.get("reasoning", "")
            actions = decision.get("actions", [])
            _log(f"  Reasoning: {reasoning}")

            # ── 修正两回合机制 ──────────────────────────────
            # 排队(Enter)和释放(Space)必须在不同回合！
            # 但同回合可以排多个技能（combo），只要不混入 release

            queue_count = get_attack_queue_count()

            # 1. 空动作列表 → 尝试排队就绪技能，而非盲等
            if not actions:
                ready = [t for t in game_state.tiles if t.cooldown_remaining == 0 and t.damage > 0]
                if ready and queue_count < 3:
                    best = ready[0]  # 简单取第一个就绪的
                    best_idx = game_state.tiles.index(best)
                    actions = [
                        {"type": "reset_cursor"},
                        {"type": "queue", "tile_index": best_idx, "tile_name": best.name},
                    ]
                    _log("  (auto-queued ready skill — queue was empty)")
                else:
                    actions.append({"type": "wait_turn"})
                    _log("  (auto-added wait_turn — no actions and no ready skills)")

            # 2. 如果既排队又释放 → 只保留排队或释放（不能同时）
            has_queue = any(a.get("type") == "queue" for a in actions)
            has_release = any(a.get("type") == "end_turn" for a in actions)

            if has_queue and has_release:
                if queue_count > 0:
                    # 队列已有技能 → 保留释放，移除排队（本回合释放）
                    actions = [a for a in actions if a.get("type") != "queue"]
                    _log("  (removed queue — releasing this turn)")
                else:
                    # 队列为空 → 保留排队，移除释放（下回合再释放）
                    actions = [a for a in actions if a.get("type") != "end_turn"]
                    _log("  (removed end_turn — queue first, release next turn)")

            # 3. 队列满3个时不能继续排
            if queue_count >= 3 and has_queue:
                actions = [a for a in actions if a.get("type") != "queue"]
                if not actions:
                    actions.append({"type": "end_turn"})
                _log("  (queue full 3/3 — must release first)")

            # 4. 有释放但队列为空 → 改成排队或等待
            if has_release and not has_queue and queue_count == 0:
                actions = [a for a in actions if a.get("type") != "end_turn"]
                ready = [t for t in game_state.tiles if t.cooldown_remaining == 0 and t.damage > 0]
                if ready and queue_count < 3:
                    best = ready[0]
                    best_idx = game_state.tiles.index(best)
                    actions = [
                        {"type": "reset_cursor"},
                        {"type": "queue", "tile_index": best_idx, "tile_name": best.name},
                    ]
                    _log("  (converted empty release → queue ready skill)")
                elif not actions:
                    actions.append({"type": "wait_turn"})
                    _log("  (converted empty release → wait — no ready skills)")

            # 5. 排队了但没有释放 → 正常，下回合再释放
            if has_queue and not has_release:
                _log(f"  (queued {sum(1 for a in actions if a.get('type') == 'queue')} skill(s) — release next turn)")

            # ── 4. 验证并执行 ──────────────────────────────
            # 后台模式：跳过窗口焦点检查，直接执行
            if not BACKGROUND_MODE:
                # 确保游戏窗口有焦点
                try:
                    from capture import get_game_window
                    win = get_game_window()
                    if not win.isActive:
                        win.activate()
                except Exception:
                    pass

            # 自动补上 reset_cursor：如果第一个 queue 之前没有 reset，强制插入
            first_queue_idx = next(
                (i for i, a in enumerate(actions) if a.get("type") == "queue"),
                -1
            )
            has_reset_before = any(
                a.get("type") == "reset_cursor"
                for a in actions[:first_queue_idx]
            ) if first_queue_idx >= 0 else True
            if not has_reset_before:
                actions.insert(0, {"type": "reset_cursor"})
                _log("  (auto-inserted reset_cursor before queue)")

            executed_count = 0
            for i, action in enumerate(actions):
                valid, reason, fixed = validate_action(action, game_state)
                if not valid:
                    if fixed:
                        _log(f"  Action [{i + 1}/{len(actions)}]: AUTO-FIX - {reason}")
                        action = fixed
                    else:
                        _log(f"  Action [{i + 1}/{len(actions)}]: SKIPPED - {reason}")
                        continue
                result = execute_action(action)
                executed_count += 1
                _log(f"  Action [{i + 1}/{len(actions)}]: {result}")

            # 兜底：所有动作都被跳过 → 至少wait一下，不能卡死
            if executed_count == 0:
                print(f"  FALLBACK: all actions skipped, executing wait_turn")
                execute_action({"type": "wait_turn"})

            # ── 4.5. 释放后同步冷却追踪 ──────────────────
            # 刚才如果执行了 end_turn (Space释放)，需要把排队的技能
            # 标记为冷却中，防止下回合CV误判导致AI再次尝试使用
            did_release = any(a.get("type") == "end_turn" for a in actions)
            if did_release:
                queued = get_queued_skill_indices()
                if queued:
                    cd_map = {}
                    for idx in queued:
                        if idx < len(game_state.tiles):
                            cd_map[idx] = game_state.tiles[idx].cooldown_max
                    flush_queued_skills(cd_map)
                    _log(f"  CD tracker: skills {queued} → cooldown {cd_map}")

            # ── 5. 等待 ──────────────────────────────────
            _log("  Waiting...")
            time.sleep(WAIT_AFTER_ENEMY_TURN)
            time.sleep(WAIT_BETWEEN_CYCLES)

            # ── 6. 保存本回合快照（下回合结算） ──────────────
            _pending_snapshot = {
                "actions": actions,
                "reasoning": reasoning,
                "fingerprint": fp,
            }

    except KeyboardInterrupt:
        print(f"\n\nStopped by user (turn {turn_count})")

    print(f"\nDone ({turn_count} turns)")

    # 最终手册统计
    pb = get_playbook()
    stats = pb.stats()
    print(f"\nPlaybook: {stats['total']} entries | avg score: {stats['avg_score']}")
    if stats.get("top_actions"):
        _log(f"  Top actions: {', '.join(f'{t}({c})' for t, c in stats['top_actions'])}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shogun AI - Auto play")
    parser.add_argument("--turns", type=int, default=500,
                        help="Max turns (default: 500)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save screenshots")
    parser.add_argument("--backend", choices=["ollama", "anthropic"],
                        default=AI_BACKEND,
                        help=f"AI backend (default: {AI_BACKEND})")
    parser.add_argument("--scan-skills", action="store_true",
                        help="Use vision model to scan skill bar (slow, may hallucinate)")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output (every 10 turns + scores)")
    parser.add_argument("--clear-playbook", action="store_true",
                        help="Clear all saved playbook entries")
    parser.add_argument("--playbook-stats", action="store_true",
                        help="Show playbook stats and exit")
    args = parser.parse_args()

    if args.playbook_stats:
        pb = get_playbook()
        stats = pb.stats()
        print(f"Playbook entries: {stats['total']}")
        print(f"Average score: {stats['avg_score']}")
        if stats.get("top_actions"):
            print("Top actions:")
            for t, c in stats["top_actions"]:
                print(f"  {t}: {c}")
        return

    if args.clear_playbook:
        pb = get_playbook()
        pb.clear()
        print("Playbook cleared.")

    if args.backend:
        import config
        config.AI_BACKEND = args.backend

    auto_loop(
        max_turns=args.turns,
        save_screenshots=not args.no_save,
        scan_skills=args.scan_skills,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
