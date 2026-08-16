"""
动作执行模块 — 模拟键盘/鼠标操作

按键映射（重要！两回合机制）：
  A/D     = 左右移动角色
  W       = 翻转朝向
  S       = 原地等待，什么都不做
  ←/→     = 技能栏光标左右移动
  Enter   = 将当前技能加入攻击队列（消耗这一回合）
  Space   = 释放攻击队列中所有技能（消耗下一回合）
            **重要：攻击队列为空时 Space 无效，不会推进回合！**

回合机制：
  - 排队回合：用 Enter 将技能加入队列 → 回合结束
  - 释放回合：用 Space 释放队列 → 回合结束
  - 排队和释放必须在不同回合

后台模式：
  使用 Windows PostMessage API 向游戏窗口直接发送按键消息，
  不需要窗口在前台，不影响用户做其他事情。
"""

import time
import ctypes
import pydirectinput
import pyautogui
from config import (
    KEY_MOVE_LEFT, KEY_MOVE_RIGHT,
    KEY_TURN, KEY_WAIT,
    KEY_SKILL_LEFT, KEY_SKILL_RIGHT, KEY_SKILL_SELECT,
    KEY_END_TURN, KEY_CONFIRM, KEY_CANCEL,
    WAIT_AFTER_ACTION,
    BACKGROUND_MODE,
)
from scoring import record_action as _score_record_action

# 禁用 pyautogui 的安全检查（加速操作）
pyautogui.FAILSAFE = False
pydirectinput.FAILSAFE = False

# ── Windows API 后台按键 ─────────────────────────────────

# 虚拟键码映射
_VK_MAP = {
    "a": 0x41, "d": 0x44, "w": 0x57, "s": 0x53,
    "q": 0x51, "e": 0x45,
    "left": 0x25, "right": 0x27, "up": 0x26, "down": 0x28,
    "enter": 0x0D, "space": 0x20, "esc": 0x1B,
}

_user32 = ctypes.windll.user32
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101

# MapVirtualKey 用于获取扫描码
_MAPVK_VK_TO_VSC = 0

# 缓存的游戏窗口句柄
_game_hwnd = None


def _get_game_hwnd() -> int:
    """获取游戏窗口句柄（缓存）"""
    global _game_hwnd
    if _game_hwnd is None or not _user32.IsWindow(_game_hwnd):
        from capture import get_game_window
        win = get_game_window()
        _game_hwnd = win._hWnd
    return _game_hwnd


def _post_key(hwnd: int, vk: int, key_down: bool) -> None:
    """
    向指定窗口发送按键消息（后台，不抢焦点）。

    构造正确的 lParam 包含扫描码，这是 Unity 游戏识别输入的关键。
    """
    scan_code = _user32.MapVirtualKeyW(vk, _MAPVK_VK_TO_VSC)

    if key_down:
        msg = _WM_KEYDOWN
        lparam = 1 | (scan_code << 16)  # repeat=1, scan_code
    else:
        msg = _WM_KEYUP
        lparam = 1 | (scan_code << 16) | (1 << 30) | (1 << 31)
        # repeat=1, scan_code, prev_state=1, transition=1

    _user32.PostMessageW(hwnd, msg, vk, lparam)


def _press_key_background(key: str, hold: float = 0.04) -> None:
    """后台按键：通过 Windows PostMessage 发送，不抢焦点"""
    hwnd = _get_game_hwnd()
    vk = _VK_MAP.get(key.lower())
    if vk is None:
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        return

    _post_key(hwnd, vk, key_down=True)
    time.sleep(hold)
    _post_key(hwnd, vk, key_down=False)
    time.sleep(WAIT_AFTER_ACTION)

# 追踪技能栏光标位置（假设起始在位置0）
_current_skill_cursor = 0

# 追踪角色朝向（初始朝右，游戏开始时默认朝向敌人）
_tracked_facing = "right"

# 追踪玩家网格位置（游戏开始时，Wanderer 在7格网格的中间位置3，面朝右）
_tracked_player_pos = 3

# 追踪攻击队列中已加入的技能数量
# 用于判断 Space 是否有东西可释放
_attack_queue_count = 0

# ── 冷却追踪（确定性兜底） ─────────────────────────────
# CV 像素分析可能把冷却中的灰色图标误判为就绪。
# 这里用和 _tracked_facing 相同的确定性思路：
# 记录每个技能槽的冷却剩余回合。当 CV 和 tracker 不一致时，
# tracker 作为安全底线（只能增加冷却，不能减少）。
# 格式: [slot_0_cd, slot_1_cd, ..., slot_5_cd]
_skill_cooldown_tracker = [0, 0, 0, 0, 0, 0]

# 本回合排队的技能索引列表，释放时用于设置冷却
_queued_skill_indices: list[int] = []

# Wanderer 换位被动冷却追踪
# 面朝相邻敌人 + 背后有空位 → 按移动键可换位，冷却 4 回合
_swap_cooldown = 0
SWAP_COOLDOWN_MAX = 4


def press_key(key: str, hold: float = 0.04):
    """按下并释放一个键（后台模式使用 PostMessage + 正确扫描码）"""
    if BACKGROUND_MODE:
        _press_key_background(key, hold)
    else:
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)
        time.sleep(WAIT_AFTER_ACTION)


def move_left():
    """向左移动一格"""
    global _tracked_player_pos
    press_key(KEY_MOVE_LEFT)
    _tracked_player_pos = max(0, _tracked_player_pos - 1)


def move_right():
    """向右移动一格"""
    global _tracked_player_pos
    press_key(KEY_MOVE_RIGHT)
    # 游戏最大网格 9 格 (0-8)，不允许超过
    _tracked_player_pos = min(_tracked_player_pos + 1, 8)


def turn_around():
    """转向/翻转角色朝向 (W键)"""
    global _tracked_facing
    press_key(KEY_TURN)
    _tracked_facing = "left" if _tracked_facing == "right" else "right"


def hold_key(key: str, duration: float) -> str:
    """
    长按某个键指定秒数（用于重启游戏等操作）。
    支持后台模式。
    """
    if BACKGROUND_MODE:
        hwnd = _get_game_hwnd()
        vk = _VK_MAP.get(key.lower())
        if vk is not None:
            _post_key(hwnd, vk, key_down=True)
            time.sleep(duration)
            _post_key(hwnd, vk, key_down=False)
            time.sleep(WAIT_AFTER_ACTION)
            return f"Held {key} for {duration}s (background)"
    # 前台模式回退
    pydirectinput.keyDown(key)
    time.sleep(duration)
    pydirectinput.keyUp(key)
    time.sleep(WAIT_AFTER_ACTION)
    return f"Held {key} for {duration}s"


def wait_turn():
    """原地等待一回合 (S键)"""
    press_key(KEY_WAIT)


def _tap_key(key: str, hold: float = 0.02) -> None:
    """快速按一下键（支持后台模式）"""
    if BACKGROUND_MODE:
        _press_key_background(key, hold)
    else:
        pydirectinput.keyDown(key)
        time.sleep(hold)
        pydirectinput.keyUp(key)


def queue_tile(tile_index: int):
    """
    将技能牌加入攻击队列。（消耗这一回合）
    用左右方向键在技能栏中导航，然后按回车选中。

    tile_index: 0-5，对应底部技能栏从左到右的 6 个位置。
    """
    global _current_skill_cursor, _attack_queue_count, _queued_skill_indices

    # 限制在 0-5 范围内
    tile_index = max(0, min(5, tile_index))

    # 计算需要移动的步数
    delta = tile_index - _current_skill_cursor

    # 移动光标到目标位置（快速导航）
    if delta < 0:
        for _ in range(abs(delta)):
            _tap_key(KEY_SKILL_LEFT, hold=0.02)
    elif delta > 0:
        for _ in range(delta):
            _tap_key(KEY_SKILL_RIGHT, hold=0.02)

    # 按回车选中技能（加入攻击队列，消耗一回合）
    press_key(KEY_SKILL_SELECT)

    _current_skill_cursor = tile_index
    _attack_queue_count += 1  # 追踪队列（CD技能被游戏忽略也不影响——Space无害）
    _queued_skill_indices.append(tile_index)  # 记录排队了哪个技能


def reset_skill_cursor():
    """
    重置技能栏光标到最左边。
    在不知道当前光标位置时使用。
    """
    global _current_skill_cursor
    # 快速连续按左键——减少单次等待
    for _ in range(8):
        _tap_key(KEY_SKILL_LEFT, hold=0.015)
    _current_skill_cursor = 0


def release_queue():
    """释放攻击队列中所有技能 (Space)（消耗这一回合）"""
    global _current_skill_cursor, _attack_queue_count, _queued_skill_indices, _skill_cooldown_tracker
    press_key(KEY_END_TURN)
    _current_skill_cursor = 0  # 游戏重置了光标，同步追踪状态
    _attack_queue_count = 0    # 队列已释放
    # 不清空 _queued_skill_indices —— main.py 需要读取它来设置冷却
    # main.py 调用 flush_queued_skills() 后再清空


def end_turn():
    """
    【已废弃】请使用 release_queue()
    保留此函数以保证向后兼容。
    """
    release_queue()


def get_attack_queue_count() -> int:
    """获取当前攻击队列中的技能数量"""
    return _attack_queue_count


def get_skill_cooldown_tracker() -> list[int]:
    """获取冷却追踪器当前状态 [slot_0, ..., slot_5]"""
    return list(_skill_cooldown_tracker)


def get_swap_cooldown() -> int:
    """获取换位技能的剩余冷却回合数"""
    return _swap_cooldown


def record_swap() -> None:
    """记录换位使用，设置冷却 4 回合"""
    global _swap_cooldown
    _swap_cooldown = SWAP_COOLDOWN_MAX


def tick_cooldowns() -> None:
    """每回合调用：所有冷却计数器 -1"""
    global _skill_cooldown_tracker, _swap_cooldown
    for i in range(6):
        if _skill_cooldown_tracker[i] > 0:
            _skill_cooldown_tracker[i] -= 1
    if _swap_cooldown > 0:
        _swap_cooldown -= 1


def flush_queued_skills(cooldown_map: dict[int, int]) -> None:
    """
    释放后调用：根据排队的技能设置冷却追踪。
    cooldown_map: {tile_index: cooldown_max} — main.py 从 game_state 提供
    """
    global _queued_skill_indices, _skill_cooldown_tracker
    for idx in _queued_skill_indices:
        cd_max = cooldown_map.get(idx, 2)  # 默认冷却2回合
        _skill_cooldown_tracker[idx] = cd_max
    _queued_skill_indices.clear()


def get_queued_skill_indices() -> list[int]:
    """获取本回合排队的技能索引列表（供 main.py 读取）"""
    return list(_queued_skill_indices)


def clear_queued_skills() -> None:
    """清空排队索引（每回合开始时调用，防止残留）"""
    global _queued_skill_indices
    _queued_skill_indices.clear()


def confirm():
    """确认操作"""
    press_key(KEY_CONFIRM)


def cancel():
    """取消操作"""
    press_key(KEY_CANCEL)


# ── 朝向追踪 ─────────────────────────────────────────────

def get_tracked_facing() -> str:
    """获取当前追踪的角色朝向"""
    return _tracked_facing


def get_tracked_player_pos() -> int:
    """获取当前追踪的玩家网格位置"""
    return _tracked_player_pos


def reset_facing():
    """重置朝向和位置追踪（游戏开始时调用）"""
    global _tracked_facing, _tracked_player_pos, _attack_queue_count, _skill_cooldown_tracker, _queued_skill_indices, _swap_cooldown
    _tracked_facing = "right"
    _tracked_player_pos = 3
    _attack_queue_count = 0
    _skill_cooldown_tracker = [0, 0, 0, 0, 0, 0]
    _queued_skill_indices.clear()
    _swap_cooldown = 0


# ── 高级动作 ─────────────────────────────────────────────

def validate_action(action: dict, game_state=None) -> tuple[bool, str, dict | None]:
    """
    验证动作是否有效。返回 (is_valid, reason, fixed_action)。

    如果动作无效但有可用的替代方案，fixed_action 包含修正后的动作。
    """
    action_type = action.get("type", "")

    if action_type == "move":
        direction = action.get("direction", "right")
        is_swap = action.get("swap", False)

        if is_swap:
            # ── 换位验证 ────────────────────────────────
            if _swap_cooldown > 0:
                return False, f"Swap on cooldown ({_swap_cooldown} turns remaining)", None
            if game_state:
                # 检查相邻敌人
                p = game_state.player
                target_pos = p.position + (1 if direction == "right" else -1)
                adjacent_enemy = None
                for e in game_state.enemies:
                    if e.position == target_pos and e.hp > 0:
                        adjacent_enemy = e
                        break
                if not adjacent_enemy:
                    return False, f"No adjacent enemy at pos {target_pos} to swap with", None
                # 检查背后是否有空位
                behind_pos = p.position + (-1 if direction == "right" else 1)
                if behind_pos < 0 or behind_pos >= game_state.grid_size:
                    return False, f"No space behind (pos {behind_pos} out of bounds)", None
                for e in game_state.enemies:
                    if e.position == behind_pos and e.hp > 0:
                        return False, f"No space behind (pos {behind_pos} occupied by {e.name})", None

        if game_state:
            pos = game_state.player.position
            grid_size = game_state.grid_size
            if direction == "left" and pos <= 0:
                if is_swap and pos <= 0:
                    return False, f"Cannot swap left: already at position 0", None
                if pos < grid_size - 1:
                    fixed = dict(action, direction="right")
                    return False, f"Cannot move left at pos 0, auto-fixed to move right", fixed
                return False, f"Cannot move left: already at position 0", None
            if direction == "right" and pos >= grid_size - 1:
                if is_swap and pos >= grid_size - 1:
                    return False, f"Cannot swap right: already at position {grid_size - 1}", None
                if pos > 0:
                    fixed = dict(action, direction="left")
                    return False, f"Cannot move right at max pos, auto-fixed to move left", fixed
                return False, f"Cannot move right: already at position {grid_size - 1}", None

    elif action_type == "queue":
        tile_index = action.get("tile_index", 0)
        if game_state and game_state.tiles:
            if tile_index < 0 or tile_index >= len(game_state.tiles):
                return False, f"Invalid tile index {tile_index}", None
            tile = game_state.tiles[tile_index]
            if tile.cooldown_remaining > 0:
                # 自动找第一个可用技能
                for i, t in enumerate(game_state.tiles):
                    if t.cooldown_remaining == 0:
                        fixed = dict(action, tile_index=i, tile_name=t.name)
                        return False, f"Skill [{tile.name}] on CD, auto-switched to [{t.name}]", fixed
                return False, f"Skill [{tile.name}] on cooldown and no ready skills available", None

    elif action_type == "end_turn":
        # 检查攻击队列是否有技能可释放
        if _attack_queue_count == 0:
            # 队列为空，Space 无效，自动换成 wait
            fixed = {"type": "wait_turn"}
            return False, "Attack queue is empty, Space has no effect — auto-switched to wait (S)", fixed

    elif action_type in ("use_consumable", "use_consumable_1", "use_consumable_2"):
        # Q/E键没用——游戏里没见过道具
        return False, "Consumable not available", None

    return True, "OK", None


def execute_action(action: dict) -> str:
    """
    执行一个动作。action 格式:
    {
        "type": "move" | "turn" | "queue" | "end_turn" | "reset_cursor" | "wait" | "wait_turn",
        ...
    }
    返回执行描述。
    """
    action_type = action.get("type", "")

    if action_type == "move":
        direction = action.get("direction", "right")
        is_swap = action.get("swap", False)
        if direction == "left":
            move_left()
            if is_swap:
                record_swap()
            result = "Move left (A)" if not is_swap else "SWAP left (A) — Wanderer passive"
        else:
            move_right()
            if is_swap:
                record_swap()
            result = "Move right (D)" if not is_swap else "SWAP right (D) — Wanderer passive"

    elif action_type == "turn":
        turn_around()
        result = "Turn around (W)"

    elif action_type == "wait_turn":
        wait_turn()
        result = "Wait one turn (S)"

    elif action_type == "queue":
        tile_index = action.get("tile_index", 0)
        queue_tile(tile_index)
        tile_name = action.get("tile_name", f"Skill {tile_index + 1}")
        result = f"Queue [{tile_name}] into attack queue (Enter) [queue count: {_attack_queue_count}]"

    elif action_type == "reset_cursor":
        reset_skill_cursor()
        result = "Reset skill cursor to leftmost"

    elif action_type == "end_turn":
        count_before = _attack_queue_count
        release_queue()
        result = f"Release queue ({count_before} skills) (Space)"

    elif action_type == "wait":
        duration = action.get("duration", 0.5)
        time.sleep(duration)
        result = f"Wait {duration}s"

    else:
        result = f"Unknown action: {action_type}"

    # 记录动作得分（每个动作 +1，鼓励多做事少发呆）
    # 原地等待不算动作，不加分也不减分
    if action_type != "wait_turn":
        reward = _score_record_action()
        if reward:
            return f"{result} [+{reward} action]"
    return result
