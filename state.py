"""
游戏状态数据结构
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class PlayerState:
    """玩家状态"""
    character: str = "Wanderer"       # 角色名
    hp: int = 60                      # 当前血量
    max_hp: int = 60                  # 最大血量
    shield: int = 0                   # 护盾值
    position: int = 3                 # 在网格上的位置 (0-based, 从左到右)
    facing: str = "right"             # 朝向: "left" | "right"
    buffs: list[str] = field(default_factory=list)    # 当前 buff
    debuffs: list[str] = field(default_factory=list)  # 当前 debuff


@dataclass
class EnemyState:
    """敌人状态"""
    name: str = ""                    # 敌人名称
    hp: int = 0                       # 当前血量
    max_hp: int = 0                   # 最大血量
    shield: int = 0                   # 护盾
    position: int = 0                 # 在网格上的位置
    next_action: str = ""             # 下回合意图（攻击/防御/移动/召唤 等）
    next_action_detail: str = ""      # 意图详情（如 "8 伤害", "格挡"）
    next_action_target: int = -1      # 意图目标位置 (-1 表示不适用)
    buffs: list[str] = field(default_factory=list)
    debuffs: list[str] = field(default_factory=list)
    is_elite: bool = False            # 是否为精英怪
    is_boss: bool = False             # 是否为 Boss


@dataclass
class TileState:
    """技能牌状态"""
    name: str = ""                    # 技能名
    description: str = ""             # 效果描述
    damage: int = 0                   # 伤害值
    cooldown_remaining: int = 0       # 剩余冷却回合
    cooldown_max: int = 0             # 最大冷却
    aoe: bool = False                 # 是否范围攻击
    range_min: int = 1                # 最小射程
    range_max: int = 1                # 最大射程
    effects: list[str] = field(default_factory=list)  # 附加效果（击退/冰冻/诅咒等）
    is_upgraded: bool = False         # 是否已升级


@dataclass
class ConsumableState:
    """道具状态"""
    name: str = ""
    effect: str = ""
    count: int = 0


@dataclass
class AttackQueueSlot:
    """攻击队列中的技能"""
    tile: Optional[TileState] = None
    target_position: int = -1         # 目标位置


@dataclass
class GameState:
    """完整游戏状态"""
    turn_number: int = 0
    grid_size: int = 7                # 当前网格大小 (5-9)
    grid_positions: list[int] = field(default_factory=list)  # 可用位置列表

    player: PlayerState = field(default_factory=PlayerState)
    enemies: list[EnemyState] = field(default_factory=list)

    tiles: list[TileState] = field(default_factory=list)       # 底部技能栏
    attack_queue: list[AttackQueueSlot] = field(default_factory=list)  # 当前攻击队列 (0-3 slots)

    consumables: list[ConsumableState] = field(default_factory=list)

    # 游戏进程信息
    stage: str = ""                   # 如 "Day 3"
    room: str = ""                    # 如 "Combat 2/5" 或 "Shop"
    gold: int = 0

    # 特殊状态
    game_over: bool = False
    victory: bool = False

    # 可用操作列表（由解析器自动生成）
    available_actions: list[str] = field(default_factory=list)


def state_to_text(state: GameState, score_info: dict | None = None) -> str:
    """
    将游戏状态转换为结构化的文本描述，
    适合发送给 Ollama / Claude 做决策。

    score_info: 评分信息 dict (可选)
        {total, kills, damage_dealt, damage_taken, last_delta, last_kills, last_dealt, last_taken}
    """
    lines = []

    lines.append("=" * 50)
    lines.append(f"=== Turn {state.turn_number} | {state.stage} {state.room} ===")
    lines.append(f"Grid Size: {state.grid_size} | Gold: {state.gold}")

    # ── 评分 ──────────────────────────────────────────
    if score_info:
        total = score_info.get("total", 0)
        kills = score_info.get("kills", 0)
        dealt = score_info.get("damage_dealt", 0)
        taken = score_info.get("damage_taken", 0)
        action_count = score_info.get("action_count", 0)
        action_bonus = score_info.get("action_bonus", 0)
        idle_penalty = score_info.get("idle_penalty", 0)
        last_delta = score_info.get("last_delta", 0)
        last_kills = score_info.get("last_kills", 0)
        last_dealt = score_info.get("last_dealt", 0)
        last_taken = score_info.get("last_taken", 0)

        # 上回合评分变化
        delta_sign = "+" if last_delta > 0 else ""
        lines.append(f"Score: {total} | Last turn: {delta_sign}{last_delta}")
        if last_kills:
            lines.append(f"  Last turn: {last_kills} kill(s), {last_dealt} dmg dealt, {last_taken} dmg taken")
        lines.append(f"  Total: {kills} kills | {dealt} dmg dealt | {taken} dmg taken")
        lines.append(f"  Actions: {action_count} (+{action_bonus}) | Idle penalty: {idle_penalty}")

        # 评分规则提醒
        lines.append(f"  [Scoring: +1/action, +5/dmg, +10/kill, -1/2s idle, -30/dmg taken, +50/victory]")

    lines.append("=" * 50)

    # Grid visualization
    grid = ["."] * state.grid_size
    grid[state.player.position] = "P" if state.player.facing == "right" else "p"
    for e in state.enemies:
        if 0 <= e.position < state.grid_size:
            grid[e.position] = "E" if e.hp > 0 else "X"
    lines.append(f"\n[Grid]: [{' | '.join(grid)}]")
    lines.append(f"   P=you(->) p=you(<-)  E=enemy X=dead")

    # Player
    p = state.player
    can_move_left = "YES" if p.position > 0 else "NO (at left edge)"
    can_move_right = "YES" if p.position < state.grid_size - 1 else "NO (at right edge)"

    lines.append(f"\n[Player] ({p.character})")
    lines.append(f"   HP: {p.hp}/{p.max_hp}  | Shield: {p.shield}")
    lines.append(f"   Pos: {p.position}/{state.grid_size - 1}  | Facing: {p.facing}")
    lines.append(f"   Can move left: {can_move_left}  |  Can move right: {can_move_right}")
    if p.buffs:
        lines.append(f"   Buff: {', '.join(p.buffs)}")
    if p.debuffs:
        lines.append(f"   Debuff: {', '.join(p.debuffs)}")

    # Skills
    ready_skills = [t for t in state.tiles if t.cooldown_remaining == 0]
    cd_skills = [t for t in state.tiles if t.cooldown_remaining > 0]
    lines.append(f"\n[Skills] ({len(ready_skills)} READY, {len(cd_skills)} on cooldown):")
    for i, tile in enumerate(state.tiles):
        if tile.cooldown_remaining == 0:
            cd_str = "READY"
            aoe_str = " [AOE]" if tile.aoe else ""
            range_str = f"Range:{tile.range_min}-{tile.range_max}"
            fx_str = f" FX:{'/'.join(tile.effects)}" if tile.effects else ""
            lines.append(f"   [{i}] **{tile.name}** | Dmg:{tile.damage} | {range_str}{aoe_str}{fx_str} | USE THIS!")
        else:
            cd_str = f"CD:{tile.cooldown_remaining}/{tile.cooldown_max}"
            lines.append(f"   [{i}] {tile.name} | {cd_str} | CANNOT USE")

    # 攻击队列
    lines.append(f"\n[Atk Queue] ({len(state.attack_queue)}/3):")
    for i, slot in enumerate(state.attack_queue):
        if slot.tile:
            tgt = f" → 位置{slot.target_position}" if slot.target_position >= 0 else ""
            lines.append(f"   [{i + 1}] {slot.tile.name}{tgt}")
        else:
            lines.append(f"   [{i + 1}] (空)")

    # 道具
    if state.consumables:
        lines.append(f"\n[Items]:")
        for i, c in enumerate(state.consumables):
            lines.append(f"   [{i + 1}] {c.name} x{c.count}: {c.effect}")

    # 敌人
    lines.append(f"\n[Enemies] ({len(state.enemies)}):")
    for i, e in enumerate(state.enemies):
        elite_str = "[Elite] " if e.is_elite else ""
        boss_str = "[BOSS] " if e.is_boss else ""
        shield_str = f"Shield:{e.shield} " if e.shield > 0 else ""
        tgt_str = f" Target:{e.next_action_target}" if e.next_action_target >= 0 else ""

        # 意图高亮标记
        if e.next_action == "attack":
            intent_marker = "!! ATTACK !!"
        elif e.next_action == "defend":
            intent_marker = "[DEFENDING]"
        elif e.next_action == "buff":
            intent_marker = "[BUFFING]"
        elif e.next_action == "move":
            intent_marker = "[MOVING]"
        else:
            intent_marker = "[?]"

        lines.append(f"   [{i + 1}] {elite_str}{boss_str}{e.name}  {intent_marker}")
        lines.append(f"       HP:{e.hp}/{e.max_hp} {shield_str}| Pos:{e.position}")
        lines.append(f"       Next turn: {e.next_action_detail}{tgt_str}")
        if e.buffs:
            lines.append(f"       Buff: {', '.join(e.buffs)}")

    # 可用操作
    if state.available_actions:
        lines.append(f"\n[Available]:")
        for a in state.available_actions:
            lines.append(f"   * {a}")

    if state.game_over:
        lines.append(f"\n{'[VICTORY]' if state.victory else '[GAME OVER]'}")

    lines.append("\n" + "=" * 50)

    return "\n".join(lines)


def state_from_dict(data: dict) -> GameState:
    """从字典构建 GameState（Claude Vision API 返回 JSON 后使用）"""
    player = PlayerState(**data.get("player", {}))

    enemies = [EnemyState(**e) for e in data.get("enemies", [])]
    tiles = [TileState(**t) for t in data.get("tiles", [])]

    queue = []
    for slot in data.get("attack_queue", []):
        tile = TileState(**slot["tile"]) if slot.get("tile") else None
        queue.append(AttackQueueSlot(tile=tile, target_position=slot.get("target_position", -1)))

    consumables = [ConsumableState(**c) for c in data.get("consumables", [])]

    return GameState(
        turn_number=data.get("turn_number", 0),
        grid_size=data.get("grid_size", 7),
        player=player,
        enemies=enemies,
        tiles=tiles,
        attack_queue=queue,
        consumables=consumables,
        stage=data.get("stage", ""),
        room=data.get("room", ""),
        gold=data.get("gold", 0),
        game_over=data.get("game_over", False),
        victory=data.get("victory", False),
    )
