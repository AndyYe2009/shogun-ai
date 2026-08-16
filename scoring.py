"""
评分系统 v3:
  - 造成伤害:    +5 分/每点伤害
  - 命中敌人:    +10 分（造成任意伤害）
  - 躲避攻击:    +15 分（本回合未受伤）
  - 击杀敌人:    +10 分/个
  - 执行动作:    +1 分/个 (原地等待 S 不计算)
  - 空闲惩罚:    -1 分/每 2 秒
  - 受到伤害:    -30 分/点 (受伤害惩罚 — 最高优先级避免)
  - 技能空放:    -10 分/次 (释放技能但未命中任何敌人)
  - 胜利:        +50 分
"""

import time
from dataclasses import dataclass, field


@dataclass
class ScoreTracker:
    """追踪评分状态"""
    total: int = 0                # 累计总分
    kills: int = 0                # 总击杀数
    damage_dealt: int = 0         # 累计造成伤害
    damage_taken: int = 0         # 累计受到伤害
    action_count: int = 0         # 总执行动作数
    action_bonus: int = 0         # 动作奖励累计
    idle_penalty: int = 0         # 空闲惩罚累计
    turn_deltas: list[int] = field(default_factory=list)  # 每回合的分数变化

    # 上一回合的快照
    _prev_player_hp: int = 0
    _prev_enemy_hps: dict = field(default_factory=dict)  # {enemy_id: hp}

    # 空闲追踪
    _idle_start: float = 0.0     # 本回合计时起点
    _actions_this_turn: int = 0  # 本回合执行的动作数

    IDLE_INTERVAL: float = 2.0   # 每 N 秒无动作扣分

    def _ensure_idle_timer(self) -> None:
        """确保空闲计时器已启动"""
        if self._idle_start == 0.0:
            self._idle_start = time.time()

    def snapshot(self, player_hp: int, enemies: list[dict]) -> None:
        """
        回合开始时调用。保存快照以便回合结束后计算变化量。

        enemies: [{"id": unique_id, "hp": int}, ...]
        """
        self._prev_player_hp = player_hp
        self._prev_enemy_hps = {}
        for e in enemies:
            self._prev_enemy_hps[e["id"]] = e["hp"]

        # 开始新一轮的空闲计时
        self._idle_start = time.time()
        self._actions_this_turn = 0

    def record_action(self) -> int:
        """
        记录一个动作被执行。返回奖励分数（+1）。
        在 executor 执行每个动作后调用。

        同时重置空闲计时器——做出动作就证明没有在发呆。
        """
        self._ensure_idle_timer()
        self._actions_this_turn += 1
        self.action_count += 1
        reward = 1
        self.action_bonus += reward
        # 重置空闲计时器：刚做了动作，从此刻重新计时
        self._idle_start = time.time()
        return reward

    def compute_idle_penalty(self) -> int:
        """
        计算空闲惩罚：从上次记录动作（或回合开始）到现在，
        每超过 IDLE_INTERVAL 秒扣 1 分。

        Returns: 扣分（负数或0）
        """
        now = time.time()
        elapsed = now - self._idle_start
        if elapsed < self.IDLE_INTERVAL:
            return 0

        # 计算有多少个完整的 IDLE_INTERVAL
        intervals = int(elapsed / self.IDLE_INTERVAL)
        penalty = -intervals

        # 重置计时起点（只让已过去的 interval 扣分，下次从新的起点算）
        self._idle_start += intervals * self.IDLE_INTERVAL

        self.idle_penalty += penalty
        return penalty

    def compute_delta(self, player_hp: int, enemies: list[dict],
                      victory: bool = False, game_over: bool = False,
                      skills_released: bool = False) -> dict:
        """
        回合结束后调用。计算本回合的分数变化。

        skills_released: 上回合是否释放了技能（end_turn），用于检测空放

        Returns: {delta, kills, damage_dealt, damage_taken, total, ...}
        """
        delta = 0
        kills_this_turn = 0
        dealt_this_turn = 0
        taken_this_turn = 0
        miss_penalty = 0

        # 动作奖励（本回合执行的动作数 × +1）
        action_reward = self._actions_this_turn
        delta += action_reward

        # 空闲惩罚（回合结束时的剩余空闲时间）
        idle_penalty = self.compute_idle_penalty()
        delta += idle_penalty

        # 伤害计算
        hit_enemy = False
        for e in enemies:
            eid = e["id"]
            prev_hp = self._prev_enemy_hps.get(eid, e["hp"])
            hp_lost = prev_hp - e["hp"]
            if hp_lost > 0:
                dealt_this_turn += hp_lost
                hit_enemy = True
                delta += hp_lost * 5   # +5 per damage dealt
            # 敌人 HP 降到 0 以下 → 击杀
            if prev_hp > 0 and e["hp"] <= 0:
                kills_this_turn += 1
                delta += 10      # +10 per kill

        if hit_enemy:
            delta += 10  # 命中敌人+10
        elif skills_released and dealt_this_turn == 0:
            # 释放了技能但未命中任何敌人 → 空放惩罚
            miss_penalty = -10
            delta += miss_penalty

        # 受伤计算
        hp_lost = self._prev_player_hp - player_hp
        if hp_lost > 0:
            taken_this_turn = hp_lost
            delta -= hp_lost * 30  # -30 per damage taken
        else:
            # 本回合未受伤：躲避成功+15
            delta += 15

        # 游戏结束奖惩
        if victory:
            delta += 50
        elif game_over:
            delta -= 30

        # 重置本回合计数器
        self._actions_this_turn = 0

        # 更新累计
        self.total += delta
        self.kills += kills_this_turn
        self.damage_dealt += dealt_this_turn
        self.damage_taken += taken_this_turn
        self.turn_deltas.append(delta)

        return {
            "delta": delta,
            "total": self.total,
            "kills": self.kills,
            "kills_this_turn": kills_this_turn,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "dealt_this_turn": dealt_this_turn,
            "taken_this_turn": taken_this_turn,
            "action_reward": action_reward,
            "idle_penalty": idle_penalty,
            "miss_penalty": miss_penalty,
            "actions_this_turn": self._actions_this_turn,
        }

    def reset(self):
        """重置所有数据（新一局开始时）"""
        self.total = 0
        self.kills = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        self.action_count = 0
        self.action_bonus = 0
        self.idle_penalty = 0
        self.turn_deltas.clear()
        self._prev_player_hp = 0
        self._prev_enemy_hps.clear()
        self._idle_start = 0.0
        self._actions_this_turn = 0


# 全局单例
_scorer = ScoreTracker()


def get_scorer() -> ScoreTracker:
    """获取全局评分追踪器"""
    return _scorer


def reset_scorer() -> None:
    """重置评分器"""
    _scorer.reset()


def record_action() -> int:
    """记录一个动作被执行（供 executor 调用）"""
    return _scorer.record_action()


def compute_idle_penalty() -> int:
    """计算空闲惩罚（供 main.py 在大等待前调用）"""
    return _scorer.compute_idle_penalty()
