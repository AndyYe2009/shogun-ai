"""
战术手册 — 让 AI 从过往成功经验中学习

工作原理：
  1. 每个回合生成"场景指纹"（网格大小、敌人数、技能状态等）
  2. 在手册中搜索最相似的过往成功案例
  3. 把相似案例作为 few-shot 示例注入到决策 prompt 中
  4. 回合结束后，高分操作自动存入手册
  5. 打越久、手册越丰富、匹配越精准
"""

import json
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# 手册存储路径
PLAYBOOK_PATH = Path(__file__).parent / "playbook.json"

# 只有得分 delta >= 此阈值才自动记录
RECORD_THRESHOLD = 10

# 检索时的最大返回数
MAX_MATCHES = 3


@dataclass
class ScenarioFingerprint:
    """场景特征指纹 — 用于相似度匹配"""
    grid_size: int = 7
    enemy_count: int = 0
    ready_skills: int = 0       # 就绪技能数
    player_hp_pct: int = 100    # 玩家血量百分比
    player_pos: int = 3         # 玩家在网格中的位置
    nearest_enemy_dist: int = 3 # 最近敌人的距离（格子数）
    player_facing_enemy: bool = True  # 玩家是否面朝敌人


@dataclass
class PlaybookEntry:
    """手册中的一条经验"""
    id: str = ""
    fingerprint: dict = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)   # 执行的动作序列
    reasoning: str = ""          # AI 当时的决策理由
    score_delta: int = 0         # 该回合的得分变化
    timestamp: str = ""          # ISO 时间戳

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlaybookEntry":
        return cls(**d)


class Playbook:
    """战术手册"""

    def __init__(self, path: Path = PLAYBOOK_PATH):
        self.path = path
        self.entries: list[PlaybookEntry] = []
        self._load()

    def _load(self):
        """从磁盘加载手册"""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("entries", [])
                self.entries = [PlaybookEntry.from_dict(e) for e in raw]
            except (json.JSONDecodeError, KeyError):
                self.entries = []
        else:
            self.entries = []

    def _save(self):
        """保存手册到磁盘"""
        data = {
            "entries": [e.to_dict() for e in self.entries],
            "total": len(self.entries),
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, fingerprint: ScenarioFingerprint,
            actions: list[dict], reasoning: str,
            score_delta: int) -> Optional[PlaybookEntry]:
        """
        添加一条经验到手册。

        只有 score_delta >= RECORD_THRESHOLD 的才记录。
        自动去重：如果已有完全相同 fingerprint + actions 的记录，只更新 score。
        """
        if score_delta < RECORD_THRESHOLD:
            return None

        fp_dict = asdict(fingerprint)

        # 去重：查找是否有相同的 fingerprint + actions
        for entry in self.entries:
            if entry.fingerprint == fp_dict and entry.actions == actions:
                # 已有相同的成功经验，保留得分更高的
                if score_delta > entry.score_delta:
                    entry.score_delta = score_delta
                    entry.timestamp = _now_iso()
                    self._save()
                return entry

        # 新条目
        entry = PlaybookEntry(
            id=str(uuid.uuid4())[:8],
            fingerprint=fp_dict,
            actions=actions,
            reasoning=reasoning,
            score_delta=score_delta,
            timestamp=_now_iso(),
        )
        self.entries.append(entry)
        self._save()
        return entry

    def search(self, fingerprint: ScenarioFingerprint,
               top_k: int = MAX_MATCHES) -> list[PlaybookEntry]:
        """
        搜索与当前场景最相似的过往成功经验。

        返回按相似度降序排列的前 top_k 条。
        """
        if not self.entries:
            return []

        scored = []
        for entry in self.entries:
            sim = _similarity(fingerprint, entry.fingerprint)
            scored.append((sim, entry))

        # 按相似度降序，同相似度按 score_delta 降序
        scored.sort(key=lambda x: (x[0], x[1].score_delta), reverse=True)

        # 只返回相似度 > 0 的
        return [entry for sim, entry in scored[:top_k] if sim > 0.3]

    def stats(self) -> dict:
        """返回手册统计信息"""
        total = len(self.entries)
        if total == 0:
            return {"total": 0, "avg_score": 0, "top_actions": []}

        avg_score = sum(e.score_delta for e in self.entries) / total

        # 最常用的动作类型
        action_counts: dict[str, int] = {}
        for entry in self.entries:
            for a in entry.actions:
                t = a.get("type", "unknown")
                action_counts[t] = action_counts.get(t, 0) + 1
        top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total": total,
            "avg_score": round(avg_score, 1),
            "top_actions": top_actions,
        }

    def clear(self):
        """清空手册"""
        self.entries = []
        self._save()


# ── 相似度计算 ────────────────────────────────────────────

def extract_fingerprint(game_state) -> ScenarioFingerprint:
    """从 GameState 提取场景指纹"""
    from state import GameState
    gs = game_state

    # 计算最近敌人的距离
    player_pos = gs.player.position
    nearest_dist = 99
    facing_enemy = False
    for e in gs.enemies:
        dist = abs(e.position - player_pos)
        if dist < nearest_dist:
            nearest_dist = dist
        # 检查是否面朝敌人
        if gs.player.facing == "right" and e.position > player_pos:
            facing_enemy = True
        elif gs.player.facing == "left" and e.position < player_pos:
            facing_enemy = True

    # 计算就绪技能数
    ready = sum(1 for t in gs.tiles if t.cooldown_remaining == 0)

    return ScenarioFingerprint(
        grid_size=gs.grid_size,
        enemy_count=len(gs.enemies),
        ready_skills=ready,
        player_hp_pct=int(gs.player.hp / max(gs.player.max_hp, 1) * 100),
        player_pos=player_pos,
        nearest_enemy_dist=min(nearest_dist, 9),  # cap at 9
        player_facing_enemy=facing_enemy,
    )


def _similarity(fp: ScenarioFingerprint, fp_dict: dict) -> float:
    """
    计算两个场景指纹的相似度（0.0 ~ 1.0）。

    加权因素：
    - 敌人数相同（权重 0.25）
    - 就绪技能数接近（权重 0.20）
    - 玩家血量接近（权重 0.15）
    - 面朝敌人状态相同（权重 0.15）
    - 网格大小相同（权重 0.10）
    - 最近敌人距离接近（权重 0.10）
    - 玩家位置接近（权重 0.05）
    """
    score = 0.0

    # 敌人数完全匹配
    if fp.enemy_count == fp_dict.get("enemy_count", -1):
        score += 0.25
    elif abs(fp.enemy_count - fp_dict.get("enemy_count", 99)) <= 1:
        score += 0.12

    # 就绪技能数接近
    d = abs(fp.ready_skills - fp_dict.get("ready_skills", -1))
    if d == 0:
        score += 0.20
    elif d == 1:
        score += 0.12
    elif d == 2:
        score += 0.06

    # 血量接近
    hp_diff = abs(fp.player_hp_pct - fp_dict.get("player_hp_pct", 0))
    if hp_diff <= 10:
        score += 0.15
    elif hp_diff <= 25:
        score += 0.08

    # 面朝敌人状态匹配
    if fp.player_facing_enemy == fp_dict.get("player_facing_enemy", False):
        score += 0.15

    # 网格大小相同
    if fp.grid_size == fp_dict.get("grid_size", -1):
        score += 0.10
    elif abs(fp.grid_size - fp_dict.get("grid_size", 99)) <= 1:
        score += 0.05

    # 最近敌人距离接近
    dist_diff = abs(fp.nearest_enemy_dist - fp_dict.get("nearest_enemy_dist", 99))
    if dist_diff == 0:
        score += 0.10
    elif dist_diff == 1:
        score += 0.05

    # 玩家位置接近
    pos_diff = abs(fp.player_pos - fp_dict.get("player_pos", -1))
    if pos_diff == 0:
        score += 0.05
    elif pos_diff <= 2:
        score += 0.02

    return score


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# ── 全局单例 ──────────────────────────────────────────────

_playbook: Optional[Playbook] = None


def get_playbook() -> Playbook:
    """获取全局战术手册"""
    global _playbook
    if _playbook is None:
        _playbook = Playbook()
    return _playbook
