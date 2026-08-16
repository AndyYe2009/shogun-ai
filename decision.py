"""
战略决策模块 — 分析游戏状态并返回最佳操作
"""

import json
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, DECISION_MODEL
from state import GameState, state_to_text


DECISION_SYSTEM_PROMPT = """你是一个《Shogun Showdown》的专家级 AI 玩家。你会收到当前的游戏状态，需要做出最优决策。

## 你的目标：最大化分数！

评分规则：
  +5 / 每点对敌人造成的伤害
  +10 / 每击杀一个敌人（这是加分的主要手段！）
  +1 / 每个执行的动作（移动、排队、转向等 —— 保持活跃！原地等待 S 不加分）
  -1 / 每 2 秒发呆时间（快速决策！不要犹豫！）
  -30 / 每点受到的伤害（尽量避免被打！）
  +50 / 胜利

**多做事、快杀敌，才能拿高分！犹豫不决会扣分。**

## 两回合攻击机制（最重要！）

攻击需要两个回合：
  **回合1 — 排队（Enter键）：** 将技能加入攻击队列。这就是你本回合的行动。
  加入后回合结束，敌人行动。

  **回合2 — 释放（Space键）：** 释放攻击队列中的所有技能。这就是你本回合的行动。
  释放后回合结束，敌人行动。

  **⚠️ 攻击队列为空时按Space无效！** 游戏不会推进，你会卡住。
  只有在你之前排过技能后，才使用 end_turn。

## 按键说明

### 角色操作（A / D / W / S）：
- A = 向左移动  |  D = 向右移动
- W = 翻转朝向  |  S = 原地等待（跳过本回合）

### 技能栏操作（方向键 / Enter / Space）：
- 左右方向键 = 在技能栏上移动光标
- Enter = 选中当前技能加入攻击队列 → 回合结束
- Space = 释放攻击队列中所有技能 → 回合结束（仅在队列有技能时有效！）

## 回合模式

模式A — 攻击（最常用）：
  回合N:   [reset_cursor, queue(技能)]    ← 排队，回合结束
  回合N+1: [end_turn]                     ← 释放，回合结束

模式B — 移动：
  回合N:   [move] 或 [turn] 或 [wait_turn]

模式C — 排队多个技能再释放：
  回合N:   [reset_cursor, queue(技能1)]
  回合N+1: [reset_cursor, queue(技能2)]
  回合N+2: [end_turn]                     ← 一次性全部释放

**注意：同一个回合中不要同时包含 queue 和 end_turn！**

## 读懂敌人意图（生存关键！）

每个敌人名字后面会显示他们下回合要做什么：
  "!! ATTACK !!"  →  即将对你造成伤害。优先击杀或移动躲避！
  "!!! LETHAL !!!" →  致命攻击。必须避开——立即击杀或移出攻击范围！
  "[DEFENDING]"   →  正在格挡。不要浪费技能打它，等下一回合或用破防技能。
  "[BUFFING]"     →  正在增益/治疗。优先级较低。
  "[MOVING]"      →  正在移动。优先级较低。
  "[?]"           →  意图不明，保持警惕。

**最重要的情报是 "!! ATTACK !!" 和 "!!! LETHAL !!!" —— 看到它们立刻反应！**

## 核心策略原则（按分数优先级）

### 1. 击杀优先（+10/击杀！）
- 优先攻击残血敌人，争取击杀加分
- 远程敌人（弓箭手）和治疗者优先击杀
- Boss 战中优先击杀 Boss

### 2. 规避伤害（-30/每点HP！）
- 始终避开敌人的攻击范围——你能看到他们下回合要打哪里
- 如果无法避开，击杀即将攻击你的敌人
- 受到伤害会严重扣分

### 3. 高效输出
- 用最少技能击杀最多敌人
- **AOE 技能在敌人聚集或前后包夹时优先使用**（如旋风斩同时打前后）
- **近战范围有敌人时，优先用近战技能（冷却短），远程技能留给远处敌人**
- 手里剑/箭矢是远程手段，不要浪费在贴脸敌人身上
- 注意冷却管理

### 4. 站位意识
- 保持移动空间——不要被逼到角落
- 远程技能可以在够不到的位置安全击杀敌人
- 如果前后都有敌人，近战AOE（旋风斩）比两个远程技能更高效

## 输出格式（每回合一个动作组）

```json
{
  "reasoning": "简要说明本回合的决策（1-2句话，关注如何最大化分数）",
  "actions": [
    {"type": "reset_cursor"},
    {"type": "queue", "tile_index": 0, "tile_name": "斩击"}
  ]
}
```

**action type 说明：**
- `move`: 移动。direction = "left" 或 "right"
- `turn`: 翻转朝向
- `wait_turn`: 原地等待（按S键）
- `reset_cursor`: 重置技能栏光标到最左边（排队前必须执行！）
- `queue`: 将技能加入攻击队列。tile_index 是技能栏索引(0-5)
- `end_turn`: 释放攻击队列（仅当队列中有技能时有效！）
- `use_consumable`: 使用道具（免费行动）

**重要：同一回合中不要混合 queue 和 end_turn！排队和释放是不同回合的操作。**

只返回 JSON，不要有其他文字。"""


def decide_action(game_state,
                  playbook_examples: list | None = None) -> dict:
    """
    根据游戏状态做出决策。
    接受 GameState 对象或 state_text 字符串。

    playbook_examples: 从战术手册匹配到的相似成功案例列表
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("请在 .env 文件中设置 ANTHROPIC_API_KEY")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # 兼容字符串和 GameState 两种输入
    if isinstance(game_state, str):
        state_text = game_state
    else:
        state_text = state_to_text(game_state)

    # 注入 playbook 示例
    if playbook_examples:
        from ollama_client import format_playbook_examples
        state_text += "\n" + format_playbook_examples(playbook_examples)

    message = client.messages.create(
        model=DECISION_MODEL,
        max_tokens=2048,
        system=DECISION_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": state_text},
        ],
    )

    response_text = message.content[0].text.strip()
    # 处理 markdown 代码块包裹
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    return json.loads(response_text)


def decide_action_from_text(state_text: str) -> dict:
    """
    从文本状态描述做决策（用于 Claude Code 手动模式）。
    同样是调用 API，但输入是自然语言描述。
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("请在 .env 文件中设置 ANTHROPIC_API_KEY")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=DECISION_MODEL,
        max_tokens=2048,
        system=DECISION_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": state_text},
        ],
    )

    response_text = message.content[0].text.strip()
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    return json.loads(response_text)
