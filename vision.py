"""
画面解析模块 — 使用 Claude Vision API 将截图转为结构化游戏状态
"""

import base64
import json
from io import BytesIO
from PIL import Image
from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, VISION_MODEL
from state import GameState, state_from_dict, TileState

# 解析 Prompt：告诉 Claude 如何从截图中提取游戏状态
VISION_PROMPT = """你是一个游戏画面解析器。请仔细分析这张《Shogun Showdown》的截图，提取所有游戏状态信息。

游戏机制说明：
- 这是一个回合制策略游戏，战斗在 1D 横向网格上进行（5-9 个格子）
- 玩家可以在网格上左右移动、转向、使用技能牌（tiles）、使用道具
- 底部一排是技能栏，每个技能有名称、伤害值、冷却时间
- 技能可以被加入攻击队列（最多3个），显示在角色上方
- 所有敌人的下回合意图都清晰可见（攻击/移动/防御等）
- 道具是免费行动（不消耗回合）

请以 JSON 格式返回完整的游戏状态，格式如下：

```json
{
  "turn_number": 0,
  "grid_size": 7,
  "stage": "Day 1",
  "room": "Combat 1/5",
  "gold": 0,
  "game_over": false,
  "victory": false,
  "player": {
    "character": "Wanderer",
    "hp": 60,
    "max_hp": 60,
    "shield": 0,
    "position": 3,
    "facing": "right",
    "buffs": [],
    "debuffs": []
  },
  "enemies": [
    {
      "name": "Ashigaru",
      "hp": 20,
      "max_hp": 20,
      "shield": 0,
      "position": 5,
      "next_action": "attack",
      "next_action_detail": "8 damage",
      "next_action_target": 3,
      "buffs": [],
      "debuffs": [],
      "is_elite": false,
      "is_boss": false
    }
  ],
  "tiles": [
    {
      "name": "斩击",
      "description": "对前方敌人造成伤害",
      "damage": 6,
      "cooldown_remaining": 0,
      "cooldown_max": 2,
      "aoe": false,
      "range_min": 1,
      "range_max": 1,
      "effects": [],
      "is_upgraded": false
    }
  ],
  "attack_queue": [],
  "consumables": [
    {
      "name": "药水",
      "effect": "回复10 HP",
      "count": 2
    }
  ]
}
```

**关键规则：**
1. position 是 0-based 的网格索引（0 = 最左边，grid_size-1 = 最右边）
2. 如果看不到某个信息，用合理的默认值或 0 / 空字符串
3. 仔细识别每个敌人的下回合意图（这是最重要的信息！）
4. 技能冷却时间如果不可见，设为 0
5. 只返回 JSON，不要有其他文字
"""


def image_to_base64(img: Image.Image) -> str:
    """PIL Image → base64 字符串"""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_screenshot(img: Image.Image) -> GameState:
    """
    将游戏截图发送给 Claude Vision API，返回结构化游戏状态。
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("请在 .env 文件中设置 ANTHROPIC_API_KEY")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    # 压缩图片（减小传输体积，加快解析速度）
    img_compressed = img.copy()
    img_compressed.thumbnail((1920, 1080), Image.LANCZOS)

    img_b64 = image_to_base64(img_compressed)

    message = client.messages.create(
        model=VISION_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )

    # 提取 JSON
    response_text = message.content[0].text
    # 处理可能的 markdown 代码块包裹
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]

    data = json.loads(response_text.strip())
    return state_from_dict(data)


def describe_screenshot(img: Image.Image) -> str:
    """
    将截图转为自然语言描述（用于手动模式，贴给 Claude Code 看）。
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("请在 .env 文件中设置 ANTHROPIC_API_KEY")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    img_compressed = img.copy()
    img_compressed.thumbnail((1920, 1080), Image.LANCZOS)
    img_b64 = image_to_base64(img_compressed)

    message = client.messages.create(
        model=VISION_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": """请详细描述这张 Shogun Showdown 游戏截图的当前状态。包括：

1. 我方角色的血量、位置、朝向、buff/debuff
2. 每个技能牌的名称、伤害、冷却状态、射程
3. 攻击队列中有哪些技能
4. 每个敌人的名称、血量、位置、以及**下回合意图**（最重要！）
5. 可用道具
6. 当前阶段（第几天、第几场战斗）
7. 金币数量

用清晰的结构化中文描述。""",
                    },
                ],
            }
        ],
    )

    return message.content[0].text
