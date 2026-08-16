"""
Ollama API 客户端 — 替代 Anthropic API
使用本地 Ollama 模型进行画面解析和决策
"""

import json
import base64
from io import BytesIO
from PIL import Image
import httpx
from config import (
    OLLAMA_HOST,
    OLLAMA_VISION_MODEL,
    OLLAMA_DECISION_MODEL,
)


def image_to_base64(img: Image.Image) -> str:
    """PIL Image → base64 字符串"""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def ollama_chat(
    model: str,
    prompt: str,
    images: list[str] | None = None,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> str:
    """
    调用 Ollama Chat API。

    Args:
        model: 模型名称
        prompt: 用户提示
        images: base64 编码的图片列表（可选，视觉模型用）
        system: 系统提示
        temperature: 温度参数
        max_tokens: 最大输出 token
        timeout: 超时秒数

    Returns:
        模型返回的文本
    """
    url = f"{OLLAMA_HOST}/api/chat"

    # 构建消息
    # llava/older models: content 是纯文本，images 是单独字段
    # 新版 models: content 是数组 [{type: "image_url", ...}, {type: "text", ...}]
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if images:
        payload["images"] = images

    if system:
        payload["messages"].insert(0, {"role": "system", "content": system})

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "")
            if not content:
                print(f"  [Ollama EMPTY] keys={list(result.keys())}", flush=True)
                if "error" in result:
                    print(f"  [Ollama ERROR] {result['error']}", flush=True)
                if "done_reason" in result:
                    print(f"  [Ollama done_reason] {result['done_reason']}", flush=True)
            return content
    except httpx.ConnectError:
        raise RuntimeError(
            f"无法连接到 Ollama ({OLLAMA_HOST})。\n"
            "请确保 Ollama 正在运行：在终端执行 'ollama serve'"
        )
    except httpx.HTTPStatusError as e:
        body = e.response.text[:500] if e.response else ""
        raise RuntimeError(f"Ollama API 错误 ({e.response.status_code}): {body}")


# ── 画面解析 Prompt ──────────────────────────────────────

VISION_PROMPT = """你是一个游戏画面解析器。请仔细分析这张《Shogun Showdown》的截图，提取所有游戏状态信息。

游戏机制说明：
- 回合制策略游戏，战斗在 1D 横向网格上进行（5-9 个格子）
- 玩家可以在网格上左右移动、转向、使用技能牌（tiles）、使用道具
- 底部一排是技能栏，每个技能有名称、伤害值、冷却时间
- 技能可以被加入攻击队列（最多3个），显示在角色上方
- 所有敌人的下回合意图都清晰可见（攻击/移动/防御等）
- 道具是免费行动（不消耗回合）

请以 JSON 格式返回完整的游戏状态：

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
      "name": "Slash",
      "description": "Damage enemy in front",
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
      "name": "Potion",
      "effect": "Restore 10 HP",
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
5. 只返回 JSON，不要有其他文字"""


def parse_screenshot_ollama(img: Image.Image) -> dict:
    """
    使用本地 Ollama 视觉模型解析游戏截图。
    返回 JSON 字典。
    """
    # 压缩图片
    img_compressed = img.copy()
    img_compressed.thumbnail((1280, 720), Image.LANCZOS)
    img_b64 = image_to_base64(img_compressed)

    response = ollama_chat(
        model=OLLAMA_VISION_MODEL,
        prompt=VISION_PROMPT,
        images=[img_b64],
        temperature=0.0,
        max_tokens=4096,
        timeout=120,
    )

    # 提取 JSON
    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    return json.loads(text)


# ── 决策 Prompt ──────────────────────────────────────────

DECISION_SYSTEM_PROMPT = """Output ONLY this JSON, nothing else:
{"reasoning":"short","actions":[{"type":"X"}]}

X must be one of:
- "move" with "direction":"left" or "right"
- "turn" to flip facing
- "wait_turn" to skip
- "reset_cursor" then "queue" with "tile_index":N (0-5)
- "end_turn" to release queued skills

RULES:
- queue and end_turn are SEPARATE turns
- Only queue skills marked USE THIS (READY, not CANNOT USE)
- **Prefer [AOE] melee skills when enemies are on both sides or adjacent**
- Use ranged skills (Arrow/Shuriken) for distant enemies, not adjacent ones
- Attack enemies for +10 score per kill +5 per dmg
- Avoid damage (-30/hp)

NO other text. ONLY the JSON."""


def format_playbook_examples(examples: list) -> str:
    """
    将 playbook 中的成功经验格式化为 few-shot 示例文本。
    """
    if not examples:
        return ""

    lines = []
    lines.append("\n## PREVIOUS SUCCESSFUL PLAYS (learn from these!):")
    for i, ex in enumerate(examples):
        lines.append(f"\n### Example {i + 1} (score: +{ex.score_delta}):")
        lines.append(f"  Scenario: {ex.fingerprint.get('enemy_count', '?')} enemies, "
                     f"{ex.fingerprint.get('ready_skills', '?')} skills ready, "
                     f"HP ~{ex.fingerprint.get('player_hp_pct', '?')}%")
        lines.append(f"  Reasoning: {ex.reasoning}")
        lines.append(f"  Actions: {json.dumps(ex.actions, ensure_ascii=False)}")
    lines.append("\nUse these as reference for similar situations.")
    return "\n".join(lines)


def decide_action_ollama(game_state_text: str,
                          playbook_examples: list | None = None) -> dict:
    """
    使用本地 Ollama 模型做决策。
    返回 {"reasoning": "...", "actions": [...]}

    playbook_examples: 从战术手册匹配到的相似成功案例列表

    如果配置的模型不可用，自动回退到可用模型。
    """
    # 将 playbook 示例注入用户 prompt
    if playbook_examples:
        examples_text = format_playbook_examples(playbook_examples)
        full_prompt = game_state_text + "\n" + examples_text
    else:
        full_prompt = game_state_text

    # 尝试配置的模型，失败则回退
    models_to_try = [OLLAMA_DECISION_MODEL]
    # 添加回退模型（去重）
    available = list_models()
    fallbacks = ["minicpm-v:8b", "llava:7b"]
    for fb in fallbacks:
        if fb not in models_to_try and fb in available:
            models_to_try.append(fb)

    last_error = None
    for model in models_to_try:
        try:
            response = ollama_chat(
                model=model,
                prompt=full_prompt,
                system=DECISION_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=1024,
                timeout=45,
            )
            break
        except Exception as e:
            last_error = e
            continue
    else:
        raise RuntimeError(f"All models failed: {last_error}")

    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        result = json.loads(text)
        print(f"  [Ollama raw JSON]: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result
    except json.JSONDecodeError:
        print(f"  [Ollama BAD JSON]: {text[:200]}")
        return {
            "reasoning": "JSON parse failed, fallback to wait",
            "actions": [{"type": "wait_turn"}],
        }


# ── 技能识别 ────────────────────────────────────────────

SKILL_IDENTIFY_PROMPT = """You are analyzing the skill bar from the game Shogun Showdown.

The image shows the bottom portion of the game screen with 6 skill tile slots (0-5, left to right).

For each of the 6 slots, identify:
1. The skill name (the text label on the tile)
2. Whether it's empty (dark/blank slot with no card)

Return ONLY valid JSON:
```json
{
  "skills": [
    {"index": 0, "name": "Slash", "empty": false},
    {"index": 1, "name": "Dagger", "empty": false},
    {"index": 2, "name": "", "empty": true},
    {"index": 3, "name": "Arrow", "empty": false},
    {"index": 4, "name": "Push", "empty": false},
    {"index": 5, "name": "Block", "empty": false}
  ]
}
```

Important:
- Read the ACTUAL text on each tile, don't guess
- If a slot is empty/dark with no visible card, set empty: true and name: ""
- Skill names are usually in English (e.g. Slash, Dagger, Sweep, Arrow, Push, Block)
- If you can't read the text clearly, set name to what you think it says
- Return ONLY the JSON, no other text"""


def identify_skills_ollama(img: Image.Image) -> list[dict]:
    """
    使用视觉模型识别技能栏中的所有技能。

    Args:
        img: 完整游戏截图

    Returns:
        [{"index": 0, "name": "Slash", "empty": False}, ...]
    """
    h, w = img.height, img.width
    # 裁剪底部技能栏区域 (画面底部 22%)
    skill_bar = img.crop((0, int(h * 0.78), w, h))

    # 压缩图片
    skill_bar.thumbnail((1280, 300), Image.LANCZOS)
    img_b64 = image_to_base64(skill_bar)

    try:
        response = ollama_chat(
            model=OLLAMA_VISION_MODEL,
            prompt=SKILL_IDENTIFY_PROMPT,
            images=[img_b64],
            temperature=0.0,
            max_tokens=1024,
            timeout=60,
        )
    except Exception as e:
        print(f"  WARNING: Skill identification failed: {e}")
        return []

    text = response.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        data = json.loads(text)
        return data.get("skills", [])
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse skill JSON: {text[:200]}")
        return []


def check_ollama_available() -> bool:
    """检查 Ollama 是否可用"""
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{OLLAMA_HOST}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


def list_models() -> list[str]:
    """列出已安装的模型"""
    try:
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{OLLAMA_HOST}/api/tags")
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
