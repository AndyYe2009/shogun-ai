"""
Shogun AI - 配置文件
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── AI 后端选择 ─────────────────────────────────────────
# "ollama" = 本地免费模型
# "anthropic" = Claude API（付费，效果最好）
AI_BACKEND = os.getenv("AI_BACKEND", "rule")

# ── Ollama 配置 ──────────────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 视觉模型（解析截图画面）
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "minicpm-v:8b")

# 决策模型（分析状态，输出操作）
OLLAMA_DECISION_MODEL = os.getenv("OLLAMA_DECISION_MODEL", "minicpm-v:8b")

# ── Anthropic API 配置（仅 AI_BACKEND="anthropic" 时使用）─
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
VISION_MODEL = "claude-sonnet-5"
DECISION_MODEL = "claude-sonnet-5"

# ── 游戏窗口 ─────────────────────────────────────────────
GAME_WINDOW_TITLE = "ShogunShowdown"

# ── 键盘映射 ──────────────────────────────────────────
KEY_MOVE_LEFT = "a"
KEY_MOVE_RIGHT = "d"
KEY_TURN = "w"               # 转向/翻转角色朝向
KEY_WAIT = "s"                # 原地等待一回合

# 技能选择：用左右方向键在技能栏中移动光标
KEY_SKILL_LEFT = "left"      # 技能栏光标左移
KEY_SKILL_RIGHT = "right"    # 技能栏光标右移
KEY_SKILL_SELECT = "enter"   # 将当前技能加入等待槽

# 释放技能
KEY_END_TURN = "space"       # 释放等待槽中所有技能

KEY_CONFIRM = "enter"
KEY_CANCEL = "esc"

# 道具快捷键
KEY_CONSUMABLE_1 = "q"
KEY_CONSUMABLE_2 = "e"

# ── HP 配置 ─────────────────────────────────────────────
# 玩家默认最大HP（画面解析无法读取数字时的回退值）
# 不同角色/难度下最大值不同，实际值优先从画面读取
PLAYER_MAX_HP = int(os.getenv("PLAYER_MAX_HP", "60"))
ENEMY_MAX_HP = int(os.getenv("ENEMY_MAX_HP", "20"))

# ── 截图区域裁剪 ─────────────────────────────────────────
# 如果游戏是无边框全屏窗口，截图会包含整个桌面。
# 设置游戏实际显示区域（相对于截图左上角的像素偏移）来裁剪。
# 格式: "left,top,width,height"  留空 = 不裁剪
GAME_CROP = os.getenv("GAME_CROP", "")
# GAME_CROP 示例: "680,0,1027,1067"  (游戏在右侧约60%区域)

# ── 后台模式 ─────────────────────────────────────────────
# True = 不抢焦点、不切屏，通过 Windows API 向游戏窗口发按键
# False = 正常模式（需要游戏窗口在前台）
BACKGROUND_MODE = os.getenv("BACKGROUND_MODE", "false").lower() in ("true", "1", "yes")

# ── 等待时间（秒） ───────────────────────────────────────
WAIT_AFTER_ACTION = 0.15     # 每次操作后等待（加快节奏）
WAIT_AFTER_ENEMY_TURN = 0.5  # 敌方回合后的等待
WAIT_BETWEEN_CYCLES = 0.2    # 决策循环间隔

# ── 自动重启 ─────────────────────────────────────────────
# 死亡后长按 W 键重新开始游戏
RESTART_HOLD_W = float(os.getenv("RESTART_HOLD_W", "3.0"))  # 长按秒数
RESTART_WAIT = float(os.getenv("RESTART_WAIT", "4.0"))      # 重启后等待秒数
