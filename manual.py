"""
手动辅助模式 — 截图后用 Claude Code 做决策
你截图 → 粘贴状态描述给我 → 我返回操作指令 → 你执行
"""

import time
import sys
from pathlib import Path

from capture import capture_screenshot
from vision import describe_screenshot


def manual_capture():
    """
    截图并生成自然语言描述，适合粘贴给 Claude Code。

    用法:
        python manual.py capture
    """
    print("📸 截图中...")
    try:
        screenshot = capture_screenshot()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    # 保存截图
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = screenshot_dir / f"manual_{timestamp}.png"
    screenshot.save(str(save_path))
    print(f"💾 截图已保存: {save_path}")

    # 生成描述
    print("\n🔍 解析画面中...")
    try:
        description = describe_screenshot(screenshot)
    except Exception as e:
        print(f"⚠️ 画面解析失败: {e}")
        print(f"你可以手动查看截图并描述状态给我")
        return

    print("\n" + "=" * 60)
    print("📋 游戏状态描述 (复制以下内容发送给 Claude Code):")
    print("=" * 60)
    print(description)
    print("\n💡 发送给我的格式: '分析这个状态，我该怎么做？' 然后粘贴上面的描述")


def manual_execute():
    """
    执行手动输入的动作序列。
    用户从 Claude Code 获取动作后，通过此函数执行。

    用法:
        python manual.py execute
    """
    from executor import execute_action
    import json

    print("📋 粘贴 Claude Code 返回的动作 JSON (粘贴后按 Enter，然后按 Ctrl+Z 再按 Enter):")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    try:
        text = "\n".join(lines)
        # 尝试提取 JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        decision = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        return

    reasoning = decision.get("reasoning", "")
    actions = decision.get("actions", [])
    print(f"\n💭 {reasoning}")
    print(f"⚡ 执行 {len(actions)} 个动作...")

    print("⏳ 3 秒后开始执行，切换到游戏窗口！")
    time.sleep(3)

    for i, action in enumerate(actions):
        result = execute_action(action)
        print(f"  [{i + 1}/{len(actions)}] {result}")

    print("\n✅ 执行完毕")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shogun AI - 手动辅助模式")
    subparsers = parser.add_subparsers(dest="command")

    capture_parser = subparsers.add_parser("capture", help="截图并生成状态描述")
    execute_parser = subparsers.add_parser("execute", help="执行 Claude Code 返回的动作")

    args = parser.parse_args()

    if args.command == "capture":
        manual_capture()
    elif args.command == "execute":
        manual_execute()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
