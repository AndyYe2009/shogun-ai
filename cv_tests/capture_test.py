"""
CV 测试截图采集工具

用法:
    python cv_tests/capture_test.py

运行后会不断截图，每次截图后提示输入场景描述。
输入 'q' 退出，输入描述文字保存截图。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TEST_DIR = Path(__file__).parent
SCREENSHOT_DIR = TEST_DIR / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

from capture import capture_screenshot


def main():
    print("=" * 60)
    print("  CV 解析器 — 测试截图采集")
    print("=" * 60)
    print()
    print("使用方法:")
    print("  1. 在游戏中摆好你要测试的场景")
    print("  2. 输入描述（如 '1敌人满血,2技能就绪,位置3朝右'）")
    print("  3. 截图自动保存到 cv_tests/screenshots/")
    print("  4. 输入 'q' 退出")
    print()

    try:
        from capture import get_window_rect
        rect = get_window_rect()
        print(f"游戏窗口: {rect['width']}x{rect['height']} "
              f"at ({rect['left']}, {rect['top']})")
    except RuntimeError:
        print("⚠ 未检测到游戏窗口，截图会截整个屏幕")
    print()

    count = 0
    while True:
        cmd = input(f"[已采集 {count} 张] 输入场景描述 (q=退出): ").strip()

        if cmd.lower() == 'q':
            break

        if not cmd:
            print("  跳过 (空输入)")
            continue

        # 截图
        try:
            img = capture_screenshot()
        except RuntimeError as e:
            print(f"  截图失败: {e}")
            continue

        # 用描述做文件名
        safe_name = cmd.replace(" ", "_").replace("/", "-")[:60]
        filename = f"{count+1:03d}_{safe_name}.png"
        save_path = SCREENSHOT_DIR / filename
        img.save(str(save_path))
        print(f"  ✓ 已保存: {filename} ({img.size[0]}x{img.size[1]})")
        count += 1

    print(f"\n共采集 {count} 张截图，保存在: {SCREENSHOT_DIR}")
    print("下一步: 编辑 cv_tests/annotations.json 标注每张图")


if __name__ == "__main__":
    main()
