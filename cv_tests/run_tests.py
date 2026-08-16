"""
CV 解析器验证框架

用法:
    python cv_tests/run_tests.py                    # 运行所有测试
    python cv_tests/run_tests.py --detail           # 显示每个字段的详细对比
    python cv_tests/run_tests.py --only 1,3,5       # 只跑指定的测试编号

标注格式见 annotations.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# 把父目录加入 sys.path，以便导入项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

TEST_DIR = Path(__file__).parent
SCREENSHOT_DIR = TEST_DIR / "screenshots"
ANNOTATION_PATH = TEST_DIR / "annotations.json"

from screen_parser import analyze_screenshot


def load_tests() -> list[dict]:
    """Load all test cases from annotations.json"""
    if not ANNOTATION_PATH.exists():
        print(f"ERROR: Annotation file not found: {ANNOTATION_PATH}")
        sys.exit(1)

    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    # 过滤掉示例（以 _ 开头的标注）
    real_tests = [t for t in tests if not t.get("_comment")]
    return real_tests


def run_single_test(test: dict) -> dict:
    """
    运行单个测试用例。

    Returns: {
        "name": str,
        "passed": bool,
        "results": {field: {expected, actual, match}},
        "errors": [str],
        "error_count": int,
    }
    """
    name = test["screenshot"]
    gt = test["ground_truth"]
    img_path = SCREENSHOT_DIR / name

    result = {
        "name": name,
        "passed": True,
        "results": {},
        "errors": [],
        "error_count": 0,
    }

    # 加载截图
    if not img_path.exists():
        result["errors"].append(f"Screenshot not found: {img_path}")
        result["passed"] = False
        result["error_count"] = 999
        return result

    try:
        img = Image.open(img_path)
    except Exception as e:
        result["errors"].append(f"Cannot open screenshot: {e}")
        result["passed"] = False
        result["error_count"] = 999
        return result

    # 运行解析器
    try:
        parsed = analyze_screenshot(img)
    except Exception as e:
        result["errors"].append(f"Parser crashed: {e}")
        result["passed"] = False
        result["error_count"] = 999
        return result

    p = parsed["player"]
    enemies = parsed["enemies"]
    skills = parsed["skills"]
    gt_player = gt.get("player", {})
    gt_enemies = gt.get("enemies", [])
    gt_skills = gt.get("skills", [])
    gt_grid = gt.get("grid_size", 7)

    # ── 1. 网格大小 ────────────────────────────────
    parsed_grid = parsed.get("grid_size", 7)
    r = _check("grid_size", gt_grid, parsed_grid, exact=True)
    result["results"]["grid_size"] = r

    # ── 2. 玩家位置 ────────────────────────────────
    parsed_pos = p.get("grid_pos", -1)
    r = _check("player.position", gt_player.get("position"), parsed_pos, exact=True)
    result["results"]["player.position"] = r

    # ── 3. 玩家朝向 ────────────────────────────────
    parsed_facing = p.get("facing", "?")
    r = _check("player.facing", gt_player.get("facing"), parsed_facing, exact=True)
    result["results"]["player.facing"] = r

    # ── 4. 玩家 HP% ────────────────────────────────
    gt_hp = gt_player.get("hp_pct")
    if gt_hp is not None:
        parsed_hp = p.get("hp_pct", -1)
        r = _check("player.hp_pct", gt_hp, parsed_hp, tolerance=15)
        result["results"]["player.hp_pct"] = r

    # ── 5. 玩家护盾 ────────────────────────────────
    if "shield" in gt_player:
        r = _check("player.shield", gt_player["shield"], p.get("shield", False), exact=True)
        result["results"]["player.shield"] = r

    # ── 6. 敌人数量 ────────────────────────────────
    gt_count = len(gt_enemies)
    parsed_count = len(enemies)
    r = _check("enemy_count", gt_count, parsed_count, exact=True)
    result["results"]["enemy_count"] = r

    # ── 7. 每个敌人的位置 ──────────────────────────
    # 按 position 排序后逐位比较
    gt_enemies_sorted = sorted(gt_enemies, key=lambda e: e.get("position", 99))
    enemies_sorted = sorted(enemies, key=lambda e: e.get("grid_pos", 99))
    for i in range(max(gt_count, parsed_count)):
        label = f"enemy[{i}].position"
        if i < gt_count and i < parsed_count:
            gt_pos = gt_enemies_sorted[i].get("position")
            parsed_epos = enemies_sorted[i].get("grid_pos", -1)
            r = _check(label, gt_pos, parsed_epos, exact=True)
            result["results"][label] = r
        elif i < gt_count:
            r = _check(label, f"pos={gt_enemies_sorted[i].get('position')}", "MISSING", exact=True)
            result["results"][label] = r
        else:
            r = _check(label, "no enemy", f"extra: pos={enemies_sorted[i].get('grid_pos')}", exact=True)
            result["results"][label] = r

    # ── 8. 敌人 HP% (只比较匹配到的) ────────────────
    for i in range(min(gt_count, parsed_count)):
        if "hp_pct" in gt_enemies_sorted[i]:
            label = f"enemy[{i}].hp_pct"
            gt_ehp = gt_enemies_sorted[i]["hp_pct"]
            parsed_ehp = enemies_sorted[i].get("hp_pct", -1)
            r = _check(label, gt_ehp, parsed_ehp, tolerance=20)
            result["results"][label] = r

    # ── 9. 技能状态 ────────────────────────────────
    for i in range(6):
        gt_skill = gt_skills[i] if i < len(gt_skills) else {"status": "empty"}
        parsed_skill = skills[i] if i < len(skills) else {"status": "empty"}

        # 统一状态名
        gt_status = gt_skill.get("status", "empty")
        parsed_status = parsed_skill.get("status", "empty")
        if parsed_skill.get("ready"):
            parsed_status = "ready"
        elif parsed_status == "cd":
            parsed_status = "cooldown"

        label = f"skill[{i}].status"
        r = _check(label, gt_status, parsed_status, exact=True)
        result["results"][label] = r

    # ── 统计 ────────────────────────────────────
    error_count = sum(1 for v in result["results"].values() if not v["match"])
    result["error_count"] = error_count
    if error_count > 0 or result["errors"]:
        result["passed"] = False

    return result


def _check(label: str, expected, actual, exact: bool = False,
           tolerance: int = 0) -> dict:
    """Compare expected vs actual value"""
    match = False
    if exact:
        match = (expected == actual)
    else:
        try:
            match = abs(float(expected) - float(actual)) <= tolerance
        except (TypeError, ValueError):
            match = (str(expected) == str(actual))

    return {
        "expected": expected,
        "actual": actual,
        "match": match,
    }


def print_results(all_results: list[dict], detail: bool = False):
    """输出测试报告"""
    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    crashed = sum(1 for r in all_results if r["error_count"] >= 999)

    # 按字段汇总
    field_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for result in all_results:
        for field, r in result["results"].items():
            field_stats[field]["total"] += 1
            if r["match"]:
                field_stats[field]["correct"] += 1

    print("\n" + "=" * 70)
    print(f"  CV Parser Test Report")
    print(f"  Tests: {total} | Passed: {passed} | Failed: {total - passed}")
    if crashed:
        print(f"  Crashed: {crashed} (screenshot missing or parser error)")
    print("=" * 70)

    # Per-field accuracy
    print(f"\n{'Field':<25} {'Accuracy':>8}  {'Correct/Total':>10}")
    print("-" * 50)
    for field in sorted(field_stats.keys()):
        s = field_stats[field]
        rate = s["correct"] / s["total"] * 100
        bar = _bar(rate)
        print(f"  {field:<23} {rate:5.1f}% {bar} {s['correct']:>3}/{s['total']}")

    # Failure details
    if detail:
        print(f"\n{'─' * 70}")
        print(f"  Failure Details")
        print(f"{'─' * 70}")
        for result in all_results:
            if result["passed"]:
                continue
            print(f"\n  [{result['name']}]")
            for error in result["errors"]:
                print(f"    ERROR: {error}")
            for field, r in result["results"].items():
                if not r["match"]:
                    print(f"    X {field}: expected={r['expected']}, actual={r['actual']}")


def _bar(pct: float) -> str:
    """ASCII 进度条"""
    if pct >= 90:
        return "######"
    elif pct >= 70:
        return "####.."
    elif pct >= 50:
        return "###..."
    else:
        return "#....."


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CV 解析器验证")
    parser.add_argument("--detail", action="store_true",
                        help="显示每个不匹配字段的详细对比")
    parser.add_argument("--only", type=str, default="",
                        help="只测试指定编号（逗号分隔，如 1,3,5）")
    parser.add_argument("--dump", type=str, default="",
                        help="导出指定截图的解析结果（调试用）")
    args = parser.parse_args()

    # Debug mode: export parsed result for a single screenshot
    if args.dump:
        img_path = SCREENSHOT_DIR / args.dump
        if not img_path.exists():
            print(f"Screenshot not found: {img_path}")
            sys.exit(1)
        img = Image.open(img_path)
        result = analyze_screenshot(img)
        # 简化输出（去掉 numpy 数组等不可序列化内容）
        clean = {
            "image_size": result["image_size"],
            "grid_y": result.get("grid_y", "?"),
            "player": {k: v for k, v in result["player"].items()
                       if not isinstance(v, (tuple, list)) or len(str(v)) < 200},
            "enemies": [{k: v for k, v in e.items()
                        if not isinstance(v, (tuple, list)) or len(str(v)) < 200}
                       for e in result["enemies"]],
            "skills": [{"index": s["index"], "status": s["status"],
                        "ready": s["ready"], "cooldown_remaining": s["cooldown_remaining"]}
                       for s in result["skills"]],
            "enemy_count": result["enemy_count"],
            "danger_level": result["danger_level"],
        }
        print(json.dumps(clean, ensure_ascii=False, indent=2))
        return

    tests = load_tests()

    if not tests:
        print("No test cases found. Please add annotations in annotations.json.")
        return

    # 过滤
    if args.only:
        indices = [int(x.strip()) - 1 for x in args.only.split(",") if x.strip().isdigit()]
        tests = [tests[i] for i in indices if 0 <= i < len(tests)]
        print(f"Running selected tests: {[t['screenshot'] for t in tests]}")

    # 运行
    all_results = []
    for i, test in enumerate(tests):
        name = test["screenshot"]
        print(f"  [{i+1}/{len(tests)}] {name} ...", end=" ")
        result = run_single_test(test)
        all_results.append(result)

        if result["error_count"] >= 999:
            print("CRASHED")
        elif result["passed"]:
            print("PASS")
        else:
            errors = result["error_count"]
            print(f"FAIL ({errors} mismatches)")

    print_results(all_results, detail=args.detail)


if __name__ == "__main__":
    main()
