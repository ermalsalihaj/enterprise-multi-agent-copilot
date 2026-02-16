"""Run evaluation over test prompts using MODEL_EVAL and write eval_results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import settings
from agents.graph import run_copilot


def main() -> None:
    prompts_path = Path(__file__).parent / "test_prompts.txt"
    if not prompts_path.exists():
        print(f"Missing {prompts_path}")
        sys.exit(1)
    lines = prompts_path.read_text(encoding="utf-8").strip().splitlines()
    results = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("||", 1)
        question = parts[0].strip()
        goal = parts[1].strip() if len(parts) > 1 else settings.eval_goal
        print(f"Running {i + 1}/{len(lines)}: {question[:50]}...")
        try:
            out = run_copilot(
                question=question,
                goal=goal,
                output_mode=settings.eval_output_mode,
            )
            results.append({
                "question": question,
                "goal": goal,
                "verified_output": out.get("verified_output", {}),
                "observability": out.get("observability", {}),
            })
        except Exception as e:
            results.append({
                "question": question,
                "goal": goal,
                "error": str(e),
                "verified_output": {},
                "observability": {},
            })
    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
