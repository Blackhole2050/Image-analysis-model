"""Real-model smoke test: selected ground-truth facts, not a general accuracy benchmark."""
import json
from pathlib import Path
import sys
import time
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import image_tool as tool

def main():
    root = tool.ROOT / "acceptance-results" / str(time.time_ns())
    root.mkdir(parents=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--case", choices=["table", "flowchart", "chart"])
    args = parser.parse_args()
    engine = tool.Engine(dtype=args.dtype)
    checks = []
    for kind in ([args.case] if args.case else ("table", "flowchart", "chart")):
        try:
            out = root / kind
            tool.analyze(engine, Path(__file__).parent / "fixtures" / (kind + ".png"), out, kind=kind, max_tokens=1024)
            data = json.loads((out / "result.json").read_text(encoding="utf-8"))["data"]
            if kind == "table":
                expected = [["Item", "Quantity", "Price"], ["Apples", "12", "3.50"], ["Pears", "7", "2.25"]]
                passed = any(t["header_rows"] + t["rows"] == expected for t in data["tables"])
            elif kind == "flowchart":
                passed = False
                for graph in data["flowcharts"]:
                    labels = {n["id"]: n["label"].strip().lower() for n in graph["nodes"]}
                    pairs = {(labels[e["source"]], labels[e["target"]]) for e in graph["edges"]}
                    passed |= {("start", "check input"), ("check input", "save result")} <= pairs
            else:
                points = [p for c in data["charts"] for p in c["points"]]
                passed = all(any(p["x"] == x and p["y"] is not None and float(p["y"]) == y and p["value_type"] == "labeled" for p in points) for x, y in (("Jan", 10), ("Feb", 20)))
            checks.append({"case": kind, "passed": bool(passed)})
        except Exception as exc:
            checks.append({"case": kind, "passed": False, "error": str(exc)})
        print(json.dumps(checks[-1]), flush=True)
    (root / "summary.json").write_text(json.dumps(checks, indent=2))
    print("Results:", root)
    return 0 if all(c["passed"] for c in checks) else 1

if __name__ == "__main__":
    raise SystemExit(main())

