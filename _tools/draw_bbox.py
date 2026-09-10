# -*- coding: utf-8 -*-
"""Draw the LLM plan bboxes on an image (article 2.2.1 diagnostic figure).

Usage:
  python draw_bbox.py --src <image> --plan <plan.json> [--out <png>]
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import imread                                   # noqa: E402
from pipeline.llm_planner import (convert_coords, expand_instances,  # noqa: E402
                                  load_plan)
from pipeline.steps.plan import draw_plan_boxes                      # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    img = imread(a.src)
    if img is None:
        raise SystemExit(f"cannot decode {a.src}")
    H, W = img.shape[:2]
    queries = (load_plan(a.plan).get("queries")) or []
    convert_coords(queries, H, W)
    inst = expand_instances(queries)
    out = a.out or os.path.splitext(a.src)[0] + "_plan.png"
    draw_plan_boxes(a.src, inst, out, W, H)
    print(f"Saved: {out}\n  queries: {len(queries)}\n  instances: {len(inst)}\n  image: {W}x{H}")