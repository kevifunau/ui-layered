# -*- coding: utf-8 -*-
"""gen_llm_plan.py -- standalone 2.2.1 LLM layer planning.

Mirrors what PlanStep does inside the pipeline, so a plan can be produced (and
inspected) before a full run:

  1. RapidOCR with the article's 2.1.1 filters -> stable ids text_001...
  2. remove every glyph (2.1.2)                  -> the OCR-cleaned working image
  3. send original + cleaned + prompt + OCR id list to the vision LLM
  4. save data/plans/<source stem>.json and optionally draw 01b_LLM_bbox.png

Usage:
  python gen_llm_plan.py --src TestCase\羊了个羊\source_screens\screenShot_...jpeg
  python gen_llm_plan.py --src ... --draw --dest output\基础测试
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config as C                                    # noqa: E402
from pipeline.llm_planner import (convert_coords, expand_instances,  # noqa: E402
                                  generate_llm_plan, load_api_key)
from pipeline.prompts import PLAN_PROMPT                            # noqa: E402
from pipeline.steps.plan import draw_plan_boxes                     # noqa: E402
from pipeline.steps.text import TextPipeline                        # noqa: E402


class _Cfg:
    """Minimal stand-in for RunConfig (TextPipeline only reads two fields)."""

    bg_estimate = "median"
    fg_core_filter = True


def main():
    ap = argparse.ArgumentParser(description="Generate the 2.2.1 LLM layer plan")
    ap.add_argument("--src", required=True, help="source UI screenshot")
    ap.add_argument("--cleaned", default=None,
                    help="OCR-cleaned image (default: generated with the 2.1.2 algorithm)")
    ap.add_argument("--dest", default=None,
                    help="where to write the cleaned image / diagnostic (default: data/tmp)")
    ap.add_argument("--out", default=None, help="plan path (default data/plans/<stem>.json)")
    ap.add_argument("--draw", action="store_true", help="also draw 01b_LLM_bbox.png")
    ap.add_argument("--model", default=C.LLM_MODEL)
    a = ap.parse_args()

    src = a.src
    if not os.path.exists(src):
        raise SystemExit(f"source not found: {src}")
    dest = a.dest or os.path.join(C.DATA_DIR, "tmp")
    os.makedirs(dest, exist_ok=True)

    img = C.imread(src)
    if img is None:
        raise SystemExit(f"cannot decode: {src}")
    H, W = img.shape[:2]

    tp = TextPipeline(img, H, W, dest, _Cfg())
    tp.run_ocr()
    tp.build_glyph_masks()
    tp.mark_all_editable()
    tp.compose_textmask()
    tp.make_plate()

    cleaned = a.cleaned
    if not cleaned:
        cleaned = os.path.join(dest, "01_text_removed.png")
        C.imwrite(cleaned, tp.plate)
        print(f"OCR-cleaned working image -> {cleaned}")
    if not os.path.exists(cleaned):
        print(f"[WARN] cleaned image missing ({cleaned}), using the source")
        cleaned = src

    plan_path = generate_llm_plan(src, cleaned, tp.entries, H, W, a.model,
                                  api_key=load_api_key())
    if a.out and os.path.abspath(a.out) != os.path.abspath(plan_path):
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        shutil.copyfile(plan_path, a.out)
        plan_path = a.out

    full = load_plan_safe(plan_path)
    queries = full.get("queries") or []
    convert_coords(queries, H, W)
    inst = expand_instances(queries)
    print(f"plan -> {plan_path}")
    print(f"  queries={len(queries)} instances={len(inst)}")
    print(f"  background_repair={full.get('background_repair')}")
    print(f"  raster_text_ids={full.get('raster_text_ids')}")
    print(f"  text_corrections={len(full.get('text_corrections') or [])}")
    print(f"  OCR ids sent={len(tp.entries)} (e.g. {[o['id'] for o in tp.entries[:3]]})")

    if a.draw:
        base = cleaned if os.path.exists(cleaned) else src
        out = draw_plan_boxes(base, inst, os.path.join(dest, "01b_LLM_bbox.png"), W, H)
        if out:
            print(f"  diagnostic -> {out}")


def load_plan_safe(path):
    from pipeline.llm_planner import load_plan
    return load_plan(path)


if __name__ == "__main__":
    print(f"prompt: {len(PLAN_PROMPT)} chars (article 2.2.1, verbatim)")
    main()