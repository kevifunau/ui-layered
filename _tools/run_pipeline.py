# -*- coding: utf-8 -*-
"""run_pipeline.py -- entry point of the UI layer decomposition pipeline.

Step order follows the article:

    2.1.1/2.1.2  OCR + removal of every glyph      -> 01_text_removed.png (LLM input)
    2.2.1        LLM layer plan (original + cleaned image + OCR id list)
    2.1.4/2.1.3  restore raster text, add OCR misses, estimate text styles
    2.2.2-2.2.4  SAM segmentation, candidate scoring, closing + component filter
    layer build  occlusion from the parent chain, alpha and repair holes
    2.3          element repair (image model atlas / traditional) + background repair
    2.4          layer export, reassembly, text re-render
    audit        lossless self-check + audit.json
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config as C                                   # noqa: E402
from pipeline.context import PipelineContext, RunConfig            # noqa: E402
from pipeline.steps import (AuditStep, ExportStep, LayerBuildStep, Pipeline,  # noqa: E402
                            PlanStep, RepairStep, SegmentationStep,
                            TextExtractStep, TextFinalizeStep)


def parse_args():
    ap = argparse.ArgumentParser(description="UI layer decomposition pipeline")
    ap.add_argument("--src", default=None, help="source UI screenshot")
    ap.add_argument("--dest", default=C.DEST_DEFAULT, help="output directory")
    ap.add_argument("--plan", default="", help="use this plan JSON instead of resolving one")
    ap.add_argument("--repair", choices=["auto", "ns", "gen"], default="auto",
                    help="auto = follow the LLM element_repair_mode (article default)")
    ap.add_argument("--bg", choices=["auto", "ns", "gen"], default="auto",
                    help="auto = follow background_repair.mode")
    ap.add_argument("--gen-backend", choices=["seedream", "flux", "dashscope"],
                    default=C.GEN_PROVIDER_DEFAULT,
                    help="ISS-035 generative inpainting provider (default seedream pro)")
    ap.add_argument("--gen", action="store_true", help="shortcut for --repair gen --bg gen")
    ap.add_argument("--ns", action="store_true", help="shortcut for --repair ns --bg ns")
    ap.add_argument("--refresh", action="store_true", help="ignore the inpaint cache")
    ap.add_argument("--bg-estimate", choices=["median", "kmeans"], default="median",
                    help="text background estimator (article: median)")
    ap.add_argument("--no-fg-core-filter", action="store_true",
                    help="use every glyph pixel for the text colour buckets")
    ap.add_argument("--no-safe-area", action="store_true",
                    help="do not add the synthetic status bar layer")
    ap.add_argument("--safe-area", type=int, default=None, help="status bar height in px")
    ap.add_argument("--cc-keep-ratio", type=float, default=C.CC_KEEP_AREA_RATIO,
                    help="article 2.2.4 rule 3 threshold")
    ap.add_argument("--no-atlas-prefill", action="store_true",
                    help="send pure black holes to the image model (article wording)")
    ap.add_argument("--model", default=C.LLM_MODEL, help="vision LLM used for 2.2.1")
    ap.add_argument("--dropped-ids", default="", help="comma separated layer ids to skip")
    a = ap.parse_args()

    repair, bg = a.repair, a.bg
    if a.gen:
        repair, bg = "gen", "gen"
    if a.ns:
        repair, bg = "ns", "ns"
    src = a.src or C.SRC_DEFAULT
    if not os.path.exists(src):
        ap.error(f"source image not found: {src}")

    case = C.load_case_config(src)
    dropped = set(case.get("dropped_ids") or ())
    if a.dropped_ids:
        dropped |= {x.strip() for x in a.dropped_ids.split(",") if x.strip()}
    safe = case.get("safe_area_height", C.SAFE_AREA_HEIGHT)
    if a.no_safe_area:
        safe = 0
    if a.safe_area is not None:
        safe = a.safe_area

    cache = os.path.join(C.OUT, "data", "inpaint_cache_v6")
    os.makedirs(cache, exist_ok=True)
    try:
        api_key = C.load_api_key()
    except RuntimeError:
        api_key = ""       # only needed when no cached plan exists

    return RunConfig(
        src=src, dest=a.dest, repair=repair, bg=bg, refresh=a.refresh, cache=cache,
        bg_estimate=a.bg_estimate, fg_core_filter=not a.no_fg_core_filter,
        cc_keep_ratio=a.cc_keep_ratio, safe_area_height=int(safe or 0),
        dropped_ids=frozenset(dropped), llm_model=a.model, api_key=api_key,
        bg_prompt=case.get("bg_prompt") or "", atlas_prompt=case.get("atlas_prompt") or "",
        atlas_prefill=not a.no_atlas_prefill, gen_backend=a.gen_backend,
        plan_path=a.plan, truth_bbox=case.get("truth_bbox") or {}, case_config=case)


def main():
    cfg = parse_args()
    os.makedirs(cfg.dest, exist_ok=True)
    layers_dir = os.path.join(cfg.dest, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    for f in os.listdir(layers_dir):
        if f.lower().endswith(".png"):
            os.remove(os.path.join(layers_dir, f))

    ctx = PipelineContext(cfg)
    ctx.src = C.imread(cfg.src)
    if ctx.src is None:
        raise SystemExit(f"cannot decode source image: {cfg.src}")
    ctx.H, ctx.W = ctx.src.shape[:2]
    ctx.log(f"source {os.path.basename(cfg.src)} {ctx.W}x{ctx.H} -> {cfg.dest}")
    ctx.log(f"repair={cfg.repair} bg={cfg.bg} bg_estimate={cfg.bg_estimate} "
            f"case={os.path.basename(cfg.case_config.get('_path', '-'))}")

    Pipeline([
        TextExtractStep(),      # 2.1.1 + 2.1.2
        PlanStep(),             # 2.2.1
        TextFinalizeStep(),     # 2.1.4 + 2.1.3
        SegmentationStep(),     # 2.2.2 + 2.2.3 + 2.2.4
        LayerBuildStep(),       # occlusion / alpha / holes
        RepairStep(),           # 2.3.1 + 2.3.2 + 2.3.3
        ExportStep(),           # 2.4
        AuditStep(),
    ]).run(ctx)

    print(f"\nDone in {ctx.timings.get('total')}s. Output: {cfg.dest}")
    print(f"  layers       : {len(ctx.manifest)}")
    print(f"  lossless     : {'PASS' if ctx.unexp == 0 else 'FAIL'} "
          f"(unexpected diff {ctx.unexp}px)")
    print(f"  element fix  : {ctx.atlas_report.get('gen', 0)} generative / "
          f"{ctx.atlas_report.get('ns', 0)} traditional")
    print(f"  background   : {ctx.bg_method}")


if __name__ == "__main__":
    main()