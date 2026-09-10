# -*- coding: utf-8 -*-
"""2.2.1 LLM layer planning as a real pipeline step.

Order matters: the article feeds the LLM the *OCR-cleaned* image, so this step runs
after TextExtractStep and before the raster-text restore.  It also expands multi-hint
queries (one geometry hint per repeated instance) into independent instances.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import OUT, PLAN_DIR, T, imread, load_api_key
from ..llm_planner import (convert_coords, corrections_to_boxes, expand_instances,
                           generate_llm_plan, load_plan, resolve_plan)
from ..prompts import PLAN_PROMPT
from .base import PipelineStep

TYPE_COLORS = {
    "panel": (245, 165, 66),
    "card": (106, 187, 102),
    "button": (77, 183, 255),
    "icon": (188, 71, 171),
    "logo": (146, 98, 240),
    "badge": (80, 83, 239),
    "progress": (212, 184, 0),
    "decoration": (156, 144, 120),
    "illustration": (38, 166, 154),
}
DEFAULT_COLOR = (200, 200, 200)
LEGACY_PLAN = os.path.join(OUT, "data", "layer_plan.json")


def draw_plan_boxes(base_png, instances, out_png, W, H):
    """Article 2.2.1 figure: visualise every LLM bbox / positive / negative point."""
    img = imread(base_png)
    if img is None:
        return None
    fs = max(12, min(W, H) // 50)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", fs)
    except Exception:
        font = ImageFont.load_default()
    for q in instances:
        col = TYPE_COLORS.get(q.get("kind", ""), DEFAULT_COLOR)
        for h in q.get("geometry_hints", []):
            bbox = h.get("bbox_px")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            for off in range(3):
                cv2.rectangle(img, (x0 - off, y0 - off), (x1 + off, y1 + off), col, 2)
            for p in h.get("positive_points_px", []):
                cv2.circle(img, tuple(p), 8, (0, 255, 0), -1)
            for p in h.get("negative_points_px", []):
                cv2.circle(img, tuple(p), 8, (0, 0, 255), -1)
    # labels are drawn with PIL, so convert *after* the OpenCV boxes are on the image
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    for q in instances:
        h0 = (q.get("geometry_hints") or [{}])[0]
        bb = h0.get("bbox_px") or [0, 0, 0, 0]
        inst = "" if q.get("instance_count", 1) == 1 else f" #{q['instance_index'] + 1}"
        lbl = f"{q['id']}{inst} (z{q.get('z_order', 0)})"
        lb = dr.textbbox((0, 0), lbl, font=font)
        lw, lh = lb[2] - lb[0] + 8, lb[3] - lb[1] + 6
        ly = max(0, bb[1] - fs - 4)
        dr.rectangle([bb[0], ly, bb[0] + lw, ly + lh], fill=col + (180,))
        dr.text((bb[0] + 4, ly + 2), lbl, fill=(255, 255, 255), font=font)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cv2.imencode(".png", out)[1].tofile(out_png)
    return out_png


class PlanStep(PipelineStep):
    name = "2.2.1 LLM layer plan"

    def run(self, ctx):
        cfg = ctx.config
        src = cfg.src
        plan_path, how = cfg.plan_path, "cli"

        if not plan_path and os.path.exists(LEGACY_PLAN):
            try:
                legacy = load_plan(LEGACY_PLAN)
            except Exception:
                legacy = {}
            if (legacy.get("source") == os.path.basename(src)
                    and legacy.get("generated_by") != "fallback_grid"):
                plan_path, how = LEGACY_PLAN, "legacy"
                T(f"[INFO] using legacy plan {os.path.basename(plan_path)}")

        if not plan_path:
            cached, is_fallback = resolve_plan(src, PLAN_DIR)
            if cached:
                plan_path, how = cached, "cache-fallback" if is_fallback else "cache"
            else:
                # article: the LLM sees the original + the OCR-cleaned working image,
                # which only exists after TextExtractStep -> hence this ordering.
                cleaned_path = os.path.join(cfg.dest, "01_text_removed.png")
                cleaned = cleaned_path if os.path.exists(cleaned_path) else src
                if cleaned == src:
                    T("[WARN] no OCR-cleaned image found, sending the original twice")
                try:
                    key = cfg.api_key or load_api_key()
                except RuntimeError as e:
                    raise RuntimeError(
                        f"{e}\nNo cached plan for {os.path.basename(src)} either. "
                        f"Run _tools/gen_llm_plan.py --src ... first, or pass --plan.")
                plan_path = generate_llm_plan(
                    src, cleaned, ctx.text_pipe.entries, ctx.H, ctx.W,
                    cfg.llm_model, api_key=key)
                how = f"generated:{cfg.llm_model}"

        plan_full = load_plan(plan_path)
        queries = plan_full.get("queries") or []
        convert_coords(queries, ctx.H, ctx.W)
        instances = expand_instances(queries)

        ctx.plan_full = plan_full
        ctx.plan = queries
        ctx.instances = instances
        ctx.plan_path = plan_path
        ctx.plan_source = how
        ctx.bg_repair = plan_full.get("background_repair") or {"mode": "none"}
        ctx.raster_text_ids = plan_full.get("raster_text_ids") or []
        ctx.corrections = corrections_to_boxes(plan_full, ctx.H, ctx.W)

        multi = sum(1 for q in queries if len(q.get("geometry_hints") or []) > 1)
        T(f"2.2.1 plan={os.path.basename(plan_path)} ({how}) queries={len(queries)} "
          f"instances={len(instances)} multi_hint_queries={multi} "
          f"bg={ctx.bg_repair.get('mode')} raster_ids={len(ctx.raster_text_ids)} "
          f"corrections={len(ctx.corrections)}")

        draw_plan_boxes(os.path.join(cfg.dest, "01_text_removed.png"), instances,
                        os.path.join(cfg.dest, "01b_LLM_bbox.png"), ctx.W, ctx.H)
        ctx.log("2.2.1 bbox diagnostic -> 01b_LLM_bbox.png")

        by_mode = {}
        for x in instances:
            by_mode[x["element_repair_mode"]] = by_mode.get(x["element_repair_mode"], 0) + 1
        ctx.note("2.2.1 LLM plan", "done",
                 f"{os.path.basename(plan_path)} via {how}; model={cfg.llm_model}; "
                 f"queries={len(queries)} -> instances={len(instances)} "
                 f"(multi-hint expanded={multi}); repair modes={by_mode}; "
                 f"prompt={len(PLAN_PROMPT)} chars + OCR id list; "
                 f"background_repair={ctx.bg_repair}")