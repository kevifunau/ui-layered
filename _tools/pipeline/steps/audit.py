# -*- coding: utf-8 -*-
"""Audit step: lossless self-check, quality gates and audit.json.

The fidelity table is *generated from what actually ran* (every step calls
``ctx.note``), so it can no longer claim a feature that is not wired up.
"""
import json
import os
import time

import cv2
import numpy as np

from .. import config as C
from ..config import OUT, jdump
from .base import PipelineStep
from .export import draw_contact_sheet

V4_REF = os.path.join(OUT, "_tools", "v4_reference", "temp_layers_v4.json")


def self_check(reb, work, holes, plan_export, dropped, masks, allmask, H, W, T_fn=None):
    """Decomposition must be lossless: reassembly == repaired work outside the holes."""
    hole_u = np.zeros((H, W), bool)
    for a in plan_export:
        hole_u |= holes[a["id"]]
    seam_u = cv2.dilate(hole_u.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    d = np.abs(reb.astype(np.int16) - work.astype(np.int16)).max(axis=2)
    unexp_mask = (d > 0) & ~seam_u
    safe = masks.get("status_bar_safe_area")
    if safe is not None:
        unexp_mask &= ~safe
    for i in dropped:
        if i in masks:
            unexp_mask &= ~masks[i]
    unexp = int(unexp_mask.sum())
    cov = float(allmask.mean())
    if T_fn:
        T_fn(f"self-check: unexpected diff={unexp}px "
             f"(holes={int(hole_u.sum())}px, seam={int((seam_u & ~hole_u).sum())}px)")
        T_fn(f"coverage: {cov:.1%}; background fills {1 - cov:.1%}")
    return unexp, cov, int(hole_u.sum())


def v4_compare(manifest, T_fn=None):
    """Migration metric against the previous internal version (opt-in per case)."""
    cmp = []
    if not os.path.exists(V4_REF):
        return cmp
    v4 = {x["id"]: x for x in json.load(open(V4_REF, encoding="utf-8"))}
    for m in manifest:
        o = v4.get(m["id"])
        if not o:
            continue
        cmp.append(dict(id=m["id"], v4_area=o["area"], v5_area=m["alpha_px"],
                        gain=m["alpha_px"] - o["area"],
                        pct=round(100 * (m["alpha_px"] - o["area"]) / max(1, o["area"]), 1),
                        v4_bbox=o["bbox"], v5_bbox=m["mask_bbox"]))
    if cmp and T_fn:
        T_fn(f"v4 compare: {sum(1 for c in cmp if c['gain'] > 0)}/{len(cmp)} grew, "
             f"max {max(c['pct'] for c in cmp):+.1f}%")
    return cmp


def param_snapshot():
    """The article constants this run actually used."""
    skip = {"OUT", "DATA_DIR", "PLAN_DIR", "CASE_DIR", "SECRET_DIR", "SRC_DEFAULT",
            "DEST_DEFAULT", "SAMDIR", "CKPT", "BASE", "LLM_MODEL", "LLM_BASE_URL",
            "YAHEI", "YAHEI_BD", "ARIAL", "ARIAL_BD"}
    out = {}
    for k, v in vars(C).items():
        if not k.isupper() or k in skip:
            continue
        if isinstance(v, (int, float, str, bool)) or isinstance(v, tuple):
            out[k] = v
    return out


class AuditStep(PipelineStep):
    name = "Audit"

    def run(self, ctx):
        cfg = ctx.config
        dropped = set(cfg.dropped_ids or ())
        ctx.unexp, ctx.cov, hole_px = self_check(
            ctx.reb_no_text, ctx.work, ctx.holes, ctx.plan_export, dropped,
            ctx.masks, ctx.allmask, ctx.H, ctx.W, T_fn=ctx.log)
        ctx.cmp = v4_compare(ctx.manifest, T_fn=ctx.log) if cfg.case_config.get("v4_reference") else []
        if ctx.cmp is None:
            ctx.cmp = []
        draw_contact_sheet(ctx.manifest, cfg.dest)

        repair_methods = {}
        for r in ctx.repair_log or []:
            k = f"{r['mode']}/{r.get('method')}"
            repair_methods[k] = repair_methods.get(k, 0) + 1
        bg_gen = (cfg.bg == "gen"
                  or (cfg.bg == "auto" and (ctx.bg_repair or {}).get("mode") == "scene"))
        margins = [v["margin"] for v in (ctx.segaudit or {}).values()
                   if v.get("margin") is not None]

        audit = dict(
            pipeline="run_pipeline.py",
            source=os.path.basename(cfg.src),
            image_size=[ctx.W, ctx.H],
            generated=time.strftime("%Y-%m-%d %H:%M:%S"),
            plan=dict(path=os.path.basename(ctx.plan_path or ""), source=ctx.plan_source,
                      queries=len(ctx.plan or []), instances=len(ctx.instances or []),
                      raster_text_ids=ctx.raster_text_ids,
                      text_corrections=len((ctx.plan_full or {}).get("text_corrections") or []),
                      scene_summary=(ctx.plan_full or {}).get("scene_summary")),
            mode=dict(element_repair=cfg.repair, background="gen" if bg_gen else "ns",
                      gen_backend=cfg.gen_backend,
                      bg_estimate=cfg.bg_estimate, fg_core_filter=cfg.fg_core_filter,
                      cc_keep_ratio=cfg.cc_keep_ratio,
                      safe_area_height=cfg.safe_area_height,
                      dropped_ids=sorted(dropped)),
            timings=ctx.timings,
            ocr=dict(kept=len(ctx.text_pipe.entries),
                     from_rapidocr=sum(1 for e in ctx.text_pipe.entries
                                       if e.get("source") == "rapidocr"),
                     llm_corrections=sum(1 for e in ctx.text_pipe.entries
                                         if e.get("source") == "llm_text_correction"),
                     raster=sum(1 for e in ctx.text_pipe.entries
                               if e.get("render") == "raster_asset"),
                     editable=sum(1 for e in ctx.text_pipe.entries
                                  if e.get("render") == "editable_text"),
                     dropped=ctx.text_pipe.drop),
            layers=len(ctx.manifest),
            duplicates=ctx.dup, deoverlap=ctx.deover,
            sprite_relocated=ctx.relocated, sprite_mismatch=ctx.mismatch,
            surface_refined=ctx.surface_refined,
            seg_pre_area={k: int(v.sum()) for k, v in (ctx.masks_pre or {}).items()},
            seg_margin=dict(min=min(margins) if margins else None,
                            median=sorted(margins)[len(margins) // 2] if margins else None,
                            max=max(margins) if margins else None),
            coverage=round(ctx.cov, 4), hole_px=hole_px,
            lossless_check=dict(unexpected_diff_px=ctx.unexp, hole_px=hole_px,
                                verdict="PASS" if ctx.unexp == 0 else "FAIL"),
            gen_calls=ctx.model_calls or [],
            background_repair=ctx.bg_repair, bg_method=ctx.bg_method,
            background_sanity=ctx.bg_sanity,
            element_repair=ctx.atlas_report,
            repair_methods=repair_methods,
            fidelity=ctx.fidelity,
            params=param_snapshot(),
            seg_audit=ctx.segaudit,
            v4_compare=ctx.cmp,
            repair_log=ctx.repair_log)
        verdict = "PASS" if ctx.unexp == 0 else "FAIL"
        total = round(ctx.elapsed(), 2)
        ctx.timings["total"] = total          # Pipeline.run overwrites it right after
        audit["timings"] = ctx.timings
        ctx.note("Audit", "done",
                 f"lossless {verdict} (unexpected diff {ctx.unexp}px), "
                 f"coverage {ctx.cov:.1%}, {len(ctx.manifest)} layers, total {total}s")
        audit["fidelity"] = ctx.fidelity          # includes the Audit line itself
        jdump(audit, os.path.join(cfg.dest, "audit.json"))
        ctx.log(f"audit: lossless={verdict} diff={ctx.unexp} "
                f"layers={len(ctx.manifest)} total={ctx.timings.get('total')}s")