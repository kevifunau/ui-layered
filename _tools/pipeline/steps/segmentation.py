# -*- coding: utf-8 -*-
"""2.2.2 SAM segmentation, 2.2.3 candidate scoring, 2.2.4 closing + component filter."""
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from .. import config as C
from ..config import imread, imwrite
from .base import PipelineStep
from .layer_build import guard, sprite_consistency, surface_refine

CAND_COLORS = ((80, 200, 255), (120, 120, 240), (200, 120, 120))


def segment(h, proc, model, emb, img):
    """One prompted SAM inference.  multimask_output=True -> the 3 candidates of 2.2.3."""
    bb = h["bbox_px"]
    pos = h["positive_points_px"]
    neg = h["negative_points_px"]
    kw = dict(input_boxes=[[[bb]]])
    if pos or neg:
        kw["input_points"] = [[pos + neg]]
        kw["input_labels"] = [[[1] * len(pos) + [0] * len(neg)]]
    inp = proc(images=img, return_tensors="pt", **kw)
    inp = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.no_grad():
        o = model(image_embeddings=emb, input_boxes=inp["input_boxes"],
                  input_points=inp.get("input_points"), input_labels=inp.get("input_labels"),
                  original_sizes=inp["original_sizes"],
                  reshaped_input_sizes=inp["reshaped_input_sizes"],
                  multimask_output=True)
    m = proc.image_processor.post_process_masks(
        o.pred_masks, inp["original_sizes"], inp["reshaped_input_sizes"])
    t = m[0] if isinstance(m, (list, tuple)) else m
    while t.dim() > 3:
        t = t[0]
    return t.cpu().numpy(), o.iou_scores[0, 0].cpu().numpy()


def _at(c, p):
    y, x = int(p[1]), int(p[0])
    y = min(max(y, 0), c.shape[0] - 1)
    x = min(max(x, 0), c.shape[1] - 1)
    return bool(c[y, x])


def cand_score(c, iou_s, h):
    """Article 2.2.3: SAM + 0.22*pos - 0.38*neg + 0.10*containment + 0.06*bbox IoU."""
    bb = np.array(h["bbox_px"])
    pos, neg = h["positive_points_px"], h["negative_points_px"]
    ph = np.mean([_at(c, p) for p in pos]) if pos else 1.0
    nh = np.mean([_at(c, p) for p in neg]) if neg else 0.0
    total = int(c.sum())
    contain = int(c[bb[1]:bb[3], bb[0]:bb[2]].sum()) / max(1, total)
    ys, xs = np.where(c)
    mb = np.array([xs.min(), ys.min(), xs.max(), ys.max()]) if len(xs) else bb
    ix0, iy0 = max(mb[0], bb[0]), max(mb[1], bb[1])
    ix1, iy1 = min(mb[2], bb[2]), min(mb[3], bb[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (mb[2] - mb[0]) * (mb[3] - mb[1]) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
    bbox_iou = inter / max(1, union)
    score = (float(iou_s) + C.CAND_W_POS * float(ph) + C.CAND_W_NEG * float(nh)
             + C.CAND_W_CONTAIN * float(contain) + C.CAND_W_IOU * float(bbox_iou))
    return score, dict(pos_hit=round(float(ph), 4), neg_hit=round(float(nh), 4),
                       containment=round(float(contain), 4), bbox_iou=round(float(bbox_iou), 4))


def cc_filter(c, pos, keep_ratio=None):
    """Article 2.2.4: 3x3 closing then three component rules.

    1. every component containing a positive point is kept;
    2. if no component contains a positive point, keep the largest one;
    3. additionally keep every component whose area is >= 8% of the largest.
    Rule 3 was missing in the previous version even though the audit claimed 3 rules.
    """
    ratio = C.CC_KEEP_AREA_RATIO if keep_ratio is None else keep_ratio
    m = cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_CLOSE,
                         np.ones(C.CC_KERNEL, np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    mx = int(areas.max())
    keep_lab = np.zeros(n, bool)
    hit = set()
    for p in pos or []:
        y, x = int(p[1]), int(p[0])
        if 0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]:
            v = int(lab[y, x])
            if v > 0:
                hit.add(v)
    for v in hit:
        keep_lab[v] = True                       # rule 1
    if not hit:
        keep_lab[1 + int(np.argmax(areas))] = True   # rule 2
    if ratio > 0 and mx > 0:
        big = np.nonzero(areas >= ratio * mx)[0] + 1
        keep_lab[big] = True                     # rule 3
    return keep_lab[lab]


def run_segmentation(instances, proc, model, emb, img, keep_ratio=None, T_fn=None,
                     viz_budget=None):
    """Segment every instance.  Returns (masks, segaudit, cand_store, viz)."""
    masks, segaudit, cand_store, viz = {}, {}, {}, {}
    budget = C.SEG_CAND_VIZ_LIMIT if viz_budget is None else viz_budget
    t0 = time.time()
    for a in instances:
        h = a["geometry_hints"][0]
        ms, iou = segment(h, proc, model, emb, img)
        cands = [ms[k] > 0 for k in range(ms.shape[0])]
        scored = [cand_score(c, float(iou[k]), h) for k, c in enumerate(cands)]
        sc = [s for s, _ in scored]
        b = int(np.argmax(sc))
        srt = sorted(sc, reverse=True)
        margin = round(srt[0] - srt[1], 4) if len(srt) > 1 else None

        if a["element_repair_mode"] == "none":
            cs = []
            for ci, c in enumerate(cands):
                cc = cc_filter(c, h["positive_points_px"], keep_ratio)
                if cc.sum() < C.CC_MIN_AREA:
                    continue
                yy, xx = np.where(cc)
                bb = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
                cs.append(dict(k=ci, crop=cc[bb[1]:bb[3] + 1, bb[0]:bb[2] + 1], bbox=bb,
                               area=int(cc.sum())))
            cand_store[a["id"]] = cs

        masks[a["id"]] = cc_filter(cands[b], h["positive_points_px"], keep_ratio)
        ys, xs = np.where(masks[a["id"]])
        segaudit[a["id"]] = dict(
            pick=b, scores=[round(v, 4) for v in sc],
            ious=[round(float(v), 4) for v in iou],
            parts=[d for _, d in scored],
            margin=margin,
            candidates=[dict(k=k, area=int(cands[k].sum()),
                             bbox=[int(v) for v in _bbox_of(cands[k])] if cands[k].any() else None)
                        for k in range(len(cands))],
            plan_bbox=h["bbox_px"],
            mask_bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            mask_area=int(masks[a["id"]].sum()))
        if margin is not None and margin < C.AMBIGUOUS_MARGIN and len(viz) < budget:
            viz[a["id"]] = [c.copy() for c in cands]
    if T_fn:
        T_fn(f"2.2.2/2.2.3/2.2.4 segmented {len(masks)} instances in {time.time() - t0:.1f}s")
    return masks, segaudit, cand_store, viz


def _bbox_of(c):
    ys, xs = np.where(c)
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def draw_candidate_viz(base_png, viz, segaudit, out_png):
    """Article 2.2.3 figure: the three SAM candidates of the ambiguous instances."""
    img = imread(base_png)
    if img is None or not viz:
        return None
    for i, (cands) in viz.items():
        pick = segaudit[i]["pick"]
        for k, c in enumerate(cands):
            if not c.any():
                continue
            col = CAND_COLORS[k % len(CAND_COLORS)]
            thick = 3 if k == pick else 1
            cnts, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, thick)
        bb = segaudit[i]["plan_bbox"]
        sc = "/".join(f"{v:.2f}" for v in segaudit[i]["scores"])
        cv2.putText(img, f"{i} pick={pick} {sc}", (bb[0], max(12, bb[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{i} pick={pick} {sc}", (bb[0], max(12, bb[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    imwrite(out_png, img)
    return out_png


class SegmentationStep(PipelineStep):
    name = "2.2.2/2.2.3/2.2.4 segmentation"

    def run(self, ctx):
        cfg = ctx.config
        img = Image.fromarray(cv2.cvtColor(ctx.plate, cv2.COLOR_BGR2RGB))
        proc = SamProcessor.from_pretrained(cfg.samdir)
        model = SamModel.from_pretrained(cfg.samdir).to("cuda").eval()
        t0 = time.time()
        with torch.no_grad():
            emb = model.get_image_embeddings(
                proc(images=img, return_tensors="pt")["pixel_values"].to("cuda"))
        encode_s = time.time() - t0
        ctx.log(f"2.2.2 SAM ViT-B encoded {ctx.W}x{ctx.H} once in {encode_s:.1f}s")

        ctx.masks, ctx.segaudit, ctx.cand_store, viz = run_segmentation(
            ctx.instances, proc, model, emb, img, keep_ratio=cfg.cc_keep_ratio,
            T_fn=ctx.log)
        del model, proc, emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for a in ctx.instances:
            ctx.masks[a["id"]] = guard(a, ctx.masks[a["id"]], ctx.H, ctx.W)
        ctx.surface_refined = surface_refine(ctx.instances, ctx.masks, ctx.plate,
                                             ctx.H, ctx.W, T_fn=ctx.log)
        ctx.dup, ctx.relocated, ctx.mismatch = sprite_consistency(
            ctx.instances, ctx.masks, ctx.cand_store, ctx.segaudit, ctx.plate,
            ctx.H, ctx.W, T_fn=ctx.log)
        ctx.masks_pre = {k: v.copy() for k, v in ctx.masks.items()}

        if viz:
            draw_candidate_viz(os.path.join(cfg.dest, "01c_text_removed_final.png"), viz,
                               ctx.segaudit, os.path.join(cfg.dest, "05_seg_candidates.png"))
            ctx.log(f"2.2.3 ambiguous candidates ({len(viz)}) -> 05_seg_candidates.png")

        margins = [v["margin"] for v in ctx.segaudit.values() if v["margin"] is not None]
        ctx.note("2.2.2 SAM", "done",
                 f"SAM1 ViT-B ({os.path.basename(cfg.samdir)}), whole image encoded once "
                 f"({encode_s:.1f}s), one prompted inference per instance, multimask_output=3")
        margin_txt = "no margins"
        if margins:
            ms = sorted(margins)
            margin_txt = (f"margin min/med/max={ms[0]:.3f}/"
                          f"{ms[len(ms) // 2]:.3f}/{ms[-1]:.3f}")
        ctx.note("2.2.3 Candidate scoring", "done",
                 f"SAM + {C.CAND_W_POS}*pos_hit + ({C.CAND_W_NEG})*neg_hit + "
                 f"{C.CAND_W_CONTAIN}*containment + {C.CAND_W_IOU}*bbox_IoU; {margin_txt}")
        ctx.note("2.2.4 CC filter", "done",
                 f"3x3 closing + rule1 positive-point components + rule2 largest when no hit + "
                 f"rule3 area >= {cfg.cc_keep_ratio:.0%} of largest; "
                 f"ambiguous(margin<{C.AMBIGUOUS_MARGIN})={len(viz)}")
        ctx.note("segmentation extensions", "done",
                 f"guard(pad>={C.GUARD_PAD_MIN}, keep>={C.GUARD_KEEP_RATIO}), "
                 f"surface_refine(<{C.SURFACE_REFINE_DIST}) x{len(ctx.surface_refined)}, "
                 f"sprite_consistency(>={C.SPRITE_MATCH_SCORE}) relocated={len(ctx.relocated)} "
                 f"dup={len(ctx.dup)} mismatch={len(ctx.mismatch)}")