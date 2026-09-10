# -*- coding: utf-8 -*-
"""2.4 export + reassembly: layer PNGs, masks, manifest, reassembled image."""
import os

import cv2
import numpy as np

from .. import config as C
from ..config import imwrite, jdump
from .base import PipelineStep
from .layer_build import boxof


def checker(h, w, s=10):
    yy, xx = np.indices((h, w))
    return np.where(((yy // s) + (xx // s)) % 2 == 0, 205, 140).astype(np.uint8)[..., None].repeat(3, 2)


def build_cut_cache(plan_export, masks, holes, work, plate, dropped):
    """Crop every exported layer once (article 2.4 works from these cut-outs)."""
    hole_union = np.zeros(work.shape[:2], bool)
    for h in holes.values():
        hole_union |= h
    cache = {}
    for a in sorted(plan_export, key=lambda a: (a["z_order"], a["id"])):
        i = a["id"]
        M = masks[i]
        if M.sum() == 0 or i in dropped:
            continue
        x0, y0, x1, y1 = boxof(M)
        other = (hole_union & ~holes[i])[y0:y1, x0:x1]
        if other.any():
            bgr = np.where(other[..., None], plate[y0:y1, x0:x1], work[y0:y1, x0:x1])
        else:
            bgr = work[y0:y1, x0:x1]
        cache[i] = (bgr, (x0, y0))
    return cache, hole_union


def export_layers(plan_export, masks, holes, dropped, cut_cache, dest, repair_log,
                  dup, mismatch):
    layers_dir = os.path.join(dest, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    manifest = []
    for a in sorted(plan_export, key=lambda a: (a["z_order"], a["id"])):
        i = a["id"]
        M = masks[i]
        if M.sum() == 0 or i in dropped:
            continue
        x0, y0, x1, y1 = boxof(M)
        bgr, _ = cut_cache[i]
        M_crop = M[y0:y1, x0:x1].astype(np.uint8)
        M_eroded = cv2.erode(M_crop, np.ones((3, 3), np.uint8), iterations=1)
        alpha_full = M_crop * 255
        alpha_edge = (M_crop & ~M_eroded.astype(bool)) * C.ALPHA_EDGE
        alpha = np.maximum(alpha_full, alpha_edge).astype(np.uint8)
        fn = f"{i}.png"
        imwrite(os.path.join(layers_dir, fn), np.dstack([bgr, alpha]))
        rl = next((r for r in repair_log if r["id"] == i), None)
        manifest.append(dict(
            id=i, name=a.get("name"), kind=a["kind"], role=a.get("role"),
            family=a.get("family"), instance_count=a.get("instance_count", 1),
            file=f"layers/{fn}", z_order=a["z_order"],
            parent_query_id=a.get("parent_query_id"), children=a.get("children") or None,
            element_repair_mode=a["element_repair_mode"],
            occluded_by=a["occluded_by"], needs_repair=a["needs_repair"],
            rect=[x0, y0, x1 - x0, y1 - y0], size=[x1 - x0, y1 - y0],
            alpha_px=int(M.sum()), hole_px=int(holes[i].sum()),
            repair_method=(rl or {}).get("method"),
            aka=[k for k, v in dup.items() if v["duplicate_of"] == i] or None,
            review=(f"sprite mismatch score={mismatch[i]}" if i in mismatch else None),
            plan_bbox=a["geometry_hints"][0]["bbox_px"],
            mask_bbox=[x0, y0, x1 - 1, y1 - 1]))

    mask_dir = os.path.join(dest, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    for a in plan_export:
        i = a["id"]
        if i in dropped:
            continue
        if i in masks and masks[i].sum() > 0:
            cv2.imencode(".png", masks[i].astype(np.uint8) * 255)[1].tofile(
                os.path.join(mask_dir, i + ".png"))
    return manifest


def reassemble(plan_export, masks, dropped, cut_cache, bg, dest):
    """Article 2.4: background plate + layers back-to-front by z_order."""
    reb = bg.copy()
    for a in sorted(plan_export, key=lambda a: a["z_order"]):
        i = a["id"]
        M = masks[i]
        if not M.sum() or i in dropped:
            continue
        bgr, (x0, y0) = cut_cache[i]
        ys, xs = np.where(M)
        y0m, y1m = int(ys.min()), int(ys.max()) + 1
        x0m, x1m = int(xs.min()), int(xs.max()) + 1
        sub = M[y0m:y1m, x0m:x1m]
        reb[y0m:y1m, x0m:x1m][sub] = bgr[y0m - y0:y1m - y0, x0m - x0:x1m - x0][sub]
    imwrite(os.path.join(dest, "03_reassembly_no_text.png"), reb)
    return reb


def draw_contact_sheet(manifest, dest):
    """All extracted assets on a checker board (article, effect showcase)."""
    tiles = []
    for m in sorted(manifest, key=lambda m: (m["z_order"], m["id"])):
        p = os.path.join(dest, m["file"].replace("/", os.sep))
        rgba = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            continue
        rgb = rgba[..., :3][..., ::-1]
        al = rgba[..., 3:4].astype(np.float32) / 255
        c = (rgb * al + checker(*rgb.shape[:2]) * (1 - al)).astype(np.uint8)
        tw = C.CONTACT_TILE_W
        s = tw / max(1, c.shape[1])
        c = cv2.resize(c, (tw, max(1, int(c.shape[0] * s))))
        c = cv2.copyMakeBorder(c, 22, 3, 3, 3, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(c, f"z{m['z_order']} {m['id']}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(c)
    if not tiles:
        return None
    per = C.CONTACT_PER_ROW
    rows = [tiles[i:i + per] for i in range(0, len(tiles), per)]
    strips = []
    for r in rows:
        hh = max(t.shape[0] for t in r)
        ww = sum(t.shape[1] for t in r) + 6 * (len(r) - 1)
        strip = np.full((hh, ww, 3), 25, np.uint8)
        x = 0
        for t in r:
            strip[:t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + 6
        strips.append(strip)
    wm = max(s.shape[1] for s in strips)
    sheet = np.vstack([s if s.shape[1] == wm else
                       cv2.copyMakeBorder(s, 0, 0, 0, wm - s.shape[1], cv2.BORDER_CONSTANT,
                                          value=(25, 25, 25)) for s in strips])
    out = os.path.join(dest, "04_contact_sheet.png")
    imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out


class ExportStep(PipelineStep):
    name = "2.4 Export + reassembly"

    def run(self, ctx):
        cfg = ctx.config
        dropped = set(cfg.dropped_ids or ())
        ctx.cut_cache, ctx.hole_union = build_cut_cache(
            ctx.plan_export, ctx.masks, ctx.holes, ctx.work, ctx.plate, dropped)
        ctx.manifest = export_layers(
            ctx.plan_export, ctx.masks, ctx.holes, dropped, ctx.cut_cache, cfg.dest,
            ctx.repair_log, ctx.dup, ctx.mismatch)
        jdump(dict(source=os.path.basename(cfg.src), image_size=[ctx.W, ctx.H],
                   plan=os.path.basename(ctx.plan_path), plan_source=ctx.plan_source,
                   background_repair=ctx.bg_repair,
                   raster_text_ids=ctx.raster_text_ids,
                   dropped_ids=sorted(dropped),
                   layer_count=len(ctx.manifest), layers=ctx.manifest),
              os.path.join(cfg.dest, "manifest.json"))
        ctx.log(f"exported {len(ctx.manifest)} layers "
                f"(dropped {sorted(dropped) if dropped else 'none'})")

        ctx.reb_no_text = reassemble(ctx.plan_export, ctx.masks, dropped, ctx.cut_cache,
                                     ctx.bg, cfg.dest)
        ctx.reb = ctx.text_pipe.render(ctx.reb_no_text.copy())
        draw_contact_sheet(ctx.manifest, cfg.dest)
        ctx.note("2.4 Reassembly", "done",
                 f"background plate + {len(ctx.manifest)} layers back-to-front by z_order + "
                 f"{sum(1 for s in (ctx.styles or []) if s.get('style'))} text layers "
                 f"re-rendered with the 2.1.3 styles; assets -> 04_contact_sheet.png")