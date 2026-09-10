# -*- coding: utf-8 -*-
"""Layer construction.

Article 2.3: top-level elements can be used directly, elements below them are hollow
and need repair.  Which element hollows out which one is driven by the LLM parent
chain (``parent_query_id``) plus ``z_order``; the previous version only used z_order
and bbox overlap and exported the parent chain without using it.

Repo extensions kept from the previous version: bbox guard, surface refine and
sprite consistency (template matching for repeated instances).
"""
import collections as _col

import cv2
import numpy as np

from .. import config as C
from ..llm_planner import children_map, family_of
from .base import PipelineStep


def boxof(M):
    ys, xs = np.where(M)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def guard(a, M, H, W):
    """Repo extension: drop mask pixels far outside the planned bbox."""
    x0, y0, x1, y1 = a["geometry_hints"][0]["bbox_px"]
    pad = int(max(C.GUARD_PAD_MIN, C.GUARD_PAD_RATIO * max(x1 - x0, y1 - y0)))
    g = np.zeros_like(M)
    g[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)] = True
    out = M & g
    return out if out.sum() >= C.GUARD_KEEP_RATIO * M.sum() else M


def _fill_holes(M):
    inv = (~M).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, 4)
    border = (set(lab[0, :].tolist()) | set(lab[-1, :].tolist())
              | set(lab[:, 0].tolist()) | set(lab[:, -1].tolist()))
    out = M.copy()
    for i in range(1, n):
        if i not in border:
            out[lab == i] = True
    return out


def surface_refine(instances, masks, plate, H, W, T_fn=None):
    """Repo extension: grow flat-colour surfaces to their measured colour region."""
    platef = plate.astype(np.float32)
    refined = []
    for a in instances:
        if a["element_repair_mode"] != "surface":
            continue
        M = masks[a["id"]]
        if int(M.sum()) < C.SURFACE_REFINE_MIN:
            continue
        x0, y0, x1, y1 = a["geometry_hints"][0]["bbox_px"]
        pad = 8
        win = np.zeros((H, W), bool)
        win[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)] = True
        col = np.median(platef[M], axis=0)
        region = (np.linalg.norm(platef - col, axis=2) < C.SURFACE_REFINE_DIST) & win
        n, lab = cv2.connectedComponents(region.astype(np.uint8), 8)
        add = np.zeros((H, W), bool)
        for i in range(1, n):
            comp = (lab == i)
            if int((comp & M).sum()) > 0:
                add |= comp
        new = _fill_holes(M | add)
        if not (M.sum() < new.sum() <= C.SURFACE_REFINE_MAX_GROW * M.sum()):
            continue
        refined.append(dict(id=a["id"], before=int(M.sum()), after=int(new.sum())))
        masks[a["id"]] = new
    if refined and T_fn:
        T_fn(f"surface refine: {len(refined)} {refined}")
    return refined


def sprite_consistency(instances, masks, cand_store, segaudit, plate, H, W, T_fn=None):
    """Repo extension: make repeated sprites identical via template matching.

    This is the "recall a reference image by similarity" idea the article lists as
    future work; it is applied to repeated instances of the same family here.
    """
    ids = [x["id"] for x in instances]
    fam = _col.Counter(family_of(i) for i in ids)
    dup = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            A, B = masks[ids[i]], masks[ids[j]]
            inter = int((A & B).sum())
            if inter < C.SPRITE_DUP_MIN_PX:
                continue
            iou = inter / max(1, int((A | B).sum()))
            if iou <= C.SPRITE_DUP_IOU:
                continue
            pa, pb = fam[family_of(ids[i])], fam[family_of(ids[j])]
            if pa != pb:
                loser = ids[j] if pa > pb else ids[i]
            else:
                loser = ids[j] if A.sum() >= B.sum() else ids[i]
            winner = ids[i] if loser == ids[j] else ids[j]
            dup[loser] = dict(duplicate_of=winner, iou=round(iou, 3))
    if dup and T_fn:
        T_fn(f"layer A: {len(dup)} duplicate(s) {dup}")

    def _gkey(x):
        x0, y0, x1, y1 = x["geometry_hints"][0]["bbox_px"]
        return (x["kind"], round((x1 - x0) / float(C.SPRITE_GROUP_BUCKET)),
                round((y1 - y0) / float(C.SPRITE_GROUP_BUCKET)))

    def _shift(cm, cb, dx, dy):
        out = np.zeros((H, W), bool)
        ys, xs = np.nonzero(cm)
        nx, ny = xs + cb[0] + dx, ys + cb[1] + dy
        ok = (nx >= 0) & (nx < W) & (ny >= 0) & (ny < H)
        out[ny[ok], nx[ok]] = True
        return out

    def _tmatch(tpl_rgb, box, pad=34):
        th, tw = tpl_rgb.shape[:2]
        sx0, sy0 = max(0, box[0] - pad), max(0, box[1] - pad)
        sx1, sy1 = min(W, box[2] + 1 + pad), min(H, box[3] + 1 + pad)
        reg = plate[sy0:sy1, sx0:sx1]
        if reg.shape[0] < th or reg.shape[1] < tw:
            return None
        r = cv2.matchTemplate(reg, tpl_rgb, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(r)
        return float(mx), [sx0 + ml[0], sy0 + ml[1], sx0 + ml[0] + tw - 1, sy0 + ml[1] + th - 1]

    groups = _col.defaultdict(list)
    for x in instances:
        if x["id"] in dup or x["element_repair_mode"] != "none":
            continue
        groups[_gkey(x)].append(x)
    relocated, mismatch, dbg = [], {}, []
    for gk, mem in groups.items():
        if len(mem) < C.SPRITE_GROUP_MIN:
            continue
        pool = [(x["id"], c) for x in mem for c in cand_store.get(x["id"], [])]
        if not pool:
            dbg.append((str(gk), "-", 0.0, "no candidates"))
            continue
        ws = [c["bbox"][2] - c["bbox"][0] + 1 for _, c in pool]
        hs = [c["bbox"][3] - c["bbox"][1] + 1 for _, c in pool]
        med_w, med_h = float(np.median(ws)), float(np.median(hs))
        best = None
        for mid, c in pool:
            w = c["bbox"][2] - c["bbox"][0] + 1
            h = c["bbox"][3] - c["bbox"][1] + 1
            tpl = plate[c["bbox"][1]:c["bbox"][3] + 1, c["bbox"][0]:c["bbox"][2] + 1]
            ss = []
            for x in mem:
                if x["id"] == mid:
                    continue
                r = _tmatch(tpl, segaudit[x["id"]]["mask_bbox"])
                if r:
                    ss.append(r[0])
            good = [s for s in ss if s >= C.SPRITE_MATCH_SCORE]
            need = max(2, int(np.ceil(0.6 * (len(mem) - 1))))
            if len(good) < need:
                continue
            avg = float(np.mean(good))
            if avg < C.SPRITE_MATCH_SCORE:
                continue
            sizefit = abs(w - med_w) / med_w + abs(h - med_h) / med_h
            key = (round(sizefit, 3), round(-avg, 4))
            if best is None or key < best[0]:
                best = (key, avg, mid, c)
        if best is None:
            dbg.append((str(gk), "-", 0.0, f"no tmpl {med_w:.0f}x{med_h:.0f}"))
            continue
        _, avg, mid, tc = best
        tpl = plate[tc["bbox"][1]:tc["bbox"][3] + 1, tc["bbox"][0]:tc["bbox"][2] + 1]
        if T_fn:
            T_fn(f"2.2.5 group {gk}: tmpl={mid} avg={avg:.3f} n={len(mem)}")
        for x in mem:
            cb = segaudit[x["id"]]["mask_bbox"]
            r = _tmatch(tpl, cb)
            if r is None:
                dbg.append((x["id"], mid, 0.0, "too small"))
                continue
            s, nb = r
            if s < C.SPRITE_MATCH_SCORE:
                mismatch[x["id"]] = round(s, 3)
                dbg.append((x["id"], mid, round(s, 3), f"score<{C.SPRITE_MATCH_SCORE}"))
                continue
            dx, dy = nb[0] - tc["bbox"][0], nb[1] - tc["bbox"][1]
            same = (x["id"] == mid and abs(dx) <= 2 and abs(dy) <= 2)
            M2 = _shift(tc["crop"], tc["bbox"], dx, dy)
            if M2.sum() < C.CC_MIN_AREA:
                dbg.append((x["id"], mid, round(s, 3), "too small"))
                continue
            old_area = int(masks[x["id"]].sum())
            if same and abs(old_area - int(M2.sum())) <= max(2, 0.01 * old_area):
                continue
            masks[x["id"]] = M2
            segaudit[x["id"]].update(relocated_from=cb, relocated_to=nb, tmpl=mid,
                                     tmpl_score=round(s, 3), mask_area=int(M2.sum()),
                                     mask_bbox=nb)
            relocated.append(dict(id=x["id"], group="/".join(map(str, gk)), template=mid,
                                  score=round(s, 3), before=cb, after=nb,
                                  area_before=old_area, area_after=int(M2.sum())))
        ta = int(tc["crop"].sum())
        for x in mem:
            cur = int(masks[x["id"]].sum())
            if abs(cur - ta) <= 0.10 * ta:
                continue
            cb = segaudit[x["id"]]["mask_bbox"]
            r = _tmatch(tpl, cb)
            if r is None or r[0] < C.SPRITE_ENFORCE_SCORE:
                dbg.append((x["id"], mid, round(r[0], 3) if r else 0.0,
                            f"enforce skip {cur} vs {ta}"))
                continue
            s, nb = r
            M2 = _shift(tc["crop"], tc["bbox"], nb[0] - tc["bbox"][0], nb[1] - tc["bbox"][1])
            if M2.sum() < C.CC_MIN_AREA:
                continue
            masks[x["id"]] = M2
            segaudit[x["id"]].update(relocated_from=cb, relocated_to=nb, tmpl=mid,
                                     tmpl_score=round(s, 3), mask_area=int(M2.sum()),
                                     mask_bbox=nb)
            relocated.append(dict(id=x["id"], group="/".join(map(str, gk)), template=mid,
                                  score=round(s, 3), before=cb, after=nb,
                                  area_before=cur, area_after=int(M2.sum()), stage="enforce"))
    for loser, info in list(dup.items()):
        w = info["duplicate_of"]
        if (w in mismatch and mismatch[w] < 0.5
                and mismatch.get(loser, 1.0) > mismatch[w]):
            dup[w] = dict(duplicate_of=loser, iou=info["iou"], swapped_for_naming=True)
            del dup[loser]
            if T_fn:
                T_fn(f"2.2.5 naming swap: keep {loser}")
            break
    if dbg and T_fn:
        for d in dbg:
            T_fn(f"      skip {d[0]:<14} tmpl={d[1]:<12} score={d[2]}  {d[3]}")
    if relocated and T_fn:
        T_fn(f"2.2.5 sprite relocate: {len(relocated)}")
    return dup, relocated, mismatch


def peel_order(active, masks_pre, H, W):
    """Pixel ownership between overlapping masks: nearest bbox centre wins.

    Repo extension.  It decides which pixels a layer actually keeps once its
    neighbours are peeled away, and therefore which pixels become repair holes.
    """
    best_d = np.full((H, W), np.inf, np.float32)
    best_k = np.full((H, W), -1, np.int32)
    for k, x in enumerate(active):
        M = masks_pre[x["id"]]
        if not M.any():
            continue
        x0, y0, x1, y1 = x["geometry_hints"][0]["bbox_px"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dd = np.full((H, W), np.inf, np.float32)
        ys, xs = np.nonzero(M)
        dd[ys, xs] = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        upd = M & (dd < best_d)
        best_d[upd] = dd[upd]
        best_k[upd] = k
        tie = M & (dd == best_d) & (best_k != k)
        if tie.any():
            best_k[tie] = k
    owner_z = np.full((H, W), -99, np.int16)
    for k, x in enumerate(active):
        owner_z[best_k == k] = x["z_order"]
    return best_k, owner_z


def build_occlusion(active, masks, byid, H, W):
    """Who hides whom.  Children (parent_query_id) always count as occluders."""
    kids = children_map(active)
    for x in active:
        ax0, ay0, ax1, ay1 = x["geometry_hints"][0]["bbox_px"]
        mine = masks.get(x["id"])
        lst = []
        # 1. the LLM parent chain: everything this element carries
        for cid in kids.get(x.get("query_id") or x["id"], []):
            if cid == x["id"] or cid not in byid:
                continue
            lst.append(cid)
        # 2. geometric occluders above it in z
        for y in active:
            yid = y["id"]
            if yid == x["id"] or yid in lst or y["z_order"] <= x["z_order"]:
                continue
            bx0, by0, bx1, by1 = y["geometry_hints"][0]["bbox_px"]
            ix = max(0, min(ax1, bx1) - max(ax0, bx0))
            iy = max(0, min(ay1, by1) - max(ay0, by0))
            touch = 0
            other = masks.get(yid)
            if mine is not None and other is not None and other.any():
                dil = cv2.dilate(other.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                touch = int((dil & mine).sum())
            if ix * iy > 100 or touch > 20:
                lst.append(yid)
        x["children"] = [c for c in kids.get(x.get("query_id") or x["id"], []) if c != x["id"]]
        x["occluded_by"] = lst
        x["needs_repair"] = bool(lst) and x["element_repair_mode"] != "none"


def build_alpha_and_holes(active, masks_pre, best_k, owner_z, byid, masks, H, W, textmask):
    """Compute the exported alpha and the repair holes of every layer."""
    holes, deover = {}, []
    for k, x in enumerate(active):
        i = x["id"]
        pre = masks_pre[i]
        post = (best_k == k)
        lost = pre & ~post
        child_lost = lost & (owner_z > x["z_order"])
        sib_lost = lost & (owner_z == x["z_order"])
        alpha = pre & ~sib_lost
        guard_hit = alpha.sum() < 0.30 * pre.sum()
        if guard_hit:
            alpha = pre
            sib_lost = np.zeros_like(sib_lost)
        masks[i] = alpha

        # the parent chain decides what has to be reconstructed behind this element
        child_set = set(x.get("children") or [])
        ax0, ay0, ax1, ay1 = x["geometry_hints"][0]["bbox_px"]
        comp = np.zeros((H, W), bool)
        for cid in x["occluded_by"]:
            cm = masks_pre.get(cid)
            if cm is None or not cm.any():
                continue
            y = byid.get(cid)
            if y is None:
                continue
            b = y["geometry_hints"][0]["bbox_px"]
            ix = max(0, min(ax1, b[2]) - max(ax0, b[0]))
            iy = max(0, min(ay1, b[3]) - max(ay0, b[1]))
            containment = ix * iy / max(1, (b[2] - b[0]) * (b[3] - b[1]))
            if cid in child_set or containment >= 0.50:
                comp |= cm
        alpha = alpha | (comp & ~sib_lost)
        masks[i] = alpha
        hl = (cv2.dilate((child_lost | (comp & ~sib_lost)).astype(np.uint8),
                         np.ones(C.HOLE_DILATE_KERNEL, np.uint8)) > 0) & alpha
        if x["element_repair_mode"] == "surface":
            hl |= (textmask > 0) & alpha
        holes[i] = hl
        if int(sib_lost.sum()) or int(child_lost.sum()) or guard_hit:
            deover.append(dict(id=i, z=x["z_order"], pre_px=int(pre.sum()),
                               alpha_px=int(alpha.sum()),
                               dropped_sibling_px=int(sib_lost.sum()),
                               hollowed_by_child_px=int(child_lost.sum()),
                               reverted=bool(guard_hit)))
    return holes, deover


def clear_duplicates(dup, masks, holes, H, W):
    for d in dup:
        masks[d] = np.zeros((H, W), bool)
        holes[d] = np.zeros((H, W), bool)


class LayerBuildStep(PipelineStep):
    name = "Layer build (occlusion / alpha / holes)"

    def run(self, ctx):
        cfg = ctx.config
        active = [x for x in ctx.instances if x["id"] not in ctx.dup]
        ctx.plan_export = active
        byid = {x["id"]: x for x in ctx.instances}
        build_occlusion(active, ctx.masks, byid, ctx.H, ctx.W)
        best_k, owner_z = peel_order(active, ctx.masks_pre, ctx.H, ctx.W)
        ctx.holes, ctx.deover = build_alpha_and_holes(
            active, ctx.masks_pre, best_k, owner_z, byid, ctx.masks,
            ctx.H, ctx.W, ctx.textmask)
        clear_duplicates(ctx.dup, ctx.masks, ctx.holes, ctx.H, ctx.W)
        ctx.allmask = np.zeros((ctx.H, ctx.W), bool)
        for a in active:
            ctx.allmask |= ctx.masks[a["id"]]
        added = self._add_safe_area(ctx)
        repaired = sum(1 for a in active if a["needs_repair"])
        with_children = sum(1 for a in active if a.get("children"))
        ctx.log(f"layer B/C: {len(ctx.deover)} adjusted, {repaired} need repair, "
                f"{with_children} have children; "
                f"hole_px(sum over layers)="
                f"{sum(int(ctx.holes[x['id']].sum()) for x in active)}")
        ctx.note("Layer build", "done",
                 f"parent_query_id chain drives occlusion repair ({with_children} parents, "
                 f"{repaired}/{len(active)} layers need repair); z_order peel + nearest-centre "
                 f"pixel ownership; hole dilate {C.HOLE_DILATE_KERNEL}; "
                 f"duplicates cleared={len(ctx.dup)}")
        if added:
            ctx.note("safe area (repo extension)", "done",
                     f"synthetic status_bar_safe_area layer, top {added}px")
        return added

    def _add_safe_area(self, ctx):
        h = int(ctx.config.safe_area_height or 0)
        if h <= 0:
            return 0
        safe = np.zeros((ctx.H, ctx.W), bool)
        safe[:min(h, ctx.H), :] = True
        safe &= ~ctx.allmask
        if safe.sum() <= 1000:
            return 0
        ctx.plan_export.append(dict(
            id="status_bar_safe_area", name="status bar safe area", query_id="status_bar_safe_area",
            family="status_bar_safe_area", instance_index=0, instance_count=1,
            kind="decoration", role="background", element_repair_mode="none",
            parent_query_id=None, children=[], z_order=0,
            geometry_hints=[dict(bbox_px=[0, 0, ctx.W, min(h, ctx.H)],
                                 positive_points_px=[[ctx.W // 2, min(h, ctx.H) // 2]],
                                 negative_points_px=[])],
            occluded_by=[], needs_repair=False))
        ctx.masks["status_bar_safe_area"] = safe
        ctx.masks_pre["status_bar_safe_area"] = safe.copy()
        ctx.holes["status_bar_safe_area"] = np.zeros((ctx.H, ctx.W), bool)
        ctx.allmask |= safe
        return min(h, ctx.H)