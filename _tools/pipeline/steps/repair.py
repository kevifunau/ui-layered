# -*- coding: utf-8 -*-
"""2.3 occlusion repair.

Article 2.3: elements that are not on top are hollow and must be repaired.  The LLM
decision ``element_repair_mode`` picks the path:

    "image"   -> 2.3.1 pack the elements into an atlas and let the image model repair it
    "surface" -> 2.3.2 traditional fill measured from the surrounding pixels
    "none"    -> nothing to repair (topmost foreground object)

2.3.3 repairs the background according to ``background_repair.mode``.

``--repair auto`` (default) follows the LLM decision, ``ns`` forces the traditional
path for everything, ``gen`` forces the image model.  Generative repair always falls
back to the traditional path per element when ComfyUI is unreachable or the result
fails the black-pixel sanity gate.
"""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np

from .. import config as C
from ..config import imread, imwrite
from ..prompts import ATLAS_PROMPT, ATLAS_PROMPT_SINGLE, BG_PROMPT, NEG_PROMPT
from ..providers import InpaintProvider, get_provider
from .base import PipelineStep
from .layer_build import boxof


# ------------------------------------------------------------ ComfyUI plumbing
def enc_png(im):
    return cv2.imencode(".png", im)[1].tobytes()


def _http_json(url, data=None, headers=None, timeout=120, what=""):
    """urlopen + JSON.  An HTTP error keeps the response body: a bare "400 Bad
    Request" from ComfyUI is otherwise impossible to diagnose."""
    r = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        return json.load(urllib.request.urlopen(r, timeout=timeout))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"ComfyUI {what or url} -> HTTP {e.code}: {body}") from None


# The multipart MIME type is assembled at runtime on purpose: some editors and
# transfer layers silently corrupt that literal, which makes ComfyUI answer 400.
MIME_MULTIPART = "multi" + "part/" + "form" + chr(45) + "data"


def upload(name, buf, base):
    bd = uuid.uuid4().hex
    head = ("\r\nContent-Disposition: form" + chr(45) + "data; name=\"image\"; "
            "filename=\"" + name + "\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body = b"--" + bd.encode() + head + buf + b"\r\n--" + bd.encode() + b"--\r\n"
    out = _http_json(base + "/upload/image", data=body,
                     headers={"Content-Type": f"{MIME_MULTIPART}; boundary={bd}"},
                     what=f"/upload/image ({name}, {len(buf)} bytes)")
    return out["name"]


def comfy_run(wf, base, timeout=C.COMFY_TIMEOUT):
    pid = _http_json(base + "/prompt", data=json.dumps({"prompt": wf}).encode(),
                     headers={"Content-Type": "application/json"},
                     what="/prompt")["prompt_id"]
    t0, hist = time.time(), {}
    while time.time() - t0 < timeout:
        hist = _http_json(base + "/history/" + pid, timeout=60, what="/history")
        if pid in hist:
            break
        time.sleep(2)
    if pid not in hist:
        raise RuntimeError(f"ComfyUI timeout after {timeout}s")
    st = hist[pid]["status"]
    if st.get("status_str") != "success":
        raise RuntimeError(str(st.get("messages"))[:400])
    return hist[pid]["outputs"]


def fetch(outputs, base):
    im = outputs["save"]["images"][0]
    url = (f"{base}/view?filename={im['filename']}"
           f"&subfolder={im.get('subfolder', '')}&type={im['type']}")
    d = urllib.request.urlopen(url, timeout=120).read()
    return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)


def inpaint_workflow(img, mask, ckpt, prompt, neg, seed, prefix, base):
    """VAE inpaint with SetLatentNoiseMask: only the white mask area is regenerated."""
    return {
        "load": {"class_type": "LoadImage",
                 "inputs": {"image": upload(prefix + ".png", enc_png(img), base)}},
        "mload": {"class_type": "LoadImage",
                  "inputs": {"image": upload(prefix + "_mask.png",
                                             enc_png(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
                                             base)}},
        "tomask": {"class_type": "ImageToMask", "inputs": {"image": ["mload", 0], "channel": "red"}},
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "enc": {"class_type": "VAEEncode", "inputs": {"pixels": ["load", 0], "vae": ["ckpt", 2]}},
        "nm": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["enc", 0], "mask": ["tomask", 0]}},
        "txt": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["ckpt", 1]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["ckpt", 1]}},
        "ks": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": C.FLUX_STEPS, "cfg": C.FLUX_CFG,
            "sampler_name": C.FLUX_SAMPLER, "scheduler": C.FLUX_SCHEDULER,
            "denoise": C.FLUX_DENOISE, "model": ["ckpt", 0],
            "positive": ["txt", 0], "negative": ["neg", 0], "latent_image": ["nm", 0]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0],
                                                       "filename_prefix": prefix}},
    }


def _hole_stats(img, hole_mask):
    """(pure-black ratio, standard deviation) of the generated hole region."""
    if not hole_mask.any():
        return 0.0, 0.0
    n = int(hole_mask.sum())
    black = float(((img.max(axis=2) < 8) & hole_mask).sum()) / n
    return black, float(img[hole_mask].std())


def _material_delta(img, hole_mask):
    """Median colour distance between the generated hole and the element's own
    visible material.  The article requires the repair to "seamlessly continue the
    surrounding background material of the same element"; a model that paints a
    near-uniform panel dark blue fails that requirement even when it adds texture.
    """
    ref = (~hole_mask) & (img.max(axis=2) > 8)
    if not ref.any() or not hole_mask.any():
        return 0.0
    a = np.median(img[hole_mask], axis=0)
    b = np.median(img[ref], axis=0)
    return float(np.linalg.norm(a - b))


def _sanity_ok(img, hole_mask, max_black, min_std=C.GEN_MIN_STD,
               max_delta=C.ATLAS_MAX_COLOR_DELTA, strict=False):
    """Accept a generation when the hole really was filled with the same material.

    * pure-black ratio <= max_black            -> filled, nothing left behind;
    * otherwise the region must carry texture (std >= min_std) *and* its median
      colour must stay close to the element's visible material (delta <= max_delta),
      which rejects "the model invented unrelated dark content" while still
      accepting legitimately dark materials.
    """
    black, std = _hole_stats(img, hole_mask)
    delta = _material_delta(img, hole_mask)
    if strict:
        ok = black <= max_black
    else:
        ok = black <= max_black or (std >= min_std and delta <= max_delta)
    return ok, black, std, delta


# ------------------------------------------------------- 2.3.2 traditional fill
def surface_fill(bgr, plate, M, hl):
    """Fill a hole with the measured colour of the same element (article 2.3.2).

    The material is measured on the OCR-cleaned original (``plate``), never on the
    progressive canvas: once a parent has been repaired, the canvas no longer holds
    this element's own pixels.  Holes are excluded from the measurement so a parent
    is not measured through its children.
    """
    if hl.sum() == 0:
        return bgr, 0, None
    x0, y0, x1, y1 = boxof(M)
    md = max(3, int(0.15 * min(x1 - x0, y1 - y0)))
    core = M.copy()
    core[:y0 + md, :] = False
    core[y1 - md:, :] = False
    core[:, :x0 + md] = False
    core[:, x1 - md:] = False
    core &= ~hl
    if core.sum() < 50:
        core = M & ~hl
    if core.sum() < 20:
        core = M
    pts = plate[core].astype(np.float32)
    if len(pts) >= 100:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, lab, cent = cv2.kmeans(pts, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        cnt = np.bincount(lab.ravel(), minlength=2)
        med = cent[int(np.argmax(cnt))]
    else:
        med = np.median(pts, axis=0) if len(pts) else np.median(bgr[M], axis=0)
    out = bgr.copy()
    out[hl] = med
    seam = (cv2.dilate(hl.astype(np.uint8), np.ones(C.SURFACE_SEAM_KERNEL, np.uint8)) > 0) & ~hl & M
    out[seam] = (0.5 * out[seam] + 0.5 * med).astype(np.uint8)
    return out, int(hl.sum()), [int(v) for v in med]


def repair_surface(plan_export, masks, holes, work, plate, T_fn=None):
    log = []
    todo = [a for a in plan_export if a["element_repair_mode"] == "surface"
            and holes[a["id"]].sum() > 0]
    for a in sorted(todo, key=lambda a: a["z_order"]):
        work, npx, med = surface_fill(work, plate, masks[a["id"]], holes[a["id"]])
        log.append(dict(id=a["id"], mode="surface", method="surface-fill",
                        hole_px=npx, fill=med))
    if T_fn:
        T_fn(f"2.3.2 surface repair: {len(todo)} element(s)")
    return work, log


# ------------------------------------- 2.3.1 traditional path for "image" parts
def _ns_fill_region(work, plate, M, hl, W, H):
    """Median prefill + NS inpaint of the hole rim (article 2.3.2 style)."""
    x0, y0, x1, y1 = boxof(M)
    pad = 10
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
    hm = hl[cy0:cy1, cx0:cx1]
    Mc = M[cy0:cy1, cx0:cx1]
    reg = work[cy0:cy1, cx0:cx1]
    base = plate[cy0:cy1, cx0:cx1][Mc & ~hm].astype(np.float32)
    med = np.median(base, axis=0) if len(base) > 50 else np.array([80., 80., 80.])
    reg[hm] = med.astype(np.uint8)
    if hm.any():
        rim = cv2.dilate(hm.astype(np.uint8), np.ones(C.NS_BORDER_DILATE, np.uint8)) > 0
        rim &= ~cv2.erode(hm.astype(np.uint8), np.ones(C.NS_BORDER_ERODE, np.uint8)).astype(bool)
        rim &= Mc
        if rim.any():
            fixed = cv2.inpaint(reg, rim.astype(np.uint8) * 255, C.NS_BORDER_RADIUS, cv2.INPAINT_NS)
            reg[rim] = fixed[rim]
    return int(hm.sum()), [int(v) for v in med]


def repair_image_ns(todo, masks, holes, work, plate, W, H, T_fn=None, note="ns"):
    log = []
    for a in todo:
        npx, med = _ns_fill_region(work, plate, masks[a["id"]], holes[a["id"]], W, H)
        log.append(dict(id=a["id"], mode="image", method=note, hole_px=npx, fill=med))
    if T_fn:
        T_fn(f"2.3.1 image repair ({note}): {len(todo)} element(s)")
    return work, log


# --------------------------------------- 2.3.1 generative path (atlas + model)
def _chunk(items, budget, area=None):
    """Split into atlas-sized batches; ``area`` defaults to the packed crop size."""
    area = area or (lambda it: it["bgr"].shape[0] * it["bgr"].shape[1])
    out, cur, acc = [], [], 0
    for it in items:
        a = area(it)
        if cur and acc + a > budget:
            out.append(cur)
            cur, acc = [], 0
        cur.append(it)
        acc += a
    if cur:
        out.append(cur)
    return out


def pack_atlas(items, max_side=C.ATLAS_MAX_SIDE, pad=C.ATLAS_PAD):
    """Greedy shelf packing with a global downscale so the atlas fits ``max_side``."""
    scale = 1.0
    for _ in range(8):
        cells = []
        for it in items:
            oh, ow = it["bgr"].shape[:2]
            sw = max(8, int(round(ow * scale)))
            sh = max(8, int(round(oh * scale)))
            sw -= sw % 2
            sh -= sh % 2
            cells.append(dict(id=it["id"], w=sw, h=sh, ow=ow, oh=oh))
        order = sorted(range(len(cells)), key=lambda i: -cells[i]["h"])
        x = y = row_h = 0
        for i in order:
            c = cells[i]
            if x > 0 and x + c["w"] > max_side:
                x, y = 0, y + row_h + pad
                row_h = 0
            c["x"], c["y"] = x, y
            x += c["w"] + pad
            row_h = max(row_h, c["h"])
        W_a = max([c["x"] + c["w"] for c in cells] + [8])
        H_a = max(y + row_h, 8)
        W_a += (-W_a) % 8
        H_a += (-H_a) % 8
        if max(W_a, H_a) <= max_side or scale <= 0.15:
            break
        scale *= max_side / float(max(W_a, H_a))
    atlas = np.zeros((H_a, W_a, 3), np.uint8)
    amask = np.zeros((H_a, W_a), np.uint8)
    for it, c in zip(items, cells):
        oh, ow = it["bgr"].shape[:2]
        if (ow, oh) != (c["w"], c["h"]):
            # cells are rounded down to an even size for the VAE, so resize even at scale 1
            interp = cv2.INTER_AREA if c["w"] < ow else cv2.INTER_LANCZOS4
            bgr = cv2.resize(it["bgr"], (c["w"], c["h"]), interpolation=interp)
            msk = cv2.resize(it["mask"].astype(np.uint8) * 255, (c["w"], c["h"]),
                             interpolation=cv2.INTER_NEAREST) > 127
        else:
            bgr, msk = it["bgr"], it["mask"]
        atlas[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] = bgr
        amask[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] = msk.astype(np.uint8) * 255
        c["scale"] = round(scale, 4)
    return atlas, amask, cells


def collect_atlas_items(todo, masks, holes, work, plate, W, H, prefill=True):
    """Crop every element that needs image-model repair.

    ``prefill`` seeds the hole with the element's own measured material colour
    before handing the atlas to the image model.  The article sends pure black
    holes; with a local Flux fp8 that leaves 30-70% of a large hole unfilled, so
    the model is given the material it has to continue and only refines it.  The
    binary mask still marks exactly the regions that must be repainted.
    """
    items = []
    for a in sorted(todo, key=lambda a: (a["z_order"], a["id"])):
        i = a["id"]
        M, hl = masks[i], holes[i]
        if int(hl.sum()) == 0:
            continue
        x0, y0, x1, y1 = boxof(M)
        pad = 10
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
        crop = work[cy0:cy1, cx0:cx1].copy()
        hm = hl[cy0:cy1, cx0:cx1]
        med = None
        if prefill:
            pcrop = plate[cy0:cy1, cx0:cx1]
            ref = M[cy0:cy1, cx0:cx1] & ~hm & (pcrop.max(axis=2) > 8)
            if int(ref.sum()) >= 50:
                med = np.median(pcrop[ref], axis=0)
        crop[hm] = 0 if med is None else med.astype(np.uint8)
        items.append(dict(id=i, box=(cx0, cy0, cx1, cy1), bgr=crop, mask=hm.copy(),
                          hole_px=int(hl.sum()), prefilled=med is not None))
    return items


def _prov_fill(prov, img, mask, prompt, dump=None, rec=None, T_fn=None, seed=None, depth=0):
    """One provider call, split along the long axis when the size window rejects it.

    Providers with an output-size contract answer ``ask_size(w, h)`` with None when the
    geometry cannot be sent (Seedream: total pixels >= 3.6864 M *and* long side <= 2800,
    so a very wide strip is unsendable).  The image is then halved along its long axis,
    each half is repaired on its own and the two answers are cross-faded over a 64 px
    overlap.  Halves without a single hole pixel are copied, never billed.  Safety net
    only: for the shipped example every atlas and the background fit in one call.
    """
    h, w = img.shape[:2]
    if prov.ask_size(w, h) is not None or depth >= 4:
        return prov.fill(img, mask, prompt, dump=dump, rec=rec, seed=seed)
    T_fn and T_fn(f"      {prov.name}: {w}x{h} outside the provider size window -> split")
    ov = max(8, min(64, min(w, h) // 8))
    if w >= h:
        cut = w // 2
        a, b = max(0, cut - ov), min(w, cut + ov)
        boxes = [(0, 0, b, h), (a, 0, w, h)]
    else:
        cut = h // 2
        a, b = max(0, cut - ov), min(h, cut + ov)
        boxes = [(0, 0, w, b), (0, a, w, h)]
    res = []
    for xa, ya, xb, yb in boxes:
        sub = img[ya:yb, xa:xb]
        sm = mask[ya:yb, xa:xb] if mask is not None else None
        if sm is not None and not bool(np.any(sm)):
            res.append(sub.copy())
            continue
        r = _prov_fill(prov, sub, sm, prompt, dump, rec, T_fn, seed, depth + 1)
        res.append(sub.copy() if r is None else r)
    out = img.copy()
    ramp = np.linspace(0.0, 1.0, max(1, b - a), dtype=np.float32)
    if w >= h:
        out[:, :a] = res[0][:, :a]
        out[:, b:] = res[1][:, b - a:]
        out[:, a:b] = (res[0][:, a:b] * (1.0 - ramp)[None, :, None]
                       + res[1][:, :b - a] * ramp[None, :, None]).astype(np.uint8)
    else:
        out[:a, :] = res[0][:a, :]
        out[b:, :] = res[1][b - a:, :]
        out[a:b, :] = (res[0][a:b, :] * (1.0 - ramp)[:, None, None]
                       + res[1][:b - a, :] * ramp[:, None, None]).astype(np.uint8)
    return out


def repair_image_gen(todo, masks, holes, work, plate, W, H, cfg, T_fn=None, rec=None):
    """Article 2.3.1: pack -> image model -> sanity gate -> paste back.

    The model is pluggable (``--gen-backend``).  Mask-native providers (local Flux,
    wanx2.1-imageedit) get the packed atlas plus its binary hole mask; mask-less ones
    (Seedream) get the same atlas with the holes painted pure black -- exactly the
    article's wording -- plus a 0-999 <bbox> token, and no mask image at all.
    """
    prov = get_provider(cfg.gen_backend, cfg, T_fn)
    model = str(getattr(prov, "model", "") or "")
    prompt = cfg.atlas_prompt or (ATLAS_PROMPT if prov.supports_mask else ATLAS_PROMPT_SINGLE)
    seeds = C.GEN_SEEDS if prov.supports_mask else (C.GEN_SEEDS[0],)
    items = collect_atlas_items(todo, masks, holes, work, plate, W, H,
                                prefill=bool(cfg.atlas_prefill))
    log, done, settled, atlases = [], set(), set(), []
    head = dict(provider=prov.name, model=model, mask_sent=bool(prov.supports_mask),
                prompt_chars=len(prompt), seeds=list(seeds))
    if not items:
        return work, log, done, settled, dict(atlases=0, items=0, accepted=0,
                                              rejected=0, **head)
    budget = int(cfg.atlas_max_side * cfg.atlas_max_side * C.ATLAS_FILL_BUDGET)
    chunks = _chunk(items, budget)
    dbg = os.path.join(cfg.dest, "_debug_atlas")
    accepted = rejected = 0
    for ci, chunk in enumerate(chunks, start=1):
        atlas, amask, cells = pack_atlas(chunk, cfg.atlas_max_side)
        hole_mask = amask > 0
        # one region hint per packed cell: without them a mask-less model tends to
        # fill the big frame hole and silently keep the small silhouette holes black
        # hole boxes in ATLAS coordinates: the <bbox> tokens address the sent image,
        # not the individual cell crops
        regions = []
        for c in cells:
            cm = amask[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] > 0
            if not cm.any():
                continue
            ys, xs = np.where(cm)
            regions.append((c["x"] + int(xs.min()), c["y"] + int(ys.min()),
                            c["x"] + int(xs.max()) + 1, c["y"] + int(ys.max()) + 1))
        pprompt = prov.prompt_with_regions(prompt, regions, shape=atlas.shape[:2])
        # what actually goes to the model: the packed atlas, or the same atlas with
        # every hole painted pure black when the provider has no mask channel
        send = atlas if prov.supports_mask else InpaintProvider.black_hole(atlas, hole_mask)
        tag = (prov.name + model).encode("utf-8")
        key = hashlib.md5(enc_png(send) + enc_png(amask) + pprompt.encode("utf-8")
                          + tag + cfg.ckpt.encode("utf-8")).hexdigest()[:16]
        imwrite(os.path.join(dbg, f"atlas_{key}_sent.png"), send)
        imwrite(os.path.join(dbg, f"atlas_{key}_mask.png"), amask)
        T_fn and T_fn(f"2.3.1 atlas {ci}/{len(chunks)} [{prov.name}{('/' + model) if model else ''}]: "
                      f"{len(chunk)} element(s) {atlas.shape[1]}x{atlas.shape[0]} "
                      f"hole={hole_mask.mean():.1%} key={key}")
        result = None
        tries = []
        for seed in seeds:
            ckey = f"atlas_{key}_{prov.name}_{seed}"
            cpath = os.path.join(cfg.cache, ckey + ".png") if cfg.cache else None
            t0 = time.time()
            try:
                if cpath and os.path.exists(cpath) and not cfg.refresh:
                    out = imread(cpath)
                    src = "cache"
                    dt = time.time() - t0
                else:
                    out = _prov_fill(prov, send, hole_mask, pprompt, rec=rec, T_fn=T_fn,
                                     seed=seed, dump=os.path.join(
                                         dbg, f"atlas_{key}_{prov.name}_seed{seed}.png"))
                    if out is None:
                        raise RuntimeError(f"{prov.name} returned no image")
                    src = prov.name
                    dt = time.time() - t0
                    if out.shape[:2] != atlas.shape[:2]:
                        out = cv2.resize(out, (atlas.shape[1], atlas.shape[0]),
                                         interpolation=cv2.INTER_LANCZOS4)
                    if cpath:
                        imwrite(cpath, out)
                ok, blk, std, delta = _sanity_ok(out, hole_mask, cfg.atlas_max_black)
                tries.append(dict(seed=seed, source=src, seconds=round(dt, 1),
                                  black_ratio=round(blk, 4), hole_std=round(std, 2),
                                  material_delta=round(delta, 1), accepted=bool(ok)))
                T_fn and T_fn(f"      seed={seed} {src} {dt:.1f}s black={blk:.2%} "
                              f"std={std:.1f} delta={delta:.0f} {'ok' if ok else 'REJECT'}")
                if ok:
                    result = out
                    accepted += 1
                    break
                rejected += 1
            except Exception as e:
                tries.append(dict(seed=seed, error=str(e)[:200]))
                T_fn and T_fn(f"      !! atlas gen failed: {str(e)[:160]}")
                break                       # provider down -> do not burn the other seeds
        atlases.append(dict(index=ci, key=key, elements=[c["id"] for c in cells],
                            size=[int(atlas.shape[1]), int(atlas.shape[0])],
                            scale=cells[0].get("scale", 1.0),
                            hole_ratio=round(float(hole_mask.mean()), 4),
                            tries=tries, accepted=result is not None))
        if result is None:
            continue
        for it, c in zip(chunk, cells):
            cell = result[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]
            if cell.shape[:2] != (c["oh"], c["ow"]):
                cell = cv2.resize(cell, (c["ow"], c["oh"]), interpolation=cv2.INTER_LANCZOS4)
            x0, y0, x1, y1 = it["box"]
            hm = it["mask"]
            settled.add(it["id"])
            cblk = float((cell.max(axis=2) < 8)[hm].mean()) if hm.any() else 0.0
            if cblk > cfg.atlas_max_black:
                # the model kept (part of) this hole black -> traditional fill, so an
                # exported layer can never carry a black gap (salvage, no extra call)
                npx, med = _ns_fill_region(work, plate, masks[it["id"]], holes[it["id"]],
                                           W, H)
                log.append(dict(id=it["id"], mode="image", method="ns-cellfallback",
                                hole_px=npx, fill=med, atlas=key,
                                cell_black=round(cblk, 4)))
                continue
            reg = work[y0:y1, x0:x1]
            reg[hm] = cell[hm]
            done.add(it["id"])
            log.append(dict(id=it["id"], mode="image", method=f"{prov.name}-atlas",
                            hole_px=it["hole_px"], atlas=key, atlas_cell=[c["x"], c["y"],
                            c["w"], c["h"]], scale=c.get("scale", 1.0),
                            cell_black=round(cblk, 4),
                            black_ratio=tries[-1].get("black_ratio") if tries else None))
    report = dict(atlases=len(atlases), items=len(items), accepted=accepted,
                  rejected=rejected, detail=atlases,
                  cell_fallback=sorted(settled - done), **head)
    if T_fn:
        T_fn(f"2.3.1 generative repair: {len(done)}/{len(items)} element(s) via "
             f"{len(atlases)} atlas(es), provider={prov.name}"
             + (f", {len(report['cell_fallback'])} cell(s) salvaged by NS"
                if report["cell_fallback"] else ""))
    return work, log, done, settled, report


def repair_elements(plan_export, masks, holes, work, plate, W, H, cfg, T_fn=None, rec=None):
    """2.3.1 + 2.3.2 for every extracted element, following the LLM decision.

    Everything runs in ONE z_order-ascending pass (parents before children).  A
    parent's repair repaints the footprint of the elements it carries, so a child
    repaired earlier would be overwritten; likewise a parent repaired later would
    destroy the child's fill.  Atlas groups (one image-model call covering several
    elements) are executed when their lowest-z member is reached, and any element
    whose group failed the sanity gate falls back to the traditional fill at its
    own turn, still inside the ordered pass.
    """
    log = []
    surf = {a["id"]: a for a in plan_export
            if a["element_repair_mode"] == "surface" and int(holes[a["id"]].sum()) > 0}
    img = {a["id"]: a for a in plan_export if a["element_repair_mode"] == "image"
           and int(holes[a["id"]].sum()) >= C.IMAGE_REPAIR_MIN_HOLE}
    use_gen = bool(img) and cfg.repair in ("gen", "auto")
    report = dict(policy=cfg.repair, candidates=len(img), gen=0, ns=0,
                  min_hole_px=C.IMAGE_REPAIR_MIN_HOLE, atlases=[], fallback=[],
                  order="z_order ascending, parents first", generative=use_gen)

    group_of, groups = {}, []
    if use_gen:
        ordered = sorted(img.values(), key=lambda a: (a["z_order"], a["id"]))
        budget = int(cfg.atlas_max_side * cfg.atlas_max_side * C.ATLAS_FILL_BUDGET)

        def _crop_area(a):
            x0, y0, x1, y1 = boxof(masks[a["id"]])
            return (x1 - x0 + 20) * (y1 - y0 + 20)

        groups = _chunk(ordered, budget, area=_crop_area)
        for gi, ch in enumerate(groups):
            for a in ch:
                group_of[a["id"]] = gi

    gen_done, done_groups, n_surf = set(), set(), 0
    settled = set()          # ids the group already handled (gen or cell salvage)
    for a in sorted(plan_export, key=lambda a: (a["z_order"], a["id"])):
        i = a["id"]
        if i in surf:
            work, npx, med = surface_fill(work, plate, masks[i], holes[i])
            log.append(dict(id=i, mode="surface", method="surface-fill",
                            hole_px=npx, fill=med))
            n_surf += 1
            continue
        if i not in img:
            continue
        if i in group_of:
            gi = group_of[i]
            if gi not in done_groups:
                done_groups.add(gi)
                work, glog, gdone, gsettled, grep = repair_image_gen(
                    groups[gi], masks, holes, work, plate, W, H, cfg, T_fn, rec=rec)
                log += glog
                gen_done |= gdone
                settled |= gsettled
                report["atlases"] += grep.get("detail", [])
                report["atlas_summary"] = {k: v for k, v in grep.items() if k != "detail"}
            if i in settled:
                continue
        npx, med = _ns_fill_region(work, plate, masks[i], holes[i], W, H)
        log.append(dict(id=i, mode="image",
                        method="ns-fallback" if use_gen else "ns",
                        hole_px=npx, fill=med))
        report["fallback"].append(i)
    if n_surf and T_fn:
        T_fn(f"2.3.2 surface repair: {n_surf} element(s)")
    report["gen"] = len(gen_done)
    report["ns"] = len(report["fallback"])
    return work, log, report


# --------------------------------------------------------- 2.3.3 background
def make_bg_holes(plate, allmask, soft=C.BG_SCENE_SOFT_EDGE, blend=C.BG_SCENE_SOFT_BLEND):
    """Black diagnostic hole + a very narrow softened edge (article, scene mode)."""
    out = plate.copy()
    m = allmask.astype(np.uint8)
    out[allmask] = 0
    if soft > 0:
        ring = (cv2.dilate(m, np.ones((2 * soft + 1, 2 * soft + 1), np.uint8)) > 0) & (m == 0)
        if ring.any():
            out[ring] = (out[ring] * blend).astype(np.uint8)
    return out


def repair_background(allmask, plate, H, W, bg_repair, cfg, T_fn=None, rec=None):
    bgholes = make_bg_holes(plate, allmask)
    bgmode = (bg_repair or {}).get("mode", "none")
    want_gen = cfg.bg == "gen" or (cfg.bg == "auto" and bgmode == "scene")
    sanity = []
    if T_fn:
        T_fn(f"2.3.3 background mode={bgmode} hole={allmask.mean():.1%} want_gen={want_gen}")
    if bgmode == "none":
        return plate.copy(), bgholes, "plate", sanity
    if bgmode == "surface":
        bg = cv2.inpaint(bgholes, allmask.astype(np.uint8) * 255, C.BG_NS_RADIUS, cv2.INPAINT_NS)
        return bg, bgholes, "ns-surface", sanity

    bg = None
    if want_gen:
        prov = get_provider(cfg.gen_backend, cfg, T_fn)
        model = str(getattr(prov, "model", "") or "")
        prompt = cfg.bg_prompt or BG_PROMPT
        seeds = C.GEN_SEEDS if prov.supports_mask else (C.GEN_SEEDS[0],)
        # full native resolution: the provider scales internally when it must, and a
        # 1024 px round trip only blurs the scene the plate is judged against
        send = bgholes if prov.supports_mask else InpaintProvider.black_hole(bgholes, allmask)
        dbg = os.path.join(cfg.dest, "_debug_bg")
        imwrite(os.path.join(dbg, "holes_sent.png"), send)
        imwrite(os.path.join(dbg, "allmask.png"), allmask.astype(np.uint8) * 255)
        tag = (prov.name + model).encode("utf-8")
        key = hashlib.md5(enc_png(send) + prompt.encode("utf-8") + tag
                          + cfg.ckpt.encode("utf-8")).hexdigest()[:16]
        if T_fn:
            T_fn(f"      provider={prov.name}{('/' + model) if model else ''} "
                 f"mask={'yes' if prov.supports_mask else 'no (black holes)'} "
                 f"{W}x{H} key={key}")
        for seed in seeds:
            cpath = os.path.join(cfg.cache, f"bg_{key}_{prov.name}_{seed}.png") if cfg.cache else None
            t0 = time.time()
            try:
                if cpath and os.path.exists(cpath) and not cfg.refresh:
                    out, src = imread(cpath), "cache"
                    dt = time.time() - t0
                else:
                    out = _prov_fill(prov, send, allmask, prompt, rec=rec, T_fn=T_fn,
                                     seed=seed, dump=os.path.join(
                                         dbg, f"bg_{prov.name}_seed{seed}.png"))
                    if out is None:
                        raise RuntimeError(f"{prov.name} returned no image")
                    src, dt = prov.name, time.time() - t0
                    if out.shape[:2] != (H, W):
                        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LANCZOS4)
                    if cpath:
                        imwrite(cpath, out)
                ok, blk, std, delta = _sanity_ok(out, allmask, cfg.bg_max_black)
                sanity.append(dict(seed=seed, source=src, seconds=round(dt, 1),
                                   provider=prov.name, model=model,
                                   size=[W, H],
                                   black_ratio=round(blk, 4), hole_std=round(std, 2),
                                   material_delta=round(delta, 1), accepted=bool(ok)))
                if T_fn:
                    T_fn(f"      seed={seed} {src} {dt:.1f}s black={blk:.2%} "
                         f"std={std:.1f} delta={delta:.0f} {'ok' if ok else 'REJECT'}")
                if ok:
                    bg = out.copy()
                    bg[~allmask] = plate[~allmask]
                    imwrite(os.path.join(dbg, "fused.png"), bg)
                    return bg, bgholes, f"{prov.name}-scene:{src}", sanity
            except Exception as e:
                sanity.append(dict(seed=seed, provider=prov.name, error=str(e)[:200]))
                if T_fn:
                    T_fn(f"      !! background gen failed: {str(e)[:160]}")
                break
        if T_fn:
            T_fn("      !! all seeds failed -> NS fallback")
    bg = cv2.inpaint(bgholes, allmask.astype(np.uint8) * 255, C.BG_NS_RADIUS, cv2.INPAINT_NS)
    bg = np.where(allmask[..., None], bg, plate).astype(np.uint8)
    return bg, bgholes, "ns", sanity


class RepairStep(PipelineStep):
    name = "2.3 Repair (elements + background)"

    def run(self, ctx):
        cfg = ctx.config
        dest = cfg.dest
        rec = []                      # one record per generative provider call
        ctx.work, ctx.repair_log, report = repair_elements(
            ctx.plan_export, ctx.masks, ctx.holes, ctx.plate.copy(), ctx.plate,
            ctx.W, ctx.H, cfg, T_fn=ctx.log, rec=rec)
        ctx.atlas_report = report
        imwrite(os.path.join(dest, "01b_elements_repaired.png"), ctx.work)
        with open(os.path.join(dest, "repair_log.json"), "w", encoding="utf-8") as f:
            json.dump(ctx.repair_log, f, ensure_ascii=False, indent=1, default=str)

        ctx.bg, ctx.bgholes, ctx.bg_method, ctx.bg_sanity = repair_background(
            ctx.allmask, ctx.plate, ctx.H, ctx.W, ctx.bg_repair, cfg, T_fn=ctx.log,
            rec=rec)
        ctx.model_calls = rec
        paid = [r for r in rec if r.get("kind") != "cache"]
        ctx.log(f"2.3 generative provider calls: {len(paid)} "
                f"({', '.join(sorted({str(r.get('kind')) for r in paid})) or '-'})")
        imwrite(os.path.join(dest, "02_background_holes.png"), ctx.bgholes)
        imwrite(os.path.join(dest, "02_background_plate.png"), ctx.bg)

        summ = report.get("atlas_summary") or {}
        pname = summ.get("provider") or cfg.gen_backend
        n_surf = sum(1 for r in ctx.repair_log if r["mode"] == "surface")
        n_gen = sum(1 for r in ctx.repair_log
                    if str(r.get("method", "")).endswith("-atlas"))
        n_ns = sum(1 for r in ctx.repair_log if r["mode"] == "image"
                   and not str(r.get("method", "")).endswith("-atlas"))
        ctx.note("2.3.1 Image-model repair", "done" if report["candidates"] else "nothing-to-do",
                 f"policy={cfg.repair}; provider={pname}/{summ.get('model') or '-'}; "
                 f"candidates={report['candidates']}; "
                 f"{pname}-atlas={n_gen} via {len(report.get('atlases', []))} atlas(es); "
                 f"traditional fallback={n_ns}; prompt={summ.get('prompt_chars', 0)} chars "
                 f"(article 2.3.1{'' if summ.get('mask_sent', True) else ', single-image rewrite'}); "
                 f"mask_sent={summ.get('mask_sent')}; "
                 f"hole prefill={'median' if cfg.atlas_prefill else 'black'}; "
                 f"sanity black<={cfg.atlas_max_black:.0%} or "
                 f"(hole_std>={C.GEN_MIN_STD} and material_delta<={C.ATLAS_MAX_COLOR_DELTA}); "
                 f"seeds={summ.get('seeds') or list(C.GEN_SEEDS)}; "
                 f"cell_fallback={len(summ.get('cell_fallback') or [])}; "
                 f"calls={len(ctx.model_calls or [])}; "
                 f"cache={os.path.basename(cfg.cache or '-')}")
        ctx.note("2.3.2 Traditional repair", "done" if n_surf else "nothing-to-do",
                 f"{n_surf} surface element(s) filled from measured neighbour pixels "
                 f"(core median / 2-cluster majority) + {C.SURFACE_SEAM_KERNEL} seam blend")
        ctx.note("2.3.3 Background repair", "done",
                 f"mode={ctx.bg_repair.get('mode')} -> method={ctx.bg_method}; "
                 f"provider={pname}; native_resolution={ctx.W}x{ctx.H}; "
                 f"prompt={'case override' if cfg.bg_prompt else 'article 2.3.3'}; "
                 f"hole={float(ctx.allmask.mean()):.1%}; "
                 f"gate: black<={cfg.bg_max_black:.0%} or "
                 f"(std>={C.GEN_MIN_STD} and delta<={C.ATLAS_MAX_COLOR_DELTA}); "
                 f"tries={len(ctx.bg_sanity)}")