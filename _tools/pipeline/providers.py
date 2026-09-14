# -*- coding: utf-8 -*-
"""Generative inpainting providers (ISS-035).

The pipeline asks every generative model exactly one question: "here is an image and
the boolean mask of its missing (hole) pixels; give me back the same-size image with
the holes plausibly continued".  Vendors answer differently, so the differences live
here and ``steps/repair.py`` only programs against :class:`InpaintProvider`:

* :class:`SeedreamProvider` -- Volcengine Ark Agent Plan (doubao-seedream-5.0-pro by
  default, ``data/secrets/ark.json``).  The API has NO mask parameter, so the hole is
  painted black in the single input image and pointed at with a 0-999 ``<bbox>`` token
  in the prompt (docs/82379/2582775).  Output size must be requested explicitly:
  >= SEEDREAM_MIN_PIXELS total and long side <= SEEDREAM_MAX_SIDE, otherwise the call
  400s or silently no-ops (measured 2026-09-14).
* :class:`FluxProvider` -- local ComfyUI, mask native.  Retired as the default.
* :class:`DashscopeProvider` -- Aliyun wanx2.1-imageedit, mask native.  Experiments.

``fill()`` returns a full-size image; repair.py pastes it back inside its own mask, so
a provider is free to touch pixels outside the hole.
"""
import base64
import json
import math
import os
import time

import cv2
import numpy as np

from . import config as C


def _png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


class InpaintProvider:
    """Base class: name / supports_mask / fill(img, mask, prompt, dump, rec)."""

    name = "base"
    supports_mask = False

    def __init__(self, cfg, T_fn=None):
        self.cfg = cfg
        self.T_fn = T_fn

    def log(self, msg):
        if self.T_fn:
            self.T_fn(msg)

    def fill(self, img, mask, prompt, dump=None, rec=None, seed=None):
        """Return a same-size BGR image with ``mask`` plausibly filled, else None.

        ``img`` already carries the hole; ``mask`` is the boolean hole mask.  A
        mask-native provider forwards both; a mask-less one gets ``img`` with the hole
        painted pure black (article wording) and uses ``mask`` only to point at the
        region.  The caller pastes the answer back inside its own mask, so a provider
        is free to touch pixels outside the hole.
        """
        raise NotImplementedError

    def ask_size(self, w, h):
        """(aw, ah) to request, or None when this geometry cannot be sent as-is.

        Providers without an output-size contract echo the input size, which also tells
        ``steps.repair._prov_fill`` that no long-axis split is needed.
        """
        return (w, h)

    def prompt_with_regions(self, prompt, regions, shape=None):
        """Attach region hints for mask-less providers; mask-native ones ignore them
        (their mask channel already points at every hole).  ``regions`` are boolean
        masks or absolute (x0, y0, x1, y1) pixel boxes of the sent image."""
        return prompt

    # ---------------- shared helpers ----------------
    @staticmethod
    def b64(img):
        return base64.b64encode(_png_bytes(img)).decode("utf-8")

    @staticmethod
    def decode(raw, shape, dump=None):
        """Decode bytes, resize back to ``shape`` (h, w) and optionally dump raw."""
        out = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if out is None:
            return None
        if dump:
            from .config import imwrite
            imwrite(dump, out)
        if out.shape[:2] != tuple(shape):
            out = cv2.resize(out, (shape[1], shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return out

    @staticmethod
    def norm_bbox(mask, pad=8):
        """Hole bbox as Seedream 0-999 normalised <bbox> tokens."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        h, w = mask.shape
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(w, int(xs.max()) + 1 + pad)
        y2 = min(h, int(ys.max()) + 1 + pad)
        f = lambda v, n: int(min(999, max(0, round(v / float(n) * 1000))))
        return (f(x1, w), f(y1, h), f(x2, w), f(y2, h))

    @staticmethod
    def black_hole(img, mask):
        out = img.copy()
        out[mask] = 0
        return out


class SeedreamProvider(InpaintProvider):
    """Volcengine Ark image generation; single image in, single image out, no mask."""

    name = "seedream"
    supports_mask = False

    def __init__(self, cfg, T_fn=None):
        super().__init__(cfg, T_fn)
        ark = _ark_json()
        self.base_url = os.environ.get("UI_LAYERED_SEEDREAM_BASE_URL",
                                       ark.get("base_url") or C.SEEDREAM_BASE_URL)
        self.model = os.environ.get("UI_LAYERED_SEEDREAM_MODEL",
                                    ark.get("model") or C.SEEDREAM_MODEL)
        self.api_key = C.load_ark_key()
        self.attempts = C.SEEDREAM_ATTEMPTS

    # ---- size rules (measured 2026-09-14) ----
    def ask_size(self, w, h):
        """(aw, ah) to request, or None when the geometry cannot satisfy the API."""
        if w * h >= C.SEEDREAM_MIN_PIXELS:
            aw, ah = w, h
        else:
            s = math.sqrt(C.SEEDREAM_MIN_PIXELS / float(w * h))
            aw, ah = int(math.ceil(w * s)), int(math.ceil(h * s))
            while aw * ah < C.SEEDREAM_MIN_PIXELS:
                ah += 1
        if max(aw, ah) > C.SEEDREAM_MAX_SIDE:
            return None
        return aw, ah

    def prompt_with_regions(self, prompt, regions, shape=None):
        """0-999 normalised <bbox> tokens (docs/82379/2582775): one per hole region, in
        the SENT image's coordinates, so the model also notices the small
        silhouette-shaped gaps of an atlas."""
        toks = []
        for r in (regions or [])[:12]:
            if r is None:
                continue
            if isinstance(r, np.ndarray):
                bb = self.norm_bbox(r > 0)
            else:
                if not shape:
                    continue
                x0, y0, x1, y1 = r
                h, w = shape
                f = lambda v, n: int(min(999, max(0, round(v / float(n) * 1000))))
                bb = (f(x0, w), f(y0, h), f(x1, w), f(y1, h))
            if bb:
                toks.append(f"<bbox>{bb[0]} {bb[1]} {bb[2]} {bb[3]}</bbox>")
        if not toks:
            return prompt
        return ("图 1 " + " ".join(toks) + " 框出的区域里，纯黑部分是缺失内容，"
                "必须全部填补，不允许保留任何黑色。\n" + prompt)

    def _call(self, img, prompt, ask):
        import requests
        url = f"{self.base_url}/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model, "prompt": prompt,
                   "image": f"data:image/png;base64,{self.b64(img)}",
                   "watermark": False,          # API default is True
                   "output_format": "png",      # API default jpeg adds DCT noise
                   "response_format": "url"}
        if ask:
            payload["size"] = f"{ask[0]}x{ask[1]}"
        r = requests.post(url, headers=headers, json=payload, timeout=C.SEEDREAM_TIMEOUT)
        if r.status_code != 200:
            self.log(f"seedream: HTTP {r.status_code} {r.text[:200]}")
            return None
        d = r.json()
        if d.get("error"):
            self.log(f"seedream: error {json.dumps(d['error'], ensure_ascii=False)[:200]}")
            return None
        data = d.get("data") or []
        if not data:
            self.log(f"seedream: empty data {json.dumps(d, ensure_ascii=False)[:200]}")
            return None
        item = data[0]
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"]), item.get("size", "")
        import requests as rq
        return rq.get(item["url"], timeout=60).content, item.get("size", "")

    def fill(self, img, mask, prompt, dump=None, rec=None, seed=None):
        """``seed`` is ignored on purpose: the Agent Plan images endpoint documents no
        seed parameter, so a mask-less run makes one call per hole set (covered by
        SEEDREAM_ATTEMPTS) instead of the article's three seeds."""
        h, w = img.shape[:2]
        ask = self.ask_size(w, h)
        if ask is None:
            self.log(f"seedream: {w}x{h} cannot satisfy "
                     f"[{C.SEEDREAM_MIN_PIXELS}px, side<={C.SEEDREAM_MAX_SIDE}] -> split needed")
            return None
        send = self.black_hole(img, mask) if mask is not None else img
        p = prompt          # region hints are baked in by prompt_with_regions()
        for attempt in range(1, self.attempts + 1):
            t0 = time.time()
            got = self._call(send, p, ask)       # explicit size first: a no-size call
            if got is None:                      # was measured to return a model bucket
                got = self._call(send, p, None)  # (2048x2048, 1824x2224), never (w, h)
            dt = time.time() - t0
            if got is None:
                if rec is not None:
                    rec.append(dict(kind=self.name, model=self.model, attempt=attempt,
                                    seconds=round(dt, 2), in_size=[w, h],
                                    asked=[ask[0], ask[1]], ok=False))
                continue
            blob, rsize = got
            self.log(f"seedream({self.model}) attempt{attempt}: {dt:.1f}s returned {rsize} "
                     f"(input {w}x{h}, asked {ask[0]}x{ask[1]})")
            out = self.decode(blob, (h, w), dump=dump)
            if out is not None:
                if rec is not None:
                    rec.append(dict(kind=self.name, model=self.model, attempt=attempt,
                                    seconds=round(dt, 2), in_size=[w, h],
                                    asked=[ask[0], ask[1]], returned=rsize, ok=True))
                return out
        return None


class FluxProvider(InpaintProvider):
    """Local ComfyUI inpainting, mask native.  Retired as default (ISS-035)."""

    name = "flux"
    supports_mask = True

    def fill(self, img, mask, prompt, dump=None, rec=None, seed=None):
        from .steps.repair import NEG_PROMPT, comfy_run, fetch, inpaint_workflow
        t0 = time.time()
        wf = inpaint_workflow(img, mask.astype(np.uint8) * 255, self.cfg.ckpt, prompt,
                              NEG_PROMPT, C.GEN_SEEDS[0] if seed is None else seed,
                              "ui_prov_flux", self.cfg.base)
        out = fetch(comfy_run(wf, self.cfg.base), self.cfg.base)
        dt = time.time() - t0
        if out.shape[:2] != img.shape[:2]:
            out = cv2.resize(out, (img.shape[1], img.shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)
        if dump:
            from .config import imwrite
            imwrite(dump, out)
        if rec is not None:
            rec.append(dict(kind="flux", ckpt=self.cfg.ckpt, seed=seed,
                            in_size=[img.shape[1], img.shape[0]],
                            seconds=round(dt, 2), ok=True))
        return out


class DashscopeProvider(InpaintProvider):
    """Aliyun wanx2.1-imageedit, mask native.  Experiments only (ISS-027/029)."""

    name = "dashscope"
    supports_mask = True

    @staticmethod
    def _data_url(img, ext=".png"):
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise RuntimeError(f"cannot encode {ext}")
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

    def _fit(self, img, mask):
        """Scale both images into the documented [512, 4096] window, keeping the mask
        strictly binary and the pair at identical resolution."""
        h, w = img.shape[:2]
        s = 1.0
        if min(h, w) < C.DASHSCOPE_MIN_SIDE:
            s = C.DASHSCOPE_MIN_SIDE / float(min(h, w))
        if max(h, w) * s > C.DASHSCOPE_MAX_SIDE:      # aspect cannot satisfy both ->
            s = C.DASHSCOPE_MAX_SIDE / float(max(h, w))  # keep the max side legal
        nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
        if (nh, nw) != (h, w):
            img = cv2.resize(img, (nw, nh),
                             interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            mask = cv2.resize(mask.astype(np.uint8), (nw, nh),
                              interpolation=cv2.INTER_NEAREST) > 0
        m = np.zeros((nh, nw, 3), np.uint8)
        m[mask.astype(bool)] = 255
        return img, m, (nh, nw)

    def fill(self, img, mask, prompt, dump=None, rec=None, seed=None):
        import requests
        key = C.load_api_key()
        h, w = img.shape[:2]
        t0 = time.time()
        base, m8, _ = self._fit(img, mask.astype(bool))
        bu, mu = self._data_url(base), self._data_url(m8)
        if len(bu) + len(mu) > C.DASHSCOPE_MAX_BYTES:      # 10 MB request ceiling
            bu = self._data_url(base, ".jpg")
        body = {"model": C.DASHSCOPE_MODEL,
                "input": {"function": C.DASHSCOPE_FUNCTION,
                          "base_image_url": bu, "mask_image_url": mu, "prompt": prompt},
                "parameters": {"n": 1, "watermark": False}}
        if seed is not None:
            body["parameters"]["seed"] = int(seed)
        head = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "X-DashScope-Async": "enable"}
        r = requests.post(C.DASHSCOPE_URL, headers=head, json=body, timeout=120)
        task = ""
        if r.status_code == 200:
            task = (r.json().get("output") or {}).get("task_id", "")
        if not task:
            self.log(f"dashscope: HTTP {r.status_code} {r.text[:200]}")
            if rec is not None:
                rec.append(dict(kind=self.name, model=C.DASHSCOPE_MODEL, seed=seed,
                                in_size=[w, h], seconds=round(time.time() - t0, 2),
                                ok=False, error=r.text[:200]))
            return None
        out_o, status = {}, ""
        for _ in range(int(C.DASHSCOPE_TIMEOUT // 5)):
            time.sleep(5)
            st = requests.get(C.DASHSCOPE_TASK_URL + task,
                              headers={"Authorization": f"Bearer {key}"}, timeout=60).json()
            out_o = st.get("output") or {}
            status = out_o.get("task_status", "")
            if status in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"):
                break
        dt = time.time() - t0
        res = out_o.get("results") or []
        if status != "SUCCEEDED" or not res or not res[0].get("url"):
            self.log(f"dashscope: task {status} {json.dumps(out_o, ensure_ascii=False)[:200]}")
            if rec is not None:
                rec.append(dict(kind=self.name, model=C.DASHSCOPE_MODEL, seed=seed,
                                in_size=[w, h], seconds=round(dt, 2), ok=False,
                                error=str(out_o)[:200]))
            return None
        blob = requests.get(res[0]["url"], timeout=120).content
        self.log(f"dashscope({C.DASHSCOPE_MODEL}) seed={seed}: {dt:.1f}s ok")
        out = self.decode(blob, (h, w), dump=dump)
        if rec is not None:
            rec.append(dict(kind=self.name, model=C.DASHSCOPE_MODEL, seed=seed,
                            in_size=[w, h], seconds=round(dt, 2), ok=out is not None))
        return out


def _ark_json():
    f = os.path.join(C.SECRET_DIR, "ark.json")
    if os.path.exists(f):
        try:
            return json.load(open(f, encoding="utf-8"))
        except Exception:
            return {}
    return {}


_PROVIDERS = {"seedream": SeedreamProvider, "flux": FluxProvider,
              "dashscope": DashscopeProvider}


def get_provider(name, cfg, T_fn=None):
    try:
        cls = _PROVIDERS[name]
    except KeyError:
        raise RuntimeError(f"unknown gen backend {name!r}; "
                           f"choose from {sorted(_PROVIDERS)}") from None
    return cls(cfg, T_fn=T_fn)
