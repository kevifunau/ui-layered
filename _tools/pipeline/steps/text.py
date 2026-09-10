# -*- coding: utf-8 -*-
"""2.1 text handling: OCR, glyph mask estimation, text removal, style extraction,
2.1.4 raster (artistic) text handling and 2.4 text re-rendering.

The article removes *every* OCR glyph first, shows that image to the LLM, and only
then restores the regions the LLM flagged as raster text.  That ordering is why this
module is split into two pipeline steps:

    TextExtractStep   2.1.1 + 2.1.2   OCR -> glyph masks -> plate with all text gone
    TextFinalizeStep  2.1.4 + 2.1.3   apply raster_text_ids / text_corrections -> final
                                      plate -> style estimation
"""
import json
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidocr_onnxruntime import RapidOCR

from .. import config as C
from ..config import T, imwrite, jdump
from .base import PipelineStep

_DIGIT_RE = re.compile(r"\d{3,}")


class TextPipeline:
    """OCR -> text removal -> style extraction -> re-render."""

    def __init__(self, src, H, W, dest, cfg):
        self.src = src
        self.H, self.W = H, W
        self.dest = dest
        self.cfg = cfg
        self.rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.entries = []          # every text region (OCR + LLM corrections)
        self.drop = {"low_conf": 0, "single_latin": 0, "vertical_deco": 0}
        self.textmask = None
        self.plate = None
        self.styles = []
        self._ocr = RapidOCR()

    # ------------------------------------------------------------ small helpers
    @staticmethod
    def _is_cjk(ch):
        """CJK ideographs plus CJK punctuation / fullwidth forms.

        A label such as "\uff1f\uff1f\uff1f" is not an ideograph but must still be
        rendered with the CJK font, otherwise Arial shows tofu boxes.
        """
        o = ord(ch)
        return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF
                or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF)

    @classmethod
    def _has_cjk(cls, s):
        return any(cls._is_cjk(c) for c in (s or ""))

    @staticmethod
    def _luma709(c):
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    @staticmethod
    def _hexs(h):
        h = h.lstrip("#")
        return tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))

    @staticmethod
    def _bucket_color(px, bg):
        """2.1.3-1: quantise each channel into 32-wide buckets (centres 16..240) and
        score a bucket by ``pixel_count * (24 + RGB distance to the local background)``.
        """
        if len(px) == 0:
            return [0, 0, 0]
        q = np.clip((px.astype(np.int32) // C.COLOR_BUCKET) * C.COLOR_BUCKET + C.COLOR_BUCKET // 2,
                    0, 240)
        keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
        uk, cnt = np.unique(keys, return_counts=True)
        best, bs = None, -1.0
        bgf = np.asarray(bg, float)
        for k, c in zip(uk, cnt):
            cen = np.array([(k >> 16) & 255, (k >> 8) & 255, k & 255], float)
            s = c * (C.COLOR_BUCKET_SCORE_BASE + float(np.linalg.norm(cen - bgf)))
            if s > bs:
                bs, best = s, cen
        return [int(v) for v in best]

    def _font_for(self, st, size):
        bd = st["font_weight"] >= C.WEIGHT_BOLD
        if st["font_family"] == "Microsoft YaHei":
            p = C.YAHEI_BD if bd else C.YAHEI
        else:
            p = C.ARIAL_BD if bd else C.ARIAL
        try:
            return ImageFont.truetype(p, max(C.FONT_SIZE_FLOOR, size))
        except Exception:
            return ImageFont.load_default()

    def path(self, name):
        return os.path.join(self.dest, name)

    # ------------------------------------------------------------- 2.1.1 OCR
    def run_ocr(self):
        T(f"2.1.1 OCR on {self.W}x{self.H} source (min conf {C.OCR_MIN_CONF})")
        raw, _ = self._ocr(self.src)
        self.entries = []
        self.drop = {"low_conf": 0, "single_latin": 0, "vertical_deco": 0}
        for quad, text, s in raw or []:
            s = float(s)
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            if s < C.OCR_MIN_CONF:
                self.drop["low_conf"] += 1
                continue
            if len(text) == 1 and text.isascii() and text.isalpha() and s < C.OCR_ICON_CONF:
                self.drop["single_latin"] += 1
                continue
            if s < C.OCR_ICON_CONF and bw > 0 and bh / bw >= C.OCR_VERTICAL_RATIO:
                self.drop["vertical_deco"] += 1
                continue
            self.entries.append(dict(
                text=text, score=s, source="rapidocr",
                quad=[[float(p[0]), float(p[1])] for p in quad],
                bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]))
        # stable ids -- the LLM prompt references them, raster_text_ids answers with them
        for k, o in enumerate(self.entries, start=1):
            o["id"] = f"text_{k:03d}"
        T(f"      kept {len(self.entries)} / {len(raw or [])}   dropped={self.drop}")
        self._review_digits()
        return self.entries

    def _review_digits(self):
        """Repo extension: re-OCR pure numbers (>= 3 digits) at 2x/3x."""
        for o in self.entries:
            if not _DIGIT_RE.fullmatch(o["text"] or ""):
                continue
            x0, y0, x1, y1 = o["bbox"]
            pad = 6
            crop = self.src[max(0, y0 - pad):min(self.H, y1 + pad),
                            max(0, x0 - pad):min(self.W, x1 + pad)]
            if crop.size == 0:
                continue
            for sc in C.OCR_DIGIT_SCALES:
                up = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                res, _ = self._ocr(up)
                if res and len(res) == 1:
                    t2, s2 = res[0][1], float(res[0][2])
                    if t2 != o["text"] and s2 >= o["score"]:
                        T(f"      digit review: {o['text']} -> {t2} "
                          f"(conf {o['score']:.2f}->{s2:.2f}, {sc}x)")
                        o["text"], o["score"] = t2, s2
                        o["digit_reviewed"] = sc
                        break

    # -------------------------------------------------- 2.1.2 glyph masks
    def _base_color(self, crop, allowed, h, w):
        """Local background colour of one text region.

        Article: median of the border band (width clamp(min(h,w)/4,1,3)) restricted to
        the OCR quadrilateral.  ``bg_estimate="kmeans"`` keeps the previous repo
        extension (2-cluster majority centre) for bands that contain two materials.
        """
        lo, hi = C.BAND_WIDTH_RANGE
        bwi = int(np.clip(min(h, w) / C.BAND_WIDTH_DIV, lo, hi))
        band = np.zeros((h, w), bool)
        band[:bwi, :] = band[-bwi:, :] = band[:, :bwi] = band[:, -bwi:] = True
        band &= allowed
        if self.cfg.bg_estimate == "kmeans" and band.sum() >= C.BAND_KMEANS_MIN:
            bp = crop[band].reshape(-1, 3).astype(np.float32)
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, lab, cen = cv2.kmeans(bp, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            cnt = np.bincount(lab.ravel(), minlength=2)
            return cen[int(np.argmax(cnt))], int(band.sum()), "kmeans"
        if band.sum():
            return np.median(crop[band].reshape(-1, 3), axis=0), int(band.sum()), "median"
        return crop[allowed].reshape(-1, 3).mean(axis=0), int(band.sum()), "mean-allowed"

    def estimate_entry(self, o):
        """Article 2.1.2 steps 1-5 for one text region."""
        x0, y0, x1, y1 = o["bbox"]
        h, w = y1 - y0, x1 - x0
        if h < 2 or w < 2:
            o["mask_mode"] = "skipped"
            return False
        quad = np.array(o["quad"], np.float32)
        crop = self.rgb[y0:y1, x0:x1]
        allowed = np.zeros((h, w), np.uint8)
        cv2.fillPoly(allowed, [(quad - np.array([x0, y0], np.float32)).astype(np.int32)], 255)
        allowed = allowed > 0
        B, band_px, bg_method = self._base_color(crop, allowed, h, w)
        d3 = np.clip(C.GLYPH_DIST_SCALE * np.linalg.norm(crop - B, axis=2), 0, 255).astype(np.uint8)
        thresh, _ = cv2.threshold(d3[allowed].reshape(1, -1), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        glyph = (d3 >= max(C.GLYPH_MIN_DIST, C.GLYPH_OTSU_RATIO * thresh)) & allowed
        glyph = cv2.morphologyEx(glyph.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones(C.CC_KERNEL, np.uint8)) > 0
        ratio = float(glyph.sum()) / max(1, int(allowed.sum()))
        coarse = ratio < C.GLYPH_AREA_MIN or ratio > C.GLYPH_AREA_MAX
        if coarse:
            glyph = allowed
        single = len(o["text"] or "") == 1 and self._is_cjk((o["text"] or " ")[0])
        fr, lo, hi = C.DILATE_CJK if single else C.DILATE_OTHER
        rad = int(np.clip(round(fr * h), lo, hi))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
        hard = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
        o.update(mask_mode=C.MASK_MODE_COARSE if coarse else C.MASK_MODE_GLYPH,
                 glyph_ratio=round(ratio, 4), bg=[int(v) for v in B],
                 bg_method=bg_method, band_px=band_px, dilate_radius=rad,
                 glyph_px=int(glyph.sum()), hard_px=int(hard.sum()),
                 glyph=glyph, hard=hard)
        return True

    def build_glyph_masks(self, entries=None):
        for o in (entries if entries is not None else self.entries):
            self.estimate_entry(o)

    def compose_textmask(self):
        """Union of the hard masks of every editable-text entry (article 2.1.2 step 5)."""
        tm = np.zeros((self.H, self.W), np.uint8)
        for o in self.entries:
            if o.get("render") != "editable_text" or o.get("hard") is None:
                continue
            x0, y0, x1, y1 = o["bbox"]
            tm[y0:y1, x0:x1] = np.maximum(tm[y0:y1, x0:x1], o["hard"].astype(np.uint8) * 255)
        self.textmask = tm
        return tm

    def make_plate(self):
        """Article 2.1.2 step 6: cv2.inpaint(radius=5, INPAINT_TELEA)."""
        self.plate = cv2.inpaint(self.src, self.textmask, C.INPAINT_RADIUS, cv2.INPAINT_TELEA)
        return self.plate

    def mark_all_editable(self):
        for o in self.entries:
            o["render"] = "editable_text"
            o["raster_reason"] = None

    # ------------------------------------------------- 2.1.4 raster text
    def apply_plan(self, instances, raster_text_ids, corrections=(), trusted_ids=False):
        """Restore artistic text (2.1.4) and add OCR misses reported by the LLM.

        ``raster_text_ids`` is the article's mechanism.  The prompt also *requires* the
        LLM to emit one non-text query (kind=logo) at the location of every raster id,
        so an id is only honoured when that query actually exists -- or when the plan
        was generated with the OCR id list this run produced (``trusted_ids``).  Plans
        cached from a run without the id list contain invented ids, and honouring them
        blindly turns ordinary labels such as "金币：100" into raster art.
        """
        ids = set(raster_text_ids or [])
        raster_boxes = []
        for a in instances or []:
            if a.get("kind") == "logo" or a.get("role") == "raster_text":
                bb = a["geometry_hints"][0].get("bbox_px")
                if bb:
                    raster_boxes.append((a["id"], bb))

        def geometric_hit(bb):
            for qid, rb in raster_boxes:
                ix = max(0, min(rb[2], bb[2]) - max(rb[0], bb[0]))
                iy = max(0, min(rb[3], bb[3]) - max(rb[1], bb[1]))
                area = max(1, (bb[2] - bb[0]) * (bb[3] - bb[1]))
                if ix * iy > C.RASTER_OVERLAP_RATIO * area:
                    return qid
            return None

        by_id = {o["id"] for o in self.entries}
        n_id = n_geo = 0
        rejected = []
        for o in self.entries:
            o["render"] = "editable_text"
            o["raster_reason"] = None
            hit = geometric_hit(o["bbox"]) if o.get("hard") is not None else None
            if o["id"] in ids:
                if hit:
                    o["render"] = "raster_asset"
                    o["raster_reason"] = f"llm_raster_text_id+{hit}"
                    n_id += 1
                    continue
                if trusted_ids:
                    o["render"] = "raster_asset"
                    o["raster_reason"] = "llm_raster_text_id(verified id list)"
                    n_id += 1
                    continue
                rejected.append(dict(id=o["id"], text=o["text"], bbox=o["bbox"],
                                     reason="raster_text_id without the logo query the "
                                            "prompt requires, and the plan carries no "
                                            "verified OCR id list"))
            if hit:
                o["render"] = "raster_asset"
                o["raster_reason"] = f"logo_bbox_overlap:{hit}"
                n_geo += 1
        unknown = sorted(i for i in ids if i not in by_id)

        added = []
        for c in corrections or []:
            entry = dict(c)
            entry["id"] = f"corr_{len(added) + 1:03d}"
            entry["render"] = "editable_text"
            if self.estimate_entry(entry):
                added.append(entry)
        self.entries.extend(added)
        T(f"2.1.4 raster text: {n_id} by raster_text_ids, {n_geo} by logo bbox, "
          f"{len(rejected)} id(s) rejected, {len(added)} LLM text correction(s) added"
          + (f", unknown ids {unknown}" if unknown else ""))
        for r in rejected:
            T(f"      reject {r['id']} {r['text']!r}: {r['reason']}")
        return dict(by_id=n_id, by_bbox=n_geo, unknown_ids=unknown, rejected=rejected,
                    corrections=len(added), trusted_ids=bool(trusted_ids))

    # --------------------------------------------------- 2.1.3 styles
    def extract_styles(self):
        core_filter = bool(self.cfg.fg_core_filter)
        for o in self.entries:
            if o.get("render") != "editable_text" or o.get("glyph") is None:
                o.pop("style", None)
                continue
            x0, y0, x1, y1 = o["bbox"]
            h, w = y1 - y0, x1 - x0
            gm = o["glyph"]                       # article: inside the glyph mask
            src_rgb = cv2.cvtColor(self.src, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1][gm]
            used = len(src_rgb)
            if core_filter and used:
                dd = np.linalg.norm(src_rgb - np.asarray(o["bg"], np.float32), axis=1)
                core = src_rgb[dd > C.FG_CORE_RATIO * float(dd.max())]
                if len(core) >= C.FG_CORE_MIN:
                    src_rgb, used = core, len(core)
            fg = self._bucket_color(src_rgb, o["bg"])
            bgc = [int(v) for v in o["bg"]]
            # article 2.1.3-4: glyph coverage drives weight and stroke width
            coverage = int(o.get("glyph_px", gm.sum())) / max(1, w * h)
            coverage_hard = int(o.get("hard_px", 0)) / max(1, w * h)
            text = o["text"] or ""
            fam = "Microsoft YaHei" if self._has_cjk(text) else "Arial"
            single_cjk = len(text) == 1 and self._is_cjk(text[0])
            size = max(C.FONT_SIZE_MIN,
                       int(round((C.FONT_SIZE_CJK if single_cjk else C.FONT_SIZE_OTHER) * h)))
            weight = C.WEIGHT_BOLD if coverage >= C.WEIGHT_COVERAGE else C.WEIGHT_SEMI
            stroke = C.STROKE_DARK if self._luma709(fg) >= self._luma709(bgc) else C.STROKE_LIGHT
            swr = C.STROKE_W_CJK if coverage > C.STROKE_COVERAGE else C.STROKE_W_OTHER
            lo, hi = C.STROKE_WIDTH_RANGE
            sw = int(np.clip(round(swr * h), lo, hi))
            o["style"] = dict(font_family=fam, font_size=size, font_weight=weight,
                              color="#%02x%02x%02x" % tuple(fg), stroke_color=stroke,
                              stroke_width=sw, coverage=round(coverage, 4),
                              coverage_hard_mask=round(coverage_hard, 4),
                              bg_color="#%02x%02x%02x" % tuple(bgc),
                              fg_pixels=used, fg_core_filter=core_filter)
        self.styles = [dict((k, v) for k, v in o.items()
                            if k not in ("glyph", "hard")) for o in self.entries]
        jdump(self.styles, self.path("text_layers.json"))
        T(f"2.1.3 styles: {sum(1 for s in self.styles if s.get('style'))} editable entries "
          f"-> text_layers.json")
        return self.styles

    # ------------------------------------------------------- 2.4 re-render
    def render(self, reb):
        pil = Image.fromarray(cv2.cvtColor(reb, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        n = 0
        for o in self.entries:
            st = o.get("style")
            if not st or o.get("render") != "editable_text":
                continue
            x0, y0, x1, y1 = o["bbox"]
            bw, bh = x1 - x0, y1 - y0
            size, sw = st["font_size"], st["stroke_width"]
            f = self._font_for(st, size)
            # article 2.1.3-3: measure the stroked bbox with PIL, shrink until it fits
            while size > C.FONT_SIZE_FLOOR:
                bb = dr.textbbox((0, 0), o["text"], font=f, stroke_width=sw)
                if bb[2] - bb[0] <= bw and bb[3] - bb[1] <= bh:
                    break
                size -= 1
                f = self._font_for(st, size)
            bb = dr.textbbox((0, 0), o["text"], font=f, stroke_width=sw)
            dr.text((x0 - bb[0], y0 - bb[1]), o["text"], font=f,
                    fill=self._hexs(st["color"]), stroke_width=sw,
                    stroke_fill=self._hexs(st["stroke_color"]))
            o["rendered"] = dict(font_size=size, stroke_width=sw)
            n += 1
        reb_txt = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        imwrite(self.path("03_reassembly.png"), reb_txt)
        T(f"2.4 reassembly + {n} re-rendered text layers -> 03_reassembly.png")
        self.extract_styles()      # persist the finally used font size
        return reb_txt


def save_mask_png(path, mask):
    cv2.imencode(".png", (mask > 0).astype(np.uint8) * 255)[1].tofile(path)


class TextExtractStep(PipelineStep):
    """2.1.1 OCR + 2.1.2 removal of *all* OCR glyphs (the LLM input image)."""

    name = "2.1.1/2.1.2 OCR + text removal"

    def run(self, ctx):
        cfg = ctx.config
        tp = TextPipeline(ctx.src, ctx.H, ctx.W, cfg.dest, cfg)
        ctx.text_pipe = tp
        tp.run_ocr()
        ctx.ocr_ids = [o["id"] for o in tp.entries]
        tp.build_glyph_masks()
        tp.mark_all_editable()
        tp.compose_textmask()
        tp.make_plate()
        ctx.textmask_all = tp.textmask.copy()
        ctx.plate_all = tp.plate.copy()
        ctx.plate = tp.plate          # provisional until TextFinalizeStep runs
        ctx.textmask = tp.textmask

        imwrite(tp.path("00_source.png"), ctx.src)
        imwrite(tp.path("01_text_removed.png"), tp.plate)
        save_mask_png(tp.path("01a_text_mask.png"), tp.textmask)

        modes = {}
        for o in tp.entries:
            modes[o.get("mask_mode", "?")] = modes.get(o.get("mask_mode", "?"), 0) + 1
        ctx.log(f"2.1.2 textmask={int((tp.textmask > 0).sum())}px modes={modes} "
                f"bg_estimate={cfg.bg_estimate}")
        ctx.note("2.1.1 OCR", "done",
                 f"RapidOCR min_conf={C.OCR_MIN_CONF}, single-latin<{C.OCR_ICON_CONF} dropped, "
                 f"h/w>={C.OCR_VERTICAL_RATIO} dropped, digit re-review; "
                 f"kept={len(tp.entries)} dropped={tp.drop}")
        ctx.note("2.1.2 Text removal", "done",
                 f"band clamp(min(h,w)/{C.BAND_WIDTH_DIV},{C.BAND_WIDTH_RANGE[0]},"
                 f"{C.BAND_WIDTH_RANGE[1]}) bg={cfg.bg_estimate}; "
                 f"{C.GLYPH_DIST_SCALE}d>=max({C.GLYPH_MIN_DIST},{C.GLYPH_OTSU_RATIO}*Otsu); "
                 f"3x3 close; {C.GLYPH_AREA_MIN:.1%}/{C.GLYPH_AREA_MAX:.0%} -> "
                 f"{C.MASK_MODE_COARSE}; ellipse dilate {C.DILATE_CJK}/{C.DILATE_OTHER}; "
                 f"Telea r={C.INPAINT_RADIUS}; modes={modes}")
        ctx.note("2.1.2 OCR mask diagnostic", "done", "01a_text_mask.png")


class TextFinalizeStep(PipelineStep):
    """2.1.4 raster text handling + 2.1.3 style extraction (needs the LLM plan)."""

    name = "2.1.3/2.1.4 raster text + styles"

    def run(self, ctx):
        cfg = ctx.config
        tp = ctx.text_pipe
        trusted = bool(ctx.plan_full) and ctx.plan_full.get("ocr_ids_sent") == ctx.ocr_ids
        report = tp.apply_plan(ctx.instances, ctx.raster_text_ids, ctx.corrections,
                              trusted_ids=trusted)
        tp.compose_textmask()
        tp.make_plate()
        ctx.textmask = tp.textmask
        ctx.plate = tp.plate
        imwrite(tp.path("01c_text_removed_final.png"), tp.plate)
        save_mask_png(tp.path("01c_text_mask_final.png"), tp.textmask)
        tp.extract_styles()
        ctx.styles = tp.styles

        raster = [o["id"] for o in tp.entries if o.get("render") == "raster_asset"]
        ctx.log(f"2.1.4 raster ids={raster}")
        ctx.note("2.1.4 Raster text", "done",
                 f"raster_text_ids={len(ctx.raster_text_ids)} honoured_by_id={report['by_id']} "
                 f"by_logo_bbox={report['by_bbox']} rejected={len(report['rejected'])} "
                 f"unknown={report['unknown_ids']} verified_id_list={report['trusted_ids']}; "
                 f"raster regions keep their original pixels (mask removed)")
        ctx.note("2.1.3 Style extraction", "done",
                 f"32-wide buckets x (24 + dist); glyph-mask pixels"
                 f"{', core>' + str(C.FG_CORE_RATIO) + '*dmax filter' if cfg.fg_core_filter else ''}; "
                 f"YaHei/Arial; {C.FONT_SIZE_CJK}h/{C.FONT_SIZE_OTHER}h min {C.FONT_SIZE_MIN}px; "
                 f"coverage>={C.WEIGHT_COVERAGE}->700 else 600; Rec.709 stroke colour; "
                 f"stroke {C.STROKE_W_CJK}h/{C.STROKE_W_OTHER}h clamp {C.STROKE_WIDTH_RANGE}")
        ctx.note("text_corrections", "done" if report["corrections"] else "empty",
                 f"{report['corrections']} OCR miss(es) reported by the LLM were added to the "
                 f"text mask and re-rendered as editable text")