# -*- coding: utf-8 -*-
"""LLM layer planning: prompt input assembly, plan resolution, instance expansion.

Article 2.2.1: the LLM receives (1) the original image, (2) the OCR-cleaned working
image and (3) the prompt, and the OCR regions are listed with stable ids so the model
can answer with ``raster_text_ids``.  Both the id list and the returned ids are used
here -- the previous version dropped them.
"""
import base64
import json
import os
import re
import time
from pathlib import Path

from .config import (LLM_BASE_URL, PLAN_DIR, T, jdump, load_api_key)
from .prompts import PLAN_PROMPT

_FAMILY_RE = re.compile(r"_(r\d+c\d+|h\d+|i\d+|\d+)$")
_JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)

DEFAULT_KIND = "decoration"
DEFAULT_ROLE = "foreground"
DEFAULT_REPAIR = "none"


# --------------------------------------------------------------------------- ids
def family_of(layer_id: str) -> str:
    """Strip the instance suffix so repeated sprites land in one family.

    ``coin_r1c1 -> coin``, ``card_h2 -> card``, ``badge_collected_3 -> badge_collected``.
    """
    return _FAMILY_RE.sub("", layer_id or "") or (layer_id or "")


def ocr_id(index: int) -> str:
    return f"text_{index:03d}"


def build_ocr_block(ocr, H, W) -> str:
    """Render the OCR list the prompt promises ("listed with stable ids below")."""
    if not ocr:
        return "\n\nOCR regions: none detected."
    lines = ["", "", f"OCR regions ({len(ocr)}), format id | text | bbox_norm | conf:"]
    for o in ocr:
        x0, y0, x1, y1 = o["bbox"]
        bn = [round(v, 4) for v in (x0 / max(1, W), y0 / max(1, H),
                                    x1 / max(1, W), y1 / max(1, H))]
        text = (o.get("text") or "").replace("\n", " ").strip()
        lines.append(f"  {o['id']} | {text} | {bn} | {float(o.get('score', 0)):.2f}")
    return "\n".join(lines)


# --------------------------------------------------------------------- plan I/O
def resolve_plan(src, plan_dir=PLAN_DIR):
    """Find the cached plan for a source image.  Returns (path, is_fallback)."""
    base = os.path.splitext(os.path.basename(src))[0]
    real = os.path.join(plan_dir, base + ".json")
    if os.path.exists(real):
        try:
            data = json.load(open(real, encoding="utf-8"))
        except Exception:
            data = {}
        return real, data.get("generated_by") == "fallback_grid"
    fb = os.path.join(plan_dir, "fallback__" + base + ".json")
    if os.path.exists(fb):
        return fb, True
    return None, False


def extract_json(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(content)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = content.find("{"), content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])
    raise RuntimeError(f"no JSON in LLM response: {content[:300]}")


def generate_llm_plan(src, cleaned, ocr, H, W, model, api_key=None, plan_dir=PLAN_DIR,
                      base_url=LLM_BASE_URL, timeout=180):
    """Call the vision LLM with original + OCR-cleaned image + prompt + OCR id list."""
    from openai import OpenAI

    api_key = api_key or load_api_key()

    def encode_image(path, mime):
        return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("utf-8")

    prompt = PLAN_PROMPT + build_ocr_block(ocr, H, W)
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": encode_image(src, "image/jpeg")}},
        {"type": "image_url", "image_url": {"url": encode_image(cleaned, "image/png")}},
        {"type": "text", "text": prompt},
    ]}]
    T(f"2.2.1 calling {model} (original + OCR-cleaned + {len(ocr or [])} OCR ids)")
    t0 = time.time()
    resp = client.chat.completions.create(model=model, messages=messages, timeout=timeout)
    content = resp.choices[0].message.content
    T(f"2.2.1 {model} replied in {time.time() - t0:.1f}s ({len(content)} chars)")
    plan = extract_json(content)
    base = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(plan_dir, base + ".json")
    plan.setdefault("generated_by", model)
    plan.setdefault("generated_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    plan.setdefault("source", os.path.basename(src))
    # lets a later run tell whether raster_text_ids refer to *these* OCR ids
    plan["ocr_ids_sent"] = [o["id"] for o in (ocr or [])]
    jdump(plan, out)
    T(f"2.2.1 plan saved -> {os.path.basename(out)} ({len(plan.get('queries', []))} queries)")
    return out


def load_plan(plan_path):
    """Load a plan file.  Returns the full dict (queries + background_repair + ...)."""
    return json.load(open(plan_path, encoding="utf-8"))


# ------------------------------------------------------- normalise / expand
def normalize_query(q):
    q = dict(q)
    q["id"] = q.get("id") or "element"
    q.setdefault("name", q["id"])
    q["kind"] = q.get("kind") or DEFAULT_KIND
    q["role"] = q.get("role") or DEFAULT_ROLE
    q["element_repair_mode"] = q.get("element_repair_mode") or DEFAULT_REPAIR
    if q["element_repair_mode"] not in ("image", "surface", "none"):
        q["element_repair_mode"] = DEFAULT_REPAIR
    q.setdefault("parent_query_id", None)
    try:
        q["z_order"] = int(q.get("z_order", 0))
    except (TypeError, ValueError):
        q["z_order"] = 0
    hints = [h for h in (q.get("geometry_hints") or []) if h]
    q["geometry_hints"] = hints
    return q


def convert_coords(plan, H, W):
    """bbox_norm / *_points_norm -> *_px, in place.  Idempotent."""
    for a in plan:
        for h in a.get("geometry_hints", []):
            if "bbox_px" not in h and "bbox_norm" in h:
                bn = h["bbox_norm"]
                h["bbox_px"] = [int(bn[0] * W), int(bn[1] * H), int(bn[2] * W), int(bn[3] * H)]
            if "positive_points_px" not in h:
                pts = h.get("positive_points_norm") or []
                h["positive_points_px"] = [[int(p[0] * W), int(p[1] * H)] for p in pts]
            if "negative_points_px" not in h:
                pts = h.get("negative_points_norm") or []
                h["negative_points_px"] = [[int(p[0] * W), int(p[1] * H)] for p in pts]
            h.setdefault("bbox_px", [0, 0, W - 1, H - 1])
            h.setdefault("positive_points_px", [])
            h.setdefault("negative_points_px", [])
            # keep hints inside the canvas, SAM rejects out-of-range boxes
            x0, y0, x1, y1 = h["bbox_px"]
            h["bbox_px"] = [max(0, min(x0, x1, W - 1)), max(0, min(y0, y1, H - 1)),
                            max(1, min(max(x0, x1), W - 1)), max(1, min(max(y0, y1), H - 1))]
            h["positive_points_px"] = [[min(max(0, p[0]), W - 1), min(max(0, p[1]), H - 1)]
                                       for p in h["positive_points_px"]]
            h["negative_points_px"] = [[min(max(0, p[0]), W - 1), min(max(0, p[1]), H - 1)]
                                       for p in h["negative_points_px"]]


def expand_instances(queries):
    """One entry per geometry hint.

    The article asks the LLM to share one query across repeated instances and give
    each instance its own hint.  Everything downstream segments exactly one hint, so
    multi-hint queries are expanded here instead of being silently truncated.
    """
    out = []
    for raw in queries:
        q = normalize_query(raw)
        hints = q["geometry_hints"]
        if not hints:
            continue
        n = len(hints)
        for k, h in enumerate(hints):
            inst = dict(q)
            inst["id"] = q["id"] if n == 1 else f"{q['id']}_h{k + 1}"
            inst["geometry_hints"] = [h]
            inst["query_id"] = q["id"]
            inst["instance_index"] = k
            inst["instance_count"] = n
            inst["family"] = family_of(q["id"])
            out.append(inst)
    # duplicate ids would corrupt every dict keyed by id -> make them unique
    seen = {}
    for inst in out:
        i = inst["id"]
        if i in seen:
            seen[i] += 1
            inst["id"] = f"{i}__{seen[i]}"
        else:
            seen[i] = 0
    return out


def children_map(instances):
    """parent_query_id -> child instance ids (article: the chain controls repair)."""
    kids = {}
    for x in instances:
        p = x.get("parent_query_id")
        if not p:
            continue
        kids.setdefault(p, []).append(x["id"])
    return kids


def corrections_to_boxes(plan_full, H, W):
    """text_corrections -> pixel boxes for text the OCR missed."""
    out = []
    for c in plan_full.get("text_corrections") or []:
        bn = c.get("bbox_norm")
        if not bn or len(bn) != 4:
            continue
        x0, y0 = int(bn[0] * W), int(bn[1] * H)
        x1, y1 = int(bn[2] * W), int(bn[3] * H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        out.append(dict(text=c.get("text") or "", score=float(c.get("confidence", 0.5) or 0.5),
                        bbox=[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                        quad=[[min(x0, x1), min(y0, y1)], [max(x0, x1), min(y0, y1)],
                              [max(x0, x1), max(y0, y1)], [min(x0, x1), max(y0, y1)]],
                        source="llm_text_correction"))
    return out