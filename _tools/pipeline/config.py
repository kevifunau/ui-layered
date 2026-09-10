# -*- coding: utf-8 -*-
"""Shared constants, paths and small IO helpers.

Every numeric parameter that the article states explicitly lives here so the whole
replication can be audited in one place.  Per-image overrides (dropped layers, safe
area, ground-truth boxes) live in ``data/cases/<source stem>.json`` instead of being
hardcoded in the algorithm.
"""
import json
import os
import time

import cv2
import numpy as np

# ============ Paths ============
OUT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(OUT, "data")
PLAN_DIR = os.path.join(DATA_DIR, "plans")
CASE_DIR = os.path.join(DATA_DIR, "cases")
SECRET_DIR = os.path.join(DATA_DIR, "secrets")
SRC_DEFAULT = os.path.join(OUT, "TestCase", "\u7f8a\u4e86\u4e2a\u7f8a", "source_screens",
                           "screenShot_20260828_095350.jpeg")
DEST_DEFAULT = os.path.join(OUT, "output", "\u7f8a\u4e86\u4e2a\u7f8a_\u660e\u4fe1\u7247\u6536\u96c6")

SAMDIR = os.environ.get("UI_LAYERED_SAMDIR", r"E:\CodeRep\Comfyui\ComfyUI\models\sams\sam-vit-base")
CKPT = os.environ.get("UI_LAYERED_CKPT", "flux1-dev-fp8.safetensors")
BASE = os.environ.get("UI_LAYERED_COMFYUI", "http://127.0.0.1:8188")
LLM_MODEL = os.environ.get("UI_LAYERED_LLM_MODEL", "qwen3.7-plus")
LLM_BASE_URL = os.environ.get("UI_LAYERED_LLM_BASE_URL",
                              "https://dashscope.aliyuncs.com/compatible-mode/v1")

YAHEI = r"C:\Windows\Fonts\msyh.ttc"
YAHEI_BD = r"C:\Windows\Fonts\msyhbd.ttc"
ARIAL = r"C:\Windows\Fonts\arial.ttf"
ARIAL_BD = r"C:\Windows\Fonts\arialbd.ttf"

# ============ 2.1.1 OCR ============
OCR_MIN_CONF = 0.35          # article: minimum confidence
OCR_ICON_CONF = 0.85         # article: single latin letter below this == icon misread
OCR_VERTICAL_RATIO = 2.4     # article: low-confidence candidates with h/w >= 2.4 dropped
OCR_DIGIT_REVIEW = 3         # repo extension: re-OCR pure numbers with >= 3 digits
OCR_DIGIT_SCALES = (2, 3)

# ============ 2.1.2 text removal ============
BAND_WIDTH_DIV = 4           # clamp(min(h,w)/4, 1, 3) border band
BAND_WIDTH_RANGE = (1, 3)
BAND_KMEANS_MIN = 40         # repo extension: only used when bg_estimate == "kmeans"
GLYPH_DIST_SCALE = 3         # d(p) = ||I(p)-B||, compared as 3d
GLYPH_MIN_DIST = 12          # 3d >= max(12, 0.65*T)
GLYPH_OTSU_RATIO = 0.65
GLYPH_AREA_MIN = 0.015       # article: < 1.5% of quad -> coarse
GLYPH_AREA_MAX = 0.68        # article: > 68% of quad -> coarse
MASK_MODE_COARSE = "coarse"
MASK_MODE_GLYPH = "estimated_glyphs"   # article wording
DILATE_CJK = (0.16, 3, 8)    # clamp(round(0.16h), 3, 8) for a single CJK char
DILATE_OTHER = (0.09, 2, 6)  # clamp(round(0.09h), 2, 6) otherwise
INPAINT_RADIUS = 5           # cv2.inpaint(..., 5, INPAINT_TELEA)
RASTER_OVERLAP_RATIO = 0.5   # geometric fallback for raster text (id match comes first)

# ============ 2.1.3 text style ============
COLOR_BUCKET = 32            # 32-wide quantisation, centres 16/48/.../240
COLOR_BUCKET_SCORE_BASE = 24
FONT_SIZE_CJK = 0.88         # single CJK char: round(0.88h)
FONT_SIZE_OTHER = 0.82
FONT_SIZE_MIN = 8
FONT_SIZE_FLOOR = 6          # shrink loop stops here
WEIGHT_COVERAGE = 0.17       # >= 0.17 -> 700 else 600
WEIGHT_BOLD, WEIGHT_SEMI = 700, 600
STROKE_COVERAGE = 0.12       # > 0.12 -> round(0.045h) else round(0.03h)
STROKE_W_CJK, STROKE_W_OTHER = 0.045, 0.03
STROKE_WIDTH_RANGE = (0, 2)
STROKE_DARK = "#1e2322"      # used when fg is not darker than bg
STROKE_LIGHT = "#f0f4f1"
FG_CORE_RATIO = 0.5          # repo extension: keep pixels farther than 0.5*dmax
FG_CORE_MIN = 20

# ============ 2.2.2 - 2.2.4 segmentation ============
CAND_W_POS = 0.22            # article scoring formula, verbatim weights
CAND_W_NEG = -0.38
CAND_W_CONTAIN = 0.10
CAND_W_IOU = 0.06
CC_KERNEL = (3, 3)           # 3x3 closing
CC_KEEP_AREA_RATIO = 0.08    # article rule 3: keep components >= 8% of the largest
CC_MIN_AREA = 150            # sprite candidate floor (repo extension)
GUARD_PAD_MIN = 10
GUARD_PAD_RATIO = 0.18
GUARD_KEEP_RATIO = 0.35
SURFACE_REFINE_MIN = 200
SURFACE_REFINE_DIST = 45
SURFACE_REFINE_MAX_GROW = 2.2
SPRITE_MATCH_SCORE = 0.85
SPRITE_ENFORCE_SCORE = 0.80
SPRITE_DUP_IOU = 0.6
SPRITE_DUP_MIN_PX = 200
SPRITE_GROUP_MIN = 3
SPRITE_GROUP_BUCKET = 16
AMBIGUOUS_MARGIN = 0.15      # diagnostic: candidates whose score margin is below this
SEG_CAND_VIZ_LIMIT = 12

# ============ 2.3 repair ============
IMAGE_REPAIR_MIN_HOLE = 30
SURFACE_SEAM_KERNEL = (5, 5)
HOLE_DILATE_KERNEL = (15, 15)
NS_BORDER_DILATE = (7, 7)
NS_BORDER_ERODE = (4, 4)
NS_BORDER_RADIUS = 3
BG_NS_RADIUS = 7
BG_SCENE_SOFT_EDGE = 3       # article: "a very narrow softened edge"
BG_SCENE_SOFT_BLEND = 0.35
BG_MAX_BLACK = 0.02          # sanity gate: black pixels inside the hole
GEN_MIN_STD = 6.0            # ...unless the hole has real texture (dark material)
ATLAS_MAX_COLOR_DELTA = 90.0 # max median-colour distance hole vs own material
ATLAS_MAX_BLACK = 0.02
ATLAS_MAX_SIDE = 1024
ATLAS_PAD = 8
ATLAS_FILL_BUDGET = 1.6      # total cell area per atlas, as a fraction of max_side^2
GEN_SEEDS = (11, 12, 13)
FLUX_STEPS = 30
FLUX_CFG = 7.0
FLUX_SAMPLER = "dpmpp_2m"
FLUX_SCHEDULER = "karras"
FLUX_DENOISE = 1.0
COMFY_TIMEOUT = 900

# ============ 2.4 export ============
ALPHA_EDGE = 128             # 1px semi-transparent rim (repo extension)
CONTACT_TILE_W = 150
CONTACT_PER_ROW = 8
SAFE_AREA_HEIGHT = 120       # repo extension, per-case overridable
DEFAULT_DROPPED_IDS = frozenset()


# ============ Utilities ============
def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, im):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(p)[1] or ".png", im)
    buf.tofile(p)
    return ok


def T(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    return path


def load_api_key():
    """DashScope key: environment first, then data/secrets/dashscope.key."""
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    f = os.path.join(SECRET_DIR, "dashscope.key")
    if os.path.exists(f):
        key = open(f, encoding="utf-8").read().strip()
        if key:
            return key
    raise RuntimeError(
        "No DashScope API key. Set DASHSCOPE_API_KEY or write "
        + f)


def load_case_config(src):
    """Per-source overrides from data/cases/<stem>.json (all keys optional)."""
    stem = os.path.splitext(os.path.basename(src))[0]
    cfg = {
        "dropped_ids": sorted(DEFAULT_DROPPED_IDS),
        "safe_area_height": SAFE_AREA_HEIGHT,
        "bg_prompt": None,
        "atlas_prompt": None,
        "truth_bbox": None,
    }
    path = os.path.join(CASE_DIR, stem + ".json")
    if os.path.exists(path):
        try:
            user = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            T(f"[WARN] bad case config {path}: {e}")
            return cfg
        cfg.update({k: v for k, v in user.items() if k in cfg})
        cfg["_path"] = path
    return cfg


for _d in (PLAN_DIR, CASE_DIR, SECRET_DIR):
    os.makedirs(_d, exist_ok=True)