# -*- coding: utf-8 -*-
"""Pipeline shared state (PipelineContext) and immutable run configuration."""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from .config import (ATLAS_MAX_SIDE, BASE, BG_MAX_BLACK, CC_KEEP_AREA_RATIO, CKPT,
                     FG_CORE_RATIO, LLM_MODEL, SAMDIR, SAFE_AREA_HEIGHT)


@dataclass(frozen=True)
class RunConfig:
    """Immutable configuration coming from the command line / case config."""

    src: str
    dest: str
    repair: str = "auto"        # "auto" | "ns" | "gen"   (auto follows the LLM decision)
    bg: str = "auto"            # "auto" | "ns" | "gen"
    refresh: bool = False       # ignore the inpaint cache
    samdir: str = SAMDIR
    ckpt: str = CKPT
    base: str = BASE
    cache: str = ""
    bg_max_black: float = BG_MAX_BLACK
    atlas_max_black: float = BG_MAX_BLACK
    atlas_max_side: int = ATLAS_MAX_SIDE
    atlas_prefill: bool = True   # seed element holes with measured material
    gen_backend: str = "seedream"  # ISS-035: seedream | flux | dashscope
    cc_keep_ratio: float = CC_KEEP_AREA_RATIO
    bg_estimate: str = "median"     # "median" (article) | "kmeans" (repo extension)
    fg_core_filter: bool = True     # repo extension, see 2.1.3
    fg_core_ratio: float = FG_CORE_RATIO
    safe_area_height: int = SAFE_AREA_HEIGHT
    dropped_ids: Set = field(default_factory=frozenset)
    llm_model: str = LLM_MODEL
    api_key: str = ""
    bg_prompt: str = ""
    atlas_prompt: str = ""
    plan_path: str = ""             # optional explicit plan override
    truth_bbox: Dict = field(default_factory=dict)
    case_config: Dict = field(default_factory=dict)


class PipelineContext:
    """Mutable shared state.  Every Step reads what it needs and writes its result."""

    def __init__(self, config: RunConfig):
        self.config = config
        self._step_name = ""
        self._t0 = time.time()
        self.fidelity: List[Dict[str, Any]] = []
        self.timings: Dict[str, float] = {}

    # ---------- logging / auditing helpers ----------
    def log(self, msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def note(self, step: str, status: str, detail: str = ""):
        """Record one fidelity line.  audit.json is generated from these, so the
        report can never claim a feature that was not actually executed."""
        self.fidelity.append(dict(step=step, status=status, note=detail))

    def elapsed(self) -> float:
        return time.time() - self._t0

    # ---------- input ----------
    src = None
    H = 0
    W = 0

    # ---------- 2.1 text ----------
    text_pipe = None
    ocr_ids: List[str] = []
    textmask = None          # final mask (raster text excluded)
    textmask_all = None      # stage 1 mask, every OCR glyph removed
    plate = None             # final cleaned image used by segmentation
    plate_all = None         # stage 1 cleaned image fed to the LLM
    styles = None
    corrections: List[Dict] = []

    # ---------- 2.2.1 plan ----------
    plan_full = None
    plan = None              # queries as returned by the LLM
    instances = None         # queries expanded to one entry per geometry hint
    plan_path = None
    plan_source = "unresolved"
    bg_repair = None
    raster_text_ids: List[str] = []

    # ---------- 2.2.2 - 2.2.4 segmentation ----------
    masks = None
    masks_pre = None
    segaudit = None
    cand_store = None

    # ---------- layer build ----------
    holes = None
    deover = None
    dup = None
    relocated = None
    mismatch = None
    surface_refined = None
    plan_export = None
    allmask = None

    # ---------- 2.3 repair ----------
    work = None
    repair_log = None
    bg = None
    bgholes = None
    bg_method = None
    bg_sanity = None
    atlas_report = None
    model_calls = None       # one record per generative provider call (audit.json)

    # ---------- 2.4 export ----------
    manifest = None
    cut_cache = None
    hole_union = None
    reb = None
    reb_no_text = None

    # ---------- audit ----------
    unexp = 0
    cov = 0.0
    cmp = None