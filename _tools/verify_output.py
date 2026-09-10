# -*- coding: utf-8 -*-
"""verify_output.py -- quality gate for a run_pipeline.py output directory.

Usage:
  python verify_output.py                       # newest output dir containing audit.json
  python verify_output.py --dest output\基础测试
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import OUT, imread, load_case_config            # noqa: E402
from pipeline.llm_planner import family_of                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dest", default=None)
A = ap.parse_args()

if A.dest:
    D = A.dest if os.path.isabs(A.dest) else os.path.join(OUT, A.dest)
else:
    odir = os.path.join(OUT, "output")
    cands = [os.path.join(odir, d) for d in os.listdir(odir)
             if os.path.isdir(os.path.join(odir, d))
             and os.path.exists(os.path.join(odir, d, "audit.json"))]
    if not cands:
        sys.exit("[ERROR] no output directory with audit.json under output/")
    D = max(cands, key=os.path.getmtime)
if not os.path.exists(os.path.join(D, "audit.json")):
    sys.exit(f"[ERROR] missing audit.json: {D}")
print("== target:", D)

au = json.load(open(os.path.join(D, "audit.json"), encoding="utf-8"))
mf = json.load(open(os.path.join(D, "manifest.json"), encoding="utf-8"))
rl_path = os.path.join(D, "repair_log.json")
rl = json.load(open(rl_path, encoding="utf-8")) if os.path.exists(rl_path) else []
byid = {m["id"]: m for m in mf["layers"]}
case = load_case_config(au.get("source", ""))

lc = au["lossless_check"]
print(f"layers: {au['layers']}  files: {len(os.listdir(os.path.join(D, 'layers')))}"
      f"  lossless: {lc['verdict']}  unexpected_diff_px: {lc['unexpected_diff_px']}")
print(f"coverage: {au.get('coverage')}  hole_px: {au.get('hole_px')}")
print(f"plan: {au.get('plan', {}).get('path')} ({au.get('plan', {}).get('source')})"
      f"  queries={au.get('plan', {}).get('queries')}"
      f"  instances={au.get('plan', {}).get('instances')}")
print(f"mode: {au.get('mode')}")
print(f"bg_method: {au.get('bg_method')}  background_sanity: {au.get('background_sanity')}")
print(f"duplicates: {au.get('duplicates')}")
print(f"sprite_mismatch: {au.get('sprite_mismatch')}")

print("\n== timings (s) ==")
for k, v in (au.get("timings") or {}).items():
    print(f"   {k:<44} {v}")

print("\n== fidelity (generated from the actual run) ==")
for f in au.get("fidelity") or []:
    print(f"   [{f.get('status'):<14}] {f.get('step')}")
    if f.get("note"):
        print(f"        {f['note']}")

print("\n== 2.3 element repair ==")
surf = [r for r in rl if r["mode"] == "surface"]
gen = [r for r in rl if r.get("method") == "flux-atlas"]
ns = [r for r in rl if r["mode"] == "image" and r.get("method") != "flux-atlas"]
print(f"   surface traditional fill : {len(surf)}")
print(f"   image model (atlas)      : {len(gen)}")
print(f"   image traditional (NS)   : {len(ns)}")
for r in gen:
    print(f"     gen  {r['id']:<18} hole={r['hole_px']:<7} atlas={r.get('atlas')}"
          f" scale={r.get('scale')}")
for r in ns:
    print(f"     ns   {r['id']:<18} hole={r['hole_px']:<7} method={r.get('method')}")
rep = au.get("element_repair") or {}
if rep.get("atlases"):
    print(f"   atlases: {len(rep['atlases'])}  summary={rep.get('atlas_summary')}")
    for a in rep["atlases"]:
        print(f"     #{a['index']} {a['size'][0]}x{a['size'][1]} scale={a['scale']} "
              f"hole={a['hole_ratio']:.1%} accepted={a['accepted']} "
              f"elements={len(a['elements'])}")

print("\n== repeated instance consistency ==")
g = collections.defaultdict(list)
for m in mf["layers"]:
    g[family_of(m["id"])].append((m["id"], tuple(m["size"]), m["alpha_px"]))
shown = 0
for k, v in sorted(g.items()):
    if len(v) < 3:
        continue
    sizes = sorted({x[1] for x in v})
    areas = [x[2] for x in v]
    modes = {byid[x[0]]["element_repair_mode"] for x in v if x[0] in byid}
    sprite_family = (modes == {"none"})
    flag = "OK" if (len(sizes) <= 2 or not sprite_family) else "SIZE MISMATCH"
    note = "" if sprite_family else "  (not a sprite family, sizes may differ)"
    print(f"   {k:<16} n={len(v)} sizes={len(sizes)} alpha={min(areas)}..{max(areas)}"
          f"  {flag}{note}")
    shown += 1
if not shown:
    print("   (no family with >= 3 instances)")

truth = case.get("truth_bbox") or {}
if truth:
    print("\n== bbox error vs hand-labelled truth ==")

    def err(b, t):
        return max(abs(b[0] - t[0]), abs(b[1] - t[1]), abs(b[2] - t[2]), abs(b[3] - t[3]))

    have = [i for i in truth if i in byid]
    if have:
        e = [err(byid[i]["mask_bbox"], truth[i]) for i in have]
        print(f"   v5: mean={np.mean(e):.1f}px max={max(e)}px n={len(have)}")
        v4 = {c["id"]: c for c in au.get("v4_compare") or []}
        e4 = [err(v4[i]["v4_bbox"], truth[i]) for i in have if i in v4]
        if e4:
            print(f"   v4: mean={np.mean(e4):.1f}px max={max(e4)}px n={len(e4)}")
    else:
        print("   (no labelled layer present in this run)")
elif au.get("v4_compare"):
    print("\n== v4 -> v5 area change ==")
    c = sorted(au["v4_compare"], key=lambda c: -c["gain"])
    print(f"   grew {sum(1 for x in c if x['gain'] > 0)}/{len(c)}  "
          f"max {c[0]['pct']:+.1f}% ({c[0]['id']})  min {c[-1]['pct']:+.1f}% ({c[-1]['id']})")
else:
    print("\n== v4 compare / truth: skipped (not enabled for this case) ==")

print("\n== layers flagged for manual review ==")
n = 0
for m in mf["layers"]:
    if m.get("review"):
        print(f"   {m['id']:<18} {m['review']}")
        n += 1
print("   total", n)

print("\n== background plate ==")
hol = imread(os.path.join(D, "02_background_holes.png"))
bg = imread(os.path.join(D, "02_background_plate.png"))
plate_name = "01c_text_removed_final.png"
if not os.path.exists(os.path.join(D, plate_name)):
    plate_name = "01_text_removed.png"
plate = imread(os.path.join(D, plate_name))
if hol is None or bg is None or plate is None:
    print("   (missing images, skipped)")
else:
    am = (hol.max(axis=2) == 0)
    g2 = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = np.hypot(cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3))
    gr = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    er = np.hypot(cv2.Sobel(gr, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gr, cv2.CV_32F, 0, 1, ksize=3))
    blk = float(((bg.max(axis=2) == 0) & am).sum()) / max(1, int(am.sum()))
    print(f"   reference plate: {plate_name}")
    print(f"   hole {am.mean():.1%}; pure black inside hole={blk:.2%}; "
          f"std={g2[am].std():.1f} edge={e[am].mean():.1f} strong={(e[am] > 60).mean() * 100:.1f}%")
    print(f"   visible background outside hole: std={gr[~am].std():.1f} "
          f"edge={er[~am].mean():.1f} strong={(er[~am] > 60).mean() * 100:.1f}%")
    diff = int((np.abs(bg[~am].astype(int) - plate[~am].astype(int)).max(axis=1) > 0).sum())
    print(f"   pixels outside the hole that differ from the cleaned plate: {diff}")