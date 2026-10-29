"""Satellite OOF assembly without the legacy Kaggle artifacts.

`scripts/sp_assemble_oof.py` needs `kaggle_import/` (the previous team's D-FINE / RT-DETRv2 weights, their OOF
predictions and `site_crs.csv`).  This script runs the same nested protocol on artifacts this repository can build by
itself:

    data/labels_3857.parquet                python -m koz3.labels
    data/spor_prep/ann.json                 python -m koz3.sp_prep --mod SpOr        (region -> fold lives here)
    work/spor/oof_hm<tag>_cands.parquet     python -m koz3.sp_train --mod SpOr --oof --tag <tag>
    work/spor/oof_det_<tag>_cands.parquet   python -m koz3.det_train --mod SpOr --oof --tag <tag>

Members are fused with the deployed fusion (`koz3.sp_fuse.fuse`, 8 m, equal weights), then post-processing is applied in
the order used by `inference/koz3_infer/experts/sp_or.py` (threshold -> framing prior -> region kurgan gate -> damaged /
intact hedge) and scored with the official-exact `koz3.score`.  Nested protocol: the setting applied to a fold is chosen
on the other folds only (`koz3.nested`), so a reported number is never tuned on the fold it is scored on.

Score semantics — do not compare the two "alone" numbers: the `score` column of an `oof_*_cands.parquet` is the emitted
*confidence* (heatmap peak × class-mask share for the heatmap member, the detector class score for a box member), and
this script thresholds the **fused confidence**, exactly as the deployed expert does.  `koz3.sp_train.sweep` instead
applies its threshold to the raw heatmap peak and emits both kurgan classes, so its "alone" figure is a different
quantity and is systematically more optimistic on a single heatmap member.

Usage
    python scripts/sp_oof_local.py                          # full grid, every available member
    python scripts/sp_oof_local.py --list                   # what candidate files exist (counts per member/class)
    python scripts/sp_oof_local.py --alone                  # each member decoded on its own
    python scripts/sp_oof_local.py --drop det_balq           # leave-one-member-out
    python scripts/sp_oof_local.py --members hm,det_dfl     # explicit member set
    python scripts/sp_oof_local.py --frame-prior            # add the big-scene framing prior to the grid
    python scripts/sp_oof_local.py --accept thr=0.25 thr=0.3,hedge=0.4    # robust acceptance between two settings

Not included: the hillfort scene gate / centre swap (needs `work/spor/oof_scene_cls.parquet`, produced by
scripts/sp_scene_gate_oof.py).  Without it the hillfort numbers here are the detectors' own, not the deployed swap.
"""
import argparse
import glob
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from koz3.labels import DATA, region_utm, to_gt_geojson  # noqa: E402
from koz3.nested import fold_counts, nested  # noqa: E402
from koz3.score import score  # noqa: E402
from koz3.sp_fuse import fuse  # noqa: E402
from sp_fusion_oof import fc  # noqa: E402

CLS = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
K2 = ("kurgany_povrezhdennye", "kurgany_tselye")
PER_CLASS_DEFAULT = {"arkhitektury": 0.15}     # deployed (nested: 0.15 in every fold)

# setting grid: (threshold, hedge tau, region kurgan gate, framing prior)
THRS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
HEDGES = (None, 0.4)
# the gate is absolute, so its useful values depend on the fused score scale: 0.45 was nested on the full six-member
# stack, while a single weak member rarely exceeds 0.3 anywhere — keep a lower candidate so small member sets are fair
GATES = (None, 0.3, 0.45)
FRAME = (0.5, 0.35)
# whole-scene hillfort gate (koz3.scene_cls → work/spor/oof_scene_cls.parquet): mode x threshold
#   swap  = add the scene centre for p >= g, and drop the fusion's own hillforts in regions where some scene has p < g
#   fill  = add only in regions where the fusion found no hillfort
#   add   = add wherever p >= g, never drop
#   prior = ungated centre on every small scene at confidence 0.05
GOR_MODES = (None, ("swap", 0.02), ("swap", 0.1), ("fill", 0.1), ("add", 0.1), ("prior", None))

# gor: None | (mode, g) — the hillfort gate; default keeps every existing 4-argument Spec valid
Spec = namedtuple("Spec", "thr hedge gate frame gor", defaults=(None,))


def fmt(s):
    parts = [f"thr={s.thr:g}"]
    if s.hedge is not None:
        parts.append(f"hedge={s.hedge:g}")
    if s.gate is not None:
        parts.append(f"gate={s.gate:g}")
    if s.frame:
        parts.append(f"frame={s.frame[0]:g}/{s.frame[1]:g}")
    if s.gor:
        parts.append(f"gor={s.gor[0]}" + (f":{s.gor[1]:g}" if s.gor[1] is not None else ""))
    return ",".join(parts)


def parse_spec(txt):
    kv = dict(p.split("=", 1) for p in txt.split(",") if p)
    fr = kv.get("frame")
    gor = None
    if kv.get("gor"):
        mode, _, g = kv["gor"].partition(":")
        gor = (mode, float(g) if g else None)
    return Spec(float(kv.get("thr", 0.25)),
                float(kv["hedge"]) if kv.get("hedge") not in (None, "") else None,
                float(kv["gate"]) if kv.get("gate") not in (None, "") else None,
                tuple(float(v) for v in fr.split("/")) if fr else None,
                gor)


# --- inputs ---------------------------------------------------------------------------------------

def discover(work="work/spor"):
    """→ {member: parquet path} from every candidate file the OOF producers wrote."""
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, work, "oof_*_cands.parquet"))):
        name = os.path.basename(p)[len("oof_"):-len("_cands.parquet")]
        out[name] = p
    return out


def region_folds(ann_path=None):
    """region → fold, exactly the chips `koz3.sp_train` trained on (no kaggle_import folds.json needed)."""
    import json
    ann_path = ann_path or os.path.join(ROOT, "data", "spor_prep", "ann.json")
    if not os.path.exists(ann_path):
        sys.exit(f"missing {ann_path} — run: python -m koz3.sp_prep --mod SpOr")
    chips = json.load(open(ann_path, encoding="utf-8"))
    return {c["region"]: int(c["fold"]) for c in chips}


def folds_from(path):
    """Optional external fold assignment: folds.json (region__scene → fold) or a folds.csv (site, fold)."""
    import csv
    import json
    if path.endswith(".json"):
        raw = json.load(open(path, encoding="utf-8"))
        return {k.rsplit("__", 1)[0]: int(v) for k, v in raw.items()}
    return {r["site"]: int(r["fold"]) for r in csv.DictReader(open(path, encoding="utf-8"))}


def scene_geometry(regions):
    """region → [(cx, cy, half_extent_m, is_small)] in EPSG:3857, from the SpOr scene headers (no pixels read).

    Mirrors the expert: scenes are read at 0.5 m and a scene is 'small' when it has < 1e6 pixels there (0.25 km²).
    """
    import rasterio
    from pyproj import Transformer
    out = {}
    for reg in regions:
        g = []
        for p in sorted(glob.glob(os.path.join(DATA, reg, "04_*", "*.tif"))):
            try:
                with rasterio.open(p) as src:
                    crs = src.crs.to_string() if (src.crs is not None and src.crs.to_epsg()) else region_utm(os.path.join(DATA, reg))
                    b, tr = src.bounds, Transformer.from_crs(crs, 3857, always_xy=True)
                    (x0, y0), (x1, y1) = tr.transform(b.left, b.bottom), tr.transform(b.right, b.top)
                    area = (b.right - b.left) * (b.top - b.bottom)
                    g.append(((x0 + x1) / 2, (y0 + y1) / 2, max(abs(x1 - x0), abs(y1 - y0)) / 2, area / 0.25 < 1e6))
            except Exception as e:
                print(f"  scene geometry skipped for {p}: {e!r}")
        out[reg] = g
    return out


# --- decoding (same order as inference/koz3_infer/experts/sp_or.py) --------------------------------

def fuse_members(frames, match_m=8.0):
    """Fuse one candidate set per member (deployed `koz3.sp_fuse.fuse` semantics).

    When the candidates carry `source_layer` (the source image, as the experts tag it), fusion runs WITHIN each source
    image: every image is an independent reference layer, so candidates from two scenes are never merged — this is what
    `experts/sp_or.py` does since the v20 build.  Candidates written before the column existed fall back to the older
    per-region fusion, so old parquets keep working.
    """
    weights = np.ones(len(frames))
    cols = ["region", "cls", "score", "x", "y", "s_kp", "s_kt"]
    layered = all("source_layer" in f.columns for f in frames) and any(
        f["source_layer"].notna().any() for f in frames)
    if not layered:
        return fuse(frames, weights, match_m=match_m)
    keys = sorted({(r, s) for f in frames for r, s in zip(f.region, f.source_layer) if pd.notna(s)})
    out = []
    for reg, lay in keys:
        sub = [f[(f.region == reg) & (f.source_layer == lay)] for f in frames]
        if not any(len(s) for s in sub):
            continue
        o = fuse(sub, weights, match_m=match_m)
        if len(o):
            out.append(o.assign(region=reg, source_layer=lay))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols + ["source_layer"])


def decode(F, s, geom, per_class, scenes=None):
    """Fused candidates → emitted points under one setting.

    threshold (per class) → framing prior (big scenes) → region-level kurgan gate → hillfort gate → hedge.
    """
    G = F[F.score >= F.cls.map(lambda c: per_class.get(c, s.thr))]
    if s.frame and geom:
        keep = np.ones(len(G), bool)
        for reg, idx in G.groupby("region").indices.items():
            g = geom.get(reg) or []
            if not g:
                continue
            a = np.array(g)
            sub = G.iloc[idx]
            d = np.hypot(sub.x.values[:, None] - a[:, 0], sub.y.values[:, None] - a[:, 1]) / np.maximum(a[:, 2], 1)
            j = d.argmin(1)
            far = (~a[j, 3].astype(bool)) & (d[np.arange(len(sub)), j] > s.frame[0])
            drop = far & (sub.score.values < s.frame[1])
            keep[idx] = ~drop
        G = G[keep]
    if s.gate is not None and len(G):
        best = G[G.cls.isin(K2)].groupby("region").score.max()
        dead = set(best[best < s.gate].index)
        if dead:
            G = G[~((G.region.isin(dead)) & G.cls.isin(K2))]
    centre = []
    if s.gor and scenes is not None and len(scenes):
        mode, g = s.gor
        c = scenes
        if mode == "prior":                       # ungated centre prior on every small scene
            add = c.assign(p=0.05)
        else:
            add = c[c.p >= g]
            if mode == "fill":                    # only where the fusion found no hillfort
                add = add[~add.region.isin(set(G[G.cls == "gorodishcha"].region))]
            elif mode == "swap":                  # and trust the detectors less than the scene classifier
                low = set(c[c.p < g].region)
                G = G[~((G.cls == "gorodishcha") & G.region.isin(low))]
        centre = [(r.region, "gorodishcha", float(r.p), r.x, r.y3857) for r in add.itertuples()]
    rows = list(centre)
    for r in G.itertuples():
        rows.append((r.region, r.cls, float(r.score), r.x, r.y))
        if s.hedge is not None and r.cls in K2:      # members disagree on damaged / intact: emit the other class too
            other = K2[1] if r.cls == K2[0] else K2[0]
            alt = getattr(r, "s_kt" if other == K2[1] else "s_kp", 0.0)
            if alt >= s.hedge:
                rows.append((r.region, other, float(r.score) * 0.99, r.x, r.y))
    return pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])


def grid(frame_prior=False, gor_gate=False, thrs=THRS, hedges=HEDGES, gates=GATES):
    frames = (None, FRAME) if frame_prior else (None,)
    gors = GOR_MODES if gor_gate else (None,)
    return [Spec(t, h, g, f, q) for t in thrs for h in hedges for g in gates for f in frames for q in gors]


def evaluate(F, specs, RF, folds, gt_by_fold, geom, per_class, scenes=None):
    """→ ({spec: {fold: counts}}, {spec: decoded frame})"""
    dec = {s: decode(F, s, geom, per_class, scenes) for s in specs}
    make = lambda s, f: fc(dec[s][dec[s].region.map(RF) == f])
    C = fold_counts(make, specs, folds, gt_by_fold, CLS)
    return C, dec


def site_counts(df, gt_by_region):
    out = {}
    for reg, g in df.groupby("region"):
        if reg not in gt_by_region:
            continue
        _, res = score(fc(g), gt_by_region[reg], CLS)
        out[reg] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
    return out


def per_class_table(tot, title):
    print(f"   {title}")
    for c in CLS:
        tp, fp, fn = tot.get(c, (0, 0, 0))
        if tp + fn:
            print(f"     {c:24s} F1 {2 * tp / max(2 * tp + fn + 0.5 * fp, 1):.3f}  tp {tp:4d}  fp {fp:4d}  fn {fn:4d}")


# --- main -----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", default="work/spor", help="directory holding oof_*_cands.parquet")
    ap.add_argument("--members", help="comma-separated member names (default: all discovered)")
    ap.add_argument("--drop", default="", help="comma-separated member names to remove")
    ap.add_argument("--list", action="store_true", help="list available members and their candidate counts")
    ap.add_argument("--alone", action="store_true", help="score every member on its own")
    ap.add_argument("--frame-prior", action="store_true", help="include the big-scene framing prior in the grid")
    ap.add_argument("--gor-gate", action="store_true", help="include the whole-scene hillfort gate in the grid")
    ap.add_argument("--scene-oof", default="work/spor/oof_scene_cls.parquet",
                    help="whole-scene hillfort probabilities (python -m koz3.scene_cls --oof)")
    ap.add_argument("--folds", help="optional folds.json (region__scene) or folds.csv (site,fold) to align folds")
    ap.add_argument("--ann", help="override data/spor_prep/ann.json")
    ap.add_argument("--match-m", type=float, default=8.0, help="fusion cluster radius, metres (deployed 8)")
    ap.add_argument("--arch-thr", type=float, default=PER_CLASS_DEFAULT["arkhitektury"], help="per-class threshold override")
    ap.add_argument("--accept", nargs="+", metavar="SPEC",
                    help="robust acceptance: first spec is the base, every later one is a treatment, "
                         "e.g. thr=0.25 thr=0.3,hedge=0.4 thr=0.25,gor=swap:0.1")
    ap.add_argument("--csv", help="write per-site counts of the honest setting here (default work/spor/oof_local_sites.csv)")
    a = ap.parse_args()
    per_class = dict(PER_CLASS_DEFAULT)
    per_class["arkhitektury"] = a.arch_thr

    avail = discover(a.work)
    if not avail:
        sys.exit(f"no candidate files under {a.work}/oof_*_cands.parquet — run\n"
                 f"  python -m koz3.sp_train --mod SpOr --oof --tag <tag>\n"
                 f"  python -m koz3.det_train --mod SpOr --oof --tag <tag>")
    if a.list:
        for n, p in avail.items():
            d = pd.read_parquet(p)
            print(f"{n:14s} {len(d):7d} candidates  {d.region.nunique():4d} regions  "
                  + ", ".join(f"{c}={v}" for c, v in d.cls.value_counts().items()))
        return

    names = [n.strip() for n in a.members.split(",")] if a.members else list(avail)
    drop = {n.strip() for n in a.drop.split(",") if n.strip()}
    missing = [n for n in names if n not in avail]
    if missing:
        sys.exit(f"unknown members {missing}; available: {sorted(avail)}")
    names = [n for n in names if n not in drop]
    if not names:
        sys.exit("no members left after --drop")

    def _cands(path):
        d = pd.read_parquet(path)
        keep = [c for c in ("region", "cls", "score", "x", "y", "source_layer") if c in d.columns]
        return d[keep]

    M = {n: _cands(avail[n]) for n in names}
    RF = folds_from(a.folds) if a.folds else region_folds(a.ann)
    covered = sorted(set().union(*[set(m.region) for m in M.values()]))
    regions = [r for r in covered if r in RF]
    if len(regions) < len(covered):
        print(f"warning: {len(covered) - len(regions)} candidate regions have no fold assignment and are dropped")
    if not regions:
        sys.exit("no region with both candidates and a fold")
    folds = sorted({RF[r] for r in regions})

    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(ROOT, "data", "labels_3857.parquet"))
    L = L[(L["mod"] == "SpOr") & L.region.isin(regions)]
    gt_by_fold = {f: to_gt_geojson(L[L.region.map(RF) == f]) for f in folds}
    gt_by_region = {r: to_gt_geojson(g) for r, g in L.groupby("region")}
    geom = scene_geometry(regions) if a.frame_prior else {}
    scenes = None
    sp_scene = os.path.join(ROOT, a.scene_oof)
    if os.path.exists(sp_scene):
        scenes = pd.read_parquet(sp_scene)
        scenes = scenes[scenes.region.isin(regions)]
        print(f"hillfort scenes: {len(scenes)} in {scenes.region.nunique()} regions, "
              f"{int(scenes.y.sum())} with a hillfort (OOF probabilities from {a.scene_oof})")
    elif a.gor_gate:
        sys.exit(f"--gor-gate needs {a.scene_oof} — run: python -m koz3.scene_cls --oof")

    print(f"{len(regions)} regions, {len(folds)} folds, members {names}")
    print("  GT per fold: " + ", ".join(f"{f}:{len(gt_by_fold[f]['features'])}" for f in folds))

    if a.alone:      # every member on its own, threshold only (the honest 'alone' number of the ledger)
        print("\nmember alone (threshold sweep only):")
        for n in names:
            F = fuse_members([M[n]], match_m=a.match_m)
            F = F[F.region.isin(regions)]
            specs = [Spec(t, None, None, None) for t in THRS]
            C, _ = evaluate(F, specs, RF, folds, gt_by_fold, geom, per_class, scenes)
            h, ch, o, os_, _ = nested(C)
            print(f"  {n:14s} honest {h:.4f}  oracle {o:.4f} @ {fmt(os_)}")

    F = fuse_members([M[n] for n in names], match_m=a.match_m)
    F = F[F.region.isin(regions)].reset_index(drop=True)
    print(f"fusion: {sum(len(m) for m in M.values())} member candidates → {len(F)} fused objects")

    specs = grid(a.frame_prior, a.gor_gate)
    if a.accept:
        specs = [parse_spec(t) for t in a.accept]
        print("acceptance base " + fmt(specs[0]) + "  |  treatments: "
              + ", ".join(fmt(s) for s in specs[1:]))
    C, dec = evaluate(F, specs, RF, folds, gt_by_fold, geom, per_class, scenes)
    h, ch, o, os_, tot = nested(C)
    print(f"\nnested honest {h:.4f}   oracle {o:.4f} @ {fmt(os_)}")
    print("chosen per fold: " + ", ".join(f"{f}:{fmt(ch[f])}" for f in folds))
    per_class_table(tot, "honest per class:")

    if a.accept:
        from robust_accept import report
        base = specs[0]
        sb = site_counts(dec[base], gt_by_region)
        for treat in specs[1:]:
            report(f"{fmt(treat)} vs {fmt(base)}", sb, site_counts(dec[treat], gt_by_region))
    else:
        # diagnostics on the honest decode: which regions carry the false alarms
        honest = pd.concat([dec[ch[f]][dec[ch[f]].region.map(RF) == f] for f in folds])
        fp = honest[honest.cls.isin(K2)].groupby("region").size()
        print("\nlargest kurgan emitters (points, honest setting): "
              + ", ".join(f"{r}:{n}" for r, n in fp.sort_values(ascending=False).head(8).items()))
        out = a.csv or os.path.join(ROOT, a.work, "oof_local_sites.csv")
        sc = site_counts(honest, gt_by_region)
        pd.DataFrame([(r, c, *v) for r, d in sc.items() for c, v in d.items()],
                     columns=["region", "cls", "tp", "fp", "fn"]).to_csv(out, index=False)
        print(f"per-site counts → {out}")


if __name__ == "__main__":
    main()
