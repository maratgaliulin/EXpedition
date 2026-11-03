"""Ae dense-heatmap OOF with a sheet-centre framing prior — does the prior rescue the heatmap that scored 0.026?

The rejected Ae heatmap (sp_train --mod Ae) fires everywhere on grey film. Measured on the training sheets, Ae labels
sit tightly around the sheet centre (median d90/half-side is well under 1), because the sheets were cut around the site
and only the site is annotated. The satellite expert already exploits exactly this (`frame_prior`); the Ae heatmap never
did. This script separates the two effects:

  1. infer: decode candidates at a low threshold from the fold model, keep each candidate's distance to its sheet centre;
  2. sweep: nested-honest selection over (threshold, prior radius) — radius as a fraction of the sheet half-side,
     inf = the previous behaviour (no prior).

Candidates are saved, so the sweep can be re-run with other grids without touching the GPU.

  python scripts/ae_hm_prior_oof.py --infer --tag ""
  python scripts/ae_hm_prior_oof.py --sweep --tag ""
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import cv2  # noqa: E402
import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from koz3 import hm, sp_train  # noqa: E402
from koz3.labels import DATA, to_gt_geojson  # noqa: E402

WORK = os.path.join(ROOT, "work", "ae")
KURGANS = ("kurgany_povrezhdennye", "kurgany_tselye")


def sheet_geometry():
    """(region, scene) → (centre x/y in 3857, half-side in metres) from the source tif."""
    tr = Transformer.from_crs("EPSG:3857", "EPSG:3857", always_xy=True)
    out = {}
    for f in glob.glob(os.path.join(DATA, "*", "03_*", "*.tif")):
        region = os.path.relpath(f, DATA).split(os.sep)[0]
        scene = os.path.basename(f)[:-4]
        with rasterio.open(f) as s:
            crs = s.crs if (s.crs is not None and s.crs.to_epsg()) else None
            b = s.bounds
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            half = max(b[2] - b[0], b[3] - b[1]) / 2
            t = Transformer.from_crs(crs, 3857, always_xy=True)
            X, Y = t.transform(cx, cy)
            k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(Y / 6378137.0)) - np.pi / 2)))
        out[f"{region}|{scene}"] = (X, Y, half * k)
    return out


def infer(tag):
    import torch  # noqa: F401  (load_model needs cuda)
    chips = sp_train.load_chips("Ae")
    geom = sheet_geometry()
    os.makedirs(WORK, exist_ok=True)
    rows = []
    for fold in sorted({c["fold"] for c in chips}):
        m = sp_train.load_model(os.path.join(WORK, f"hm{tag}_f{fold}.pt"))
        feats = []
        for c in [c for c in chips if c["fold"] == fold]:
            rgb = cv2.imread(os.path.join(sp_train.prep_dir("Ae"), "chips", c["file"]))[:, :, ::-1]
            x, v = sp_train.norm_img(rgb)
            prob = hm.predict_tiled(m, x, sp_train.N_HM + sp_train.N_MK, tile=1024, overlap=0)
            feats += sp_train.decode_chip(prob, v, c["transform"], c["crs"], c["region"], "ae", thr=0.05, dual=1.1,
                                          scene=c["scene"])
        for f in sp_train.dedupe(feats):
            p = f["properties"]
            g = shape(f["geometry"]).centroid
            key = f"{p['region_name']}|{p.get('source_layer')}"
            if key not in geom:
                continue
            X, Y, half = geom[key]
            rows.append(dict(region=p["region_name"], scene=p.get("source_layer"), cls=p["class_name"],
                             score=p["confidence"], x=g.x, y=g.y, fold=fold,
                             d=float(np.hypot(g.x - X, g.y - Y)), half=float(half)))
        print("inferred fold", fold, len(rows), flush=True)
    D = pd.DataFrame(rows)
    D.to_parquet(os.path.join(WORK, f"cands{tag}.parquet"))
    print(len(D), "candidates →", os.path.join(WORK, f"cands{tag}.parquet"))


def sweep(tag, thrs, radii):
    from koz3.nested import fold_counts, nested
    D = pd.read_parquet(os.path.join(WORK, f"cands{tag}.parquet"))
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    L = L[(L["mod"] == "Ae") & L.class_name.isin(sp_train.CLS)]
    folds = sorted(D.fold.unique())
    gt_by_fold = {f: to_gt_geojson(L[L.region.isin(D[D.fold == f].region.unique())]) for f in folds}

    def make(sett, f):
        from shapely.geometry import Point
        thr, r = sett
        d = D[(D.fold == f) & (D.score >= thr) & (D.d <= r * D.half)]
        feats = []
        for row in d.itertuples():
            k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(row.y / 6378137.0)) - np.pi / 2)))
            feats.append({"type": "Feature",
                          # the metric only matches Polygon predictions: same 0.5 m circle decode_chip emits
                          "geometry": Point(row.x, row.y).buffer(0.5 * k, 12).__geo_interface__,
                          "properties": {"class_name": row.cls, "region_name": row.region,
                                         "markup_type": "ae", "confidence": row.score}})
        return {"type": "FeatureCollection", "features": feats}
    settings = [(t, r) for t in thrs for r in radii]
    C = fold_counts(make, settings, folds, gt_by_fold, sp_train.CLS)
    h, ch, o, os_, tot = nested(C)
    print(f"Ae heatmap honest {h:.4f}  (oracle {o:.4f} @ {os_})  chosen per fold {ch}")
    for c, (tp, fp, fn) in tot.items():
        if tp + fn or fp:
            print(f"   {c:24s} F1 {2 * tp / max(2 * tp + fn + 0.5 * fp, 1):.3f} tp {tp} fp {fp} fn {fn}")
    print("\nper-setting pooled (all folds, for reference — selection must stay nested):")
    for (t, r), cell in C.items():
        allc = {}
        for f in folds:
            for c, v in cell[f].items():
                a = allc.get(c, (0, 0, 0)); allc[c] = tuple(np.add(a, v))
        from koz3.nested import weighted
        kt = sum(v[0] for c, v in allc.items() if c in KURGANS)
        print(f"   thr {t:4.2f} r {r:5.2f}·half  pooled {weighted(allc):.4f}  kurgan tp {kt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--thrs", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    ap.add_argument("--radii", default="inf,1.0,0.7,0.5,0.35")
    a = ap.parse_args()
    if a.infer:
        infer(a.tag)
    if a.sweep:
        sweep(a.tag, [float(x) for x in a.thrs.split(",")], [float(x) for x in a.radii.split(",")])
