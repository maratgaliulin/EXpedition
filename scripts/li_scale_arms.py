"""Re-score arbitrary scale-mixture arms from the per-scale maps saved by scripts/li_scale_tta_oof.py.

Decoding is the expensive part of nothing — the maps are on disk — so mixed arms (weights, subsets) can be tried without
touching the GPU. Same held-out sites, same decode as deployment.

  python scripts/li_scale_arms.py --fold 0 --arms base=1.0 pyramid=1.0,0.8,1.25 wide=1.0,0.75,1.0x1.5,1.5
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3 import li_train  # noqa: E402
from koz3.labels import to_gt_geojson  # noqa: E402
from koz3.nested import _add_all, weighted  # noqa: E402
from koz3.score import score  # noqa: E402

CFG = dict(thr=0.15, dual=0.05, comp_thr=0.5, fort_min_px=30, gor_min_px=600)


def parse(arm):
    """'1.0x2,0.8' → [(1.0,2.0),(0.8,1.0)] (scale x weight)."""
    out = []
    for tok in arm.split(","):
        s, _, w = tok.partition("x")
        out.append((float(s), float(w or 1.0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--thrs", default="")        # e.g. 0.1,0.15,0.2,0.3 — otherwise the deployed 0.15 only
    ap.add_argument("--duals", default="")
    a = ap.parse_args()
    thrs = [float(t) for t in a.thrs.split(",")] if a.thrs else [CFG["thr"]]
    duals = [float(d) for d in a.duals.split(",")] if a.duals else [CFG["dual"]]
    L = li_train.labels()
    items = [it for it in li_train.site_items(L) if it["fold"] == a.fold and it["objs"]]
    gt = to_gt_geojson(L[L.region.isin([it["region"] for it in items])])
    gts = {}
    for f in gt["features"]:
        gts.setdefault(f["properties"]["region_name"], []).append(f)

    res = {}
    for spec in a.arms:
        name, _, scales = spec.partition("=")
        mix = parse(scales)
        cnt = {}
        for it in items:
            maps = [np.load(os.path.join(li_train.RUNS, f"scale_s{s:g}_{it['region']}.npy")).astype(np.float32)
                    for s, _ in mix]
            w = np.array([w for _, w in mix], np.float32); w /= w.sum()
            p = np.tensordot(w, np.stack(maps), axes=1)
            meta = dict(it["meta"], region=it["region"])
            for t in thrs:
                for d in duals:
                    key = f"{name}@t{t:g}d{d:g}"
                    fs = li_train.decode_site_v2(p, it["valid"], meta, r_out=1.0, prior=dict(R=200, K=3, snb=0.2),
                                                 **{**CFG, "thr": t, "dual": d})
                    _, r = score({"type": "FeatureCollection", "features": fs},
                                 {"type": "FeatureCollection", "features": gts[it["region"]]})
                    cnt.setdefault(key, {})[it["region"]] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in r.items()}
        res[name] = cnt
    base = res.get(a.arms[0].partition("=")[0])
    for name, cnt in res.items():
        for key, cell in cnt.items():
            tot = _add_all(cell.values())
            d = ""
            if base is not None and name != a.arms[0].partition("=")[0]:
                bkey = key.replace(name, a.arms[0].partition("=")[0], 1)
                bcell = base.get(bkey)
                if bcell:
                    wins = sum(1 for r in bcell if weighted(cell[r]) > weighted(bcell[r]) + 1e-9)
                    d = (f"  (Δ {weighted(tot) - weighted(_add_all(bcell.values())):+.4f}, "
                         f"better on {wins}/{len(bcell)} sites)")
            print(f"{key:26s} pooled {weighted(tot):.4f}{d}")
            for c, (tp, fp, fn) in tot.items():
                if tp + fn or fp:
                    print(f"   {c:24s} F1 {2 * tp / max(2 * tp + fn + 0.5 * fp, 1):.3f} tp {tp} fp {fp} fn {fn}")
    json.dump({n: {k: v for k, v in c.items()} for n, c in res.items()},
              open(os.path.join(li_train.RUNS, "scale_arms_counts.json"), "w"), default=str)


if __name__ == "__main__":
    main()
