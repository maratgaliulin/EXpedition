"""Held-out LiDAR study of decode-side levers that do not need retraining: scale-pyramid TTA.

The deployed LiDAR expert runs one resolution (NET_RES = 0.5 m). Training augments scale by stretching the ground
x0.7-1.45, so the network has seen resampled terrain and averaging its answers over a small scale pyramid is a free
(spatially denser) vote. The ledger already rejects rotation TTA (kept: d8), NMS/centroid tweaks, anchors, per-site
score normalisation and the neighbourhood prior is already deployed; scale-pyramid TTA is the remaining untried one.

Measured on the LiDAR fold LEFT OUT of training, so the number is honest: same decode (koz3.li_train.decode_site_v2,
deployed thresholds) for every arm, only the probability map changes.

  LI_AUG=1 RELIEF_EXTRA=1 python scripts/li_scale_tta_oof.py --fold 0 --tag _x0 --scales 1.0,0.8,1.25
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3 import hm, li_train  # noqa: E402
from koz3.labels import to_gt_geojson  # noqa: E402
from koz3.nested import _add, _add_all, weighted  # noqa: E402
from koz3.score import score  # noqa: E402

CFG = dict(thr=0.15, dual=0.05, comp_thr=0.5, fort_min_px=30, gor_min_px=600)


def resize(x, s):
    """channel stack (C,H,W) → scale s (bilinear), the terrain-scale analogue of the training zoom augmentation."""
    if s == 1.0:
        return x, (x.shape[1], x.shape[2])
    t = torch.from_numpy(np.ascontiguousarray(x))[None]
    h, w = int(round(x.shape[1] * s)), int(round(x.shape[2] * s))
    return torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0].numpy(), (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="_x0")
    ap.add_argument("--scales", default="1.0,0.8,1.25")
    ap.add_argument("--tta", default="flip")          # flip | d8 | none (same for every arm)
    a = ap.parse_args()

    scales = [float(s) for s in a.scales.split(",")]
    L = li_train.labels()
    items = [it for it in li_train.site_items(L) if it["fold"] == a.fold and it["objs"]]
    print(f"fold {a.fold}: {len(items)} held-out LiDAR regions, {sum(len(it['objs']) for it in items)} objects", flush=True)
    model = li_train.load_model(os.path.join(li_train.RUNS, f"li{a.tag}_f{a.fold}.pt"))

    arms = {"base": [1.0], "pyramid": scales}
    per_arm = {k: {} for k in arms}
    for it in items:
        x = np.asarray(it["x"], np.float32)
        meta = dict(it["meta"], region=it["region"])
        probs = {}
        for s in sorted(set(scales)):
            xs, shape = resize(x, s)
            p = hm.predict_tiled(model, xs, li_train.N_HM + li_train.N_MK, tta=a.tta)
            if s != 1.0:
                p = torch.nn.functional.interpolate(torch.from_numpy(p)[None], size=x.shape[1:],
                                                    mode="bilinear", align_corners=False)[0].numpy()
            probs[s] = p.astype(np.float32)
        for s in sorted(set(scales)):      # keep per-scale maps so new scale sets can be decoded without re-running
            np.save(os.path.join(li_train.RUNS, f"scale_s{s:g}_{it['region']}.npy"), probs[s].astype(np.float16))
        for arm, ss in arms.items():
            p = np.mean([probs[s] for s in ss], axis=0)
            fs = li_train.decode_site_v2(p, it["valid"], meta, r_out=1.0, prior=dict(R=200, K=3, snb=0.2), **CFG)
            per_arm[arm][it["region"]] = fs
        print("done", it["region"], flush=True)

    gt = to_gt_geojson(L[L.region.isin(list(per_arm["base"]))])
    gts = {f["properties"]["region_name"]: [] for f in gt["features"]}
    for f in gt["features"]:
        gts[f["properties"]["region_name"]].append(f)
    res = {}
    for arm, feats in per_arm.items():
        cnt = {}
        for reg, fs in feats.items():
            _, r = score({"type": "FeatureCollection", "features": fs},
                         {"type": "FeatureCollection", "features": gts[reg]})
            cnt[reg] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in r.items()}
        res[arm] = cnt
    for arm, cnt in res.items():
        tot = _add_all(cnt.values())
        print(f"\n== {arm}  pooled weighted {weighted(tot):.4f}")
        for c, (tp, fp, fn) in tot.items():
            if tp + fn or fp:
                print(f"   {c:24s} F1 {2 * tp / max(2 * tp + fn + 0.5 * fp, 1):.3f} tp {tp} fp {fp} fn {fn}")
    if "base" in res and "pyramid" in res:
        d = {reg: weighted(res["pyramid"][reg]) - weighted(res["base"][reg]) for reg in res["base"]}
        print(f"\nregions better with pyramid: {sum(1 for v in d.values() if v > 1e-9)}/{len(d)}")
        for reg in sorted(d, key=d.get):
            print(f"   {reg:36s} base {weighted(res['base'][reg]):.4f} → pyramid {weighted(res['pyramid'][reg]):.4f}")
    json.dump({k: {r: v for r, v in d.items()} for k, d in res.items()},
              open(os.path.join(li_train.RUNS, "scale_tta_counts.json"), "w"))


if __name__ == "__main__":
    main()
