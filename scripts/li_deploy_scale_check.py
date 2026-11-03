"""Deployed-path check of the LiDAR scale pyramid: the shipped TorchScript models, on GT regions, from the raw LAS.

Unlike scripts/li_scale_tta_oof.py (a fold model trained here, honest but not the shipped weights), this runs exactly
what `inference/koz3_infer/experts/las.py` runs — weights, d8 TTA, 12-channel relief for `li_x_*`, neighbour prior,
decode_site_v2 — and reports wall time, because the pyramid multiplies LiDAR inference cost.

  python scripts/li_deploy_scale_check.py --models archive/koz3_submission_v20_layers_ae_fort_20261027/inference/models \
      --regions 104_Шетнево,022_Лихуша --scales 1.0,0.8,1.25
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3 import dtm, hm, li_train, relief  # noqa: E402
from koz3.labels import DATA, to_gt_geojson  # noqa: E402
from koz3.score import score  # noqa: E402


def channels(d, v, res, extra=False, unclip=False):
    e0, n0, u0 = relief.EXTRA, relief.N_CH, getattr(relief, "UNCLIP", False)
    relief.EXTRA, relief.N_CH, relief.UNCLIP = extra, 12 if extra else 9, unclip
    try:
        return relief.channels_tiled(d, v, res)          # (channels, valid at NET_RES)
    finally:
        relief.EXTRA, relief.N_CH, relief.UNCLIP = e0, n0, u0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--scales", default="1.0,0.8,1.25")
    a = ap.parse_args()
    c = json.load(open(os.path.join(a.models, "config.json")))["li"]
    names = {os.path.basename(w): float(v) for w, v in c.get("weights", {}).items()}
    md = [(torch.jit.load(os.path.join(a.models, "li", w), map_location="cuda").float().eval(), w, wt)
          for w, wt in names.items()]          # stored fp16, run fp32 (+autocast): same as models.load_ts
    scales = [(float(s), 1.0) for s in a.scales.split(",")]
    L = li_train.labels()
    want = [r.strip() for r in a.regions.split(",")]
    for reg in want:
        las = glob.glob(os.path.join(DATA, reg, "01_*", "*.las"))[0]
        t0 = time.time()
        d, v, (x0, ytop, res) = dtm.build(las)
        x9, v9 = channels(d, v, res)
        xs = {"9": x9}
        if any("li_x_" in w for _, w, _ in md):
            xs["x"] = channels(d, v, res, extra=True)[0]
        if any("li_u_" in w for _, w, _ in md):
            xs["u"] = channels(d, v, res, extra=True, unclip=True)[0]
        t_ch = time.time() - t0
        meta = {"x0": x0, "ytop": ytop, "res": relief.NET_RES, "crs": dtm.load(os.path.join(ROOT, "data", "las_prep", reg + ".npz"))[3], "region": reg}
        masks = {"9": v9, "x": v9, "u": v9}
        for arm, sc in (("base", None), ("pyramid", scales)):
            t1 = time.time()
            ps = []
            for m, w, wt in md:
                k = "x" if w.startswith("li_x_") else ("u" if w.startswith("li_u_") else "9")
                tta = c.get("tta", True)
                ps.append(hm.predict_tiled(m, xs[k], li_train.N_HM + li_train.N_MK, tta=tta, scales=sc))
            ws = np.array([wt for _, _, wt in md], np.float32)
            prob = np.tensordot(ws / ws.sum(), np.stack(ps), axes=1)
            fs = li_train.decode_site_v2(prob, masks["9"], meta, thr=c["thr"], dual=c["dual"], comp_thr=c["comp_thr"],
                                         fort_min_px=c["fort_min_px"], gor_min_px=c["gor_min_px"], r_out=c["r_out"],
                                         prior=c.get("nbr_prior"))
            gts = [f for f in to_gt_geojson(L[L.region == reg])["features"]]
            _, r = score({"type": "FeatureCollection", "features": fs},
                         {"type": "FeatureCollection", "features": gts})
            from koz3.nested import weighted
            print(f"{reg:26s} {arm:8s} {len(fs):5d} feats  {time.time() - t1:6.1f}s  "
                  f"weighted {weighted({k: (v['tp'], v['fp'], v['fn']) for k, v in r.items()}):.4f}  "
                  f"(dtm+channels {t_ch:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
