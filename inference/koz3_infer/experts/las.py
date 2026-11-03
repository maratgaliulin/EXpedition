"""LiDAR expert: LAS → DTM (koz3.dtm) → relief channels → koz3.li_train U-Net ensemble → peaks."""
import os

import numpy as np

from ..common import log, utm_crs, oom_retry
from ..models import cfg, ts_files, load_ts

FAMILIES = ["las"]
_MODELS = None


def READY():
    return bool(ts_files("li"))


def _models():
    """[(model, extra)]: li_x_*.ts take the 12-channel relief (+ openness / sky-view, koz3.relief EXTRA)."""
    global _MODELS
    if _MODELS is None:
        wmap = cfg()["li"].get("weights", {})          # file name → weight (validated blend), default 1
        kind = lambda b: "u" if b.startswith("li_u_") else ("x" if b.startswith("li_x_") else "")   # u: unclipped 12-ch
        _MODELS = [(load_ts(w), kind(os.path.basename(w)), float(wmap.get(os.path.basename(w), 1.0))) for w in ts_files("li")]
        log(f"li: {len(_MODELS)} models, kinds {[k or '9ch' for _, k, _ in _MODELS]}, weights {[w for _, _, w in _MODELS]}")
    return _MODELS


def _channels_extra(d, v, res, unclip=False):
    """12-channel relief exactly as in training (koz3.relief with EXTRA on; UNCLIP for li_u_ models), then restored."""
    from koz3 import relief
    e0, n0, u0 = relief.EXTRA, relief.N_CH, getattr(relief, "UNCLIP", False)
    relief.EXTRA, relief.N_CH, relief.UNCLIP = True, 12, unclip
    try:
        return relief.channels_tiled(d, v, res)[0]
    finally:
        relief.EXTRA, relief.N_CH, relief.UNCLIP = e0, n0, u0


def predict(region, paths, deadline):
    import time
    from koz3 import dtm, relief, hm
    from koz3.li_train import decode_site_v2, N_HM, N_MK
    c = cfg()["li"]
    feats = []
    for p in paths:
        t0 = time.time()
        d, v, (x0, ytop, res) = dtm.build(str(p))
        x, vv = relief.channels_tiled(d, v, res)
        models = _models()
        if time.time() > deadline - 120:
            models = models[:1]      # protect the time budget on huge clouds
        xs = {"": x}
        if any(k == "x" for _, k, _ in models):
            xs["x"] = _channels_extra(d, v, res)
        if any(k == "u" for _, k, _ in models):
            xs["u"] = _channels_extra(d, v, res, unclip=True)
        tta = c.get("tta", True)        # "d8": 4 rotations × 2 flips (nested OOF 0.5651 → 0.5695); True: 3 flips
        if tta == "d8" and time.time() > deadline - 300:
            tta = True                     # protect the time budget
        scales = c.get("scale_tta")     # [[scale, weight], ...] pyramid TTA (held-out fold 0: 0.4719 → 0.5046);
        if scales and time.time() > deadline - 300 * max(1, len(scales) - 1):
            scales = None                   # drop the extra scales rather than overrun the 30-minute budget
        ps = [oom_retry(lambda t, m=m, xx=xs[k]: hm.predict_tiled(m, xx, N_HM + N_MK, tile=t, overlap=min(192, t // 4),
                                                                  tta=tta, scales=scales))
              for m, k, _ in models]
        ws = np.array([w for _, _, w in models], np.float32)
        prob = np.tensordot(ws / ws.sum(), np.stack(ps), axes=1)
        meta = {"x0": x0, "ytop": ytop, "res": relief.NET_RES, "crs": utm_crs(region), "region": region.name}
        fs = decode_site_v2(prob, vv, meta, thr=c["thr"], dual=c["dual"], comp_thr=c["comp_thr"], fort_min_px=c["fort_min_px"], gor_min_px=c["gor_min_px"], r_out=c["r_out"], prior=c.get("nbr_prior"))
        for f in fs:
            f["properties"].update({"sub_region_name": "", "crs": "urn:ogc:def:crs:EPSG::3857"})
        feats += fs
        log(f"li {p.name}: dtm {d.shape} res {res} → {len(fs)} feats {time.time() - t0:.0f}s")
    return feats
