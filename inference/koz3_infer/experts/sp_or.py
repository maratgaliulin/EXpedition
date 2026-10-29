"""Satellite expert: D-FINE-L + RT-DETRv2 (previous team, their 0.5 m UTM scene representation) + our heatmap U-Net
(our chip representation), fused per object (koz3.sp_fuse, equal weights) → threshold → small circles.
Honest OOF of exactly this recipe: 0.2729 (nested over 4 site folds)."""
import os

import numpy as np
import pandas as pd

from ..common import log, utm_crs, to_feature, circle
from ..models import cfg, ts_files, load_ts
from . import optical

FAMILIES = ["sp_or"]
NAME = "sp_or"
PREV_CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
_DET = {}


def _det_weights(name):
    from ..models import model_dir
    d = model_dir(name)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("w_") and f.endswith(".pt")) if os.path.isdir(d) else []


def READY():
    return bool(_det_weights("sp_dfine") or _det_weights("sp_rtdetr") or optical.weights(NAME))


def _det(name):
    """Eager HF detectors rebuilt offline from the saved config + fp16 fold weights."""
    if name not in _DET:
        import torch
        from transformers import AutoConfig, AutoModelForObjectDetection
        from koz3.det_train import _Wrap
        from ..models import model_dir
        ws = _det_weights(name)
        _DET[name] = []
        if ws:
            cfg_ = AutoConfig.from_pretrained(model_dir(name)); cfg_.use_pretrained_backbone = False
            for w in ws:
                m = AutoModelForObjectDetection.from_config(cfg_)
                m.load_state_dict({k: v.float() for k, v in torch.load(w, map_location="cpu").items()})
                _DET[name].append(_Wrap(m.cuda().eval()).eval())
        log(f"{name}: {len(_DET[name])} models")
    return _DET[name]


def _to_rows(dets, T_crs, region):
    from pyproj import Transformer
    if not dets:
        return pd.DataFrame(columns=["region", "cls", "score", "x", "y"])
    tr = Transformer.from_crs(T_crs, 3857, always_xy=True)
    c = [e.centroid for _, e, _ in dets]
    X, Y = tr.transform([p.x for p in c], [p.y for p in c])
    return pd.DataFrame({"region": region, "cls": [PREV_CLASSES[k] for k, _, _ in dets], "score": [s for _, _, s in dets], "x": X, "y": Y})


def _scene_gate(region, path):
    """Whole-scene hillfort probability for small screenshots (koz3/scene_cls.py preprocessing: long side 384,
    centred zero pad, ImageNet norm, mean of identity / h-flip / v-flip) → (p, centre x, centre y in EPSG:3857)
    or None for big scenes / no model. Honest OOF: AUC 0.892."""
    import cv2
    import rasterio
    import torch
    from pyproj import Transformer
    from koz3 import sp_io
    from koz3.sp_train import MEAN, STD
    from ..common import utm_crs
    from ..models import load_ts, ts_files
    ws = ts_files("sp_scene")
    if not ws:
        return None
    if "sp_scene" not in _GATE:
        _GATE["sp_scene"] = [load_ts(w) for w in ws]
    with rasterio.open(str(path)) as src:
        if not sp_io.is_small(src):
            return None
        crs = src.crs.to_string() if (src.crs is not None and src.crs.to_epsg()) else utm_crs(region)
        rgb, _, t = sp_io.read_small(src)
    S = 384; h, w = rgb.shape[:2]; f = S / max(h, w)
    im = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    pad = np.zeros((S, S, 3), np.uint8); oy, ox = (S - im.shape[0]) // 2, (S - im.shape[1]) // 2
    pad[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    x = torch.from_numpy(((pad.astype(np.float32).transpose(2, 0, 1) - MEAN) / STD).astype(np.float32))[None].cuda()
    with torch.no_grad():
        p = float(np.mean([torch.sigmoid(m(x.flip(fl) if fl else x)).item() for m in _GATE["sp_scene"] for fl in ((), (-1,), (-2,))]))
    X, Y = Transformer.from_crs(crs, 3857, always_xy=True).transform(*(t * (w / 2, h / 2)))
    return p, X, Y


_GATE = {}


def _predict_pooled(region, paths, deadline):
    import time
    from koz3 import prev_sp_io, prev_sp_det
    from koz3.sp_fuse import fuse
    from shapely.geometry import shape
    c = cfg()["sp_or"]
    feats = []
    gates = []
    scenes_geo = []
    members_by_model = {}           # all scenes of the region pooled per model, then ONE fusion (as in the OOF study:
    for p in paths:                 # overlapping scenes of one data type must give one object, not duplicates)
        t0 = time.time()
        members = []
        arr = None
        for name in ("sp_dfine", "sp_rtdetr", "sp_dfine2", "sp_dfine3", "sp_dfine4"):   # 2-4: local D-FINE-L no denoising (3, 4: site-balanced + quality aug, seeds 2 / 3); separate members
            ms = _det(name)
            if ms:
                if arr is None:
                    arr, T, crs, _ = prev_sp_io.read_scene(str(p), 0.5)
                    H_, W_ = arr.shape[1:]                   # scene framing geometry (EPSG:3857), as in sp_centre_prior_oof
                    from pyproj import Transformer as _Tr
                    _t = _Tr.from_crs(crs.to_string(), 3857, always_xy=True)
                    (x0_, y0_), (x1_, y1_) = _t.transform(*(T * (0, 0))), _t.transform(*(T * (W_, H_)))
                    scenes_geo.append(((x0_ + x1_) / 2, (y0_ + y1_) / 2, max(abs(x1_ - x0_), abs(y1_ - y0_)) / 2, W_ * H_ < 1_000_000))
                members.append(_to_rows(prev_sp_det.detect_scene(ms, arr, T, bs=4, time_left=lambda: deadline - time.time()), crs.to_string(), region.name))
        if optical.weights(NAME):
            hf = optical.predict_scene(NAME, region, p, deadline, thr=0.05, dual=0.3)
            members.append(pd.DataFrame([{"region": region.name, "cls": f["properties"]["class_name"],
                                          "score": f["properties"]["confidence"], "x": shape(f["geometry"]).centroid.x,
                                          "y": shape(f["geometry"]).centroid.y} for f in hf], columns=["region", "cls", "score", "x", "y"]))
        for i, m in enumerate(members):
            members_by_model.setdefault(i, []).append(m)
        if c.get("gate_thr") is not None:
            try:
                gate = _scene_gate(region, p)
                if gate is not None:
                    gates.append(gate)
                    log(f"sp_or {p.name}: hillfort scene p={gate[0]:.3f}")
            except Exception as e:                                   # the gate is optional: never lose the scene
                log(f"sp_or {p.name}: scene gate failed {e!r}")
        log(f"sp_or {p.name}: {sum(len(m) for m in members)} member candidates {time.time() - t0:.0f}s")
    members = [pd.concat(v, ignore_index=True) for _, v in sorted(members_by_model.items())]
    if not members:
        return feats
    Fz = fuse(members, np.ones(len(members)), match_m=c.get("match_m", 8.0))
    thr_cls = c.get("fused_thr_cls", {})            # per-class fused thresholds (nested: architecture 0.15 in every fold)
    Fz = Fz[Fz.score >= Fz.cls.map(lambda k: thr_cls.get(k, c["fused_thr"]))]
    fr = c.get("frame_prior")                       # big scenes are framed on the site: far detections need a higher score
    if fr and scenes_geo and len(Fz):
        g = np.array([q[:3] for q in scenes_geo]); sm = np.array([q[3] for q in scenes_geo])
        d = np.hypot(Fz.x.values[:, None] - g[:, 0], Fz.y.values[:, None] - g[:, 1]) / np.maximum(g[:, 2], 1)
        j = d.argmin(1); far = (~sm[j]) & (d[np.arange(len(Fz)), j] > fr["x"])
        Fz = Fz[~far | (Fz.score.values >= fr["t2"])]
    if gates:                       # scripts/sp_scene_gate_oof.py "swap": confident scene → centre hillfort (conf = p);
        g = c["gate_thr"]           # any small scene below the gate → drop the detectors' hillforts in this region
        add = [(region.name, "gorodishcha", p_, X, Y) for p_, X, Y in gates if p_ >= g]
        if any(p_ < g for p_, _, _ in gates):
            Fz = Fz[Fz.cls != "gorodishcha"]
        Fz = pd.concat([Fz, pd.DataFrame(add, columns=["region", "cls", "score", "x", "y"])], ignore_index=True)
    hedge = c.get("kurgan_hedge")                   # members disagree on damaged / intact → emit the other class too
    K2 = ("kurgany_povrezhdennye", "kurgany_tselye")
    for n, r in enumerate(Fz.itertuples()):
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(r.y / 6378137.0)) - np.pi / 2)))
        emit = [(r.cls, r.score)]
        if hedge is not None and r.cls in K2 and "s_kp" in Fz.columns:
            other = K2[1] if r.cls == K2[0] else K2[0]
            if (r.s_kt if other == K2[1] else r.s_kp) >= hedge:
                emit.append((other, r.score * 0.99))
        for cls_, sc_ in emit:
            ft = to_feature(circle(r.x, r.y, c["r_out"] * k, 12), cls_, sc_, region.name, "sp_or")
            if ft:
                ft["properties"]["image"] = f"spobj{n}"     # one fused object (already merged across scenes): the hedge
                feats.append(ft)                            # pair must survive region dedupe
    log(f"sp_or {region.name}: {len(paths)} scene(s) → {len(Fz)} fused objects")
    return feats


def predict(region, paths, deadline):
    """Run every source image as an independent reference layer.

    The organisers clarified that two views of the same physical object are two
    reference objects.  Model/member fusion remains within one image; detections
    are never fused across source images.
    """
    if len(paths) <= 1:
        return _predict_pooled(region, paths, deadline)
    feats = []
    for path in paths:
        one = _predict_pooled(region, [path], deadline)
        for ft in one:
            # Source image is part of the target identity.  Keep the existing
            # per-object image id so within-image duplicate removal is unchanged;
            # dedupe_region uses source_layer to isolate independent images.
            ft["properties"]["source_layer"] = path.name
        feats.extend(one)
    return feats
