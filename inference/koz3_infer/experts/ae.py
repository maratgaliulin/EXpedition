"""Ae (1940s German aerial film) expert.

Our Ae-trained heatmap failed out-of-fold (honest 0.026: it fires everywhere on grey film), so Ae uses the satellite
D-FINE models (trained on SpOr only, greyscale-augmented) on the Ae sheet resampled to 0.5 m, keeping only damaged
kurgans at confidence ≥ 0.7: nested OOF on Ae labels pooled with the LiDAR + satellite experts 0.3085 → 0.3103
(the other classes lowered the pooled score). LiDAR twins (koz3.fuse) add co-located LiDAR detections."""
import numpy as np

from ..common import log, to_feature, circle
from ..models import cfg

FAMILIES = ["ae"]


def READY():
    from .sp_or import _det_weights
    return bool(_det_weights("sp_dfine")) or bool(_ae_weights())


def _ae_weights(name="ae_det"):
    import os
    from ..models import model_dir
    d = model_dir(name)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")) if os.path.isdir(d) else []


_AE = {}


def predict_ae_det(region, paths, deadline, c, name="ae_det", res=0.5, markup="ae"):
    """Ae-trained detector (koz3/ae_det_train.py): the sheet is cut exactly like data/ae_prep (koz3.sp_io: small scene
    whole, else 0.5 m grid in 1024 px tiles with 256 px overlap), detected per tile (grey, 640 px windows, flip TTA),
    tiles merged within 4 m (koz3.sp_fuse), per-class thresholds from the nested study (scripts/ae_det_oof.py)."""
    import time
    import pandas as pd
    import rasterio
    from pyproj import Transformer
    from koz3 import sp_io
    from koz3.ae_det_train import detect_image, load_model
    from koz3.sp_fuse import fuse
    from koz3.sp_train import CLS
    from ..common import utm_crs
    if name not in _AE:
        _AE[name] = [load_model(w) for w in _ae_weights(name)]
    thr = c["class_thr"]
    deadline = min(deadline, time.time() + c.get("max_s", 240))     # per-region cap: never starve the others
    rows = []
    for p in paths:
        t0 = time.time(); n0 = len(rows)
        with rasterio.open(str(p)) as src:
            crs = src.crs.to_string() if (src.crs is not None and src.crs.to_epsg()) else utm_crs(region)
            tr = Transformer.from_crs(crs, 3857, always_xy=True)
            if sp_io.is_small(src):
                chips = [sp_io.read_small(src)]
            else:
                W, H, tg = sp_io.big_grid(src, res)
                try:
                    raw = sp_io.read_grid_raw(src, W, H)      # one read, tiles sliced in memory (pixel parity 0.1/255)
                except Exception:
                    raw = None
                chips = []
                for x, y, w, h in sp_io.tiles(W, H):
                    if time.time() > deadline:
                        log(f"{markup} {p.name}: deadline, {len(chips)} tiles"); break
                    try:
                        rgb, valid = sp_io.tile_from_grid(raw, x, y, w, h) if raw is not None else sp_io.read_window_resampled(src, x, y, w, h, W, H)
                    except rasterio.errors.RasterioIOError:
                        continue
                    if valid.mean() < 0.1:
                        continue
                    chips.append((rgb, valid, tg * rasterio.Affine.translation(x, y)))
            for rgb, valid, t in chips:
                if time.time() > deadline:
                    break
                for m in _AE[name]:
                    for cl, cx, cy, _, _, s in detect_image(m, rgb, fast=c.get("fast", True)):
                        if CLS[cl] in thr and s >= thr[CLS[cl]]:
                            X, Y = tr.transform(*(t * (cx, cy)))
                            rows.append((region.name, CLS[cl], s, X, Y, p.name))
        log(f"{markup} {p.name}: {len(chips)} tiles → {len(rows) - n0} detections {time.time() - t0:.0f}s")
    if not rows:
        return []
    feats = []
    D = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y", "image"])
    # Every aerial/orthophoto source image is an independent reference layer.
    # Fuse overlapping tiles within a source image, never between images.
    for image, Di in D.groupby("image", sort=False):
        F = fuse([Di[["region", "cls", "score", "x", "y"]]], np.ones(1), match_m=4.0)
        for r in F.itertuples():
            kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(r.y / 6378137.0)) - np.pi / 2)))
            ft = to_feature(circle(r.x, r.y, c["r_out"] * kk, 12), r.cls, r.score, region.name, markup)
            if ft:
                ft["properties"]["image"] = image
                ft["properties"]["source_layer"] = image
                feats.append(ft)
    return feats


def predict(region, paths, deadline, cfg_key="ae", markup="ae"):
    import time
    from pyproj import Transformer
    from koz3 import prev_sp_io, prev_sp_det
    from .sp_or import _det, PREV_CLASSES
    c = cfg()[cfg_key]
    feats = []
    det_name, res = {"ae": ("ae_det", 0.5), "or": ("or_det", 0.1)}[cfg_key]
    if c.get("class_thr") and _ae_weights(det_name):            # own-data FCOS (Ae fortifications / Or architecture)
        try:
            feats += predict_ae_det(region, paths, deadline, c, det_name, res, markup)
        except Exception as e:
            log(f"{markup} {region.name}: own detector failed {e!r}")
    keep = set(c.get("classes", ["kurgany_povrezhdennye"]))
    if not keep or not _det("sp_dfine"):
        return feats
    models = _det("sp_dfine")[: c.get("n_models", 2)]
    for p in paths:
        t0 = time.time()
        arr, T, crs, _ = prev_sp_io.read_scene(str(p), 0.5, max_px=1_600_000_000)   # large Ae sheets exceed the 120 Mpx default
        dets = prev_sp_det.detect_scene(models, arr, T, bs=4, time_left=lambda: deadline - time.time())
        tr = Transformer.from_crs(crs.to_string(), 3857, always_xy=True)
        n = 0
        for k, e, s in dets:
            cls = PREV_CLASSES[k]
            if cls not in keep or s < c["det_thr"]:
                continue
            x, y = tr.transform(e.centroid.x, e.centroid.y)
            kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(y / 6378137.0)) - np.pi / 2)))
            ft = to_feature(circle(x, y, c["r_out"] * kk, 12), cls, s, region.name, markup)
            if ft:
                ft["properties"]["image"] = p.name
                ft["properties"]["source_layer"] = p.name
                feats.append(ft); n += 1
        log(f"{markup} {p.name}: {arr.shape[1]}x{arr.shape[2]} px @0.5 m, {len(dets)} detections → {n} kept {time.time() - t0:.0f}s")
    return feats
