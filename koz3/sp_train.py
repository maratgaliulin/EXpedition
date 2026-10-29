"""Optical expert (SpOr / Ae / Or): chips from koz3.sp_prep → U-Net (timm ConvNeXt-T) with CenterNet heads.

Heads (9): heatmaps [kurgan_any, gorodishcha, arkhitektury, fortifikatsii] + masks for the 5 classes.
Same decoding idea as LiDAR: kurgan peaks take damaged/intact from the masks (dual emission when uncertain).

  python -m koz3.sp_train --mod SpOr --fold 0
  python -m koz3.sp_train --mod SpOr --oof
"""
import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3 import hm  # noqa: E402
from koz3.labels import DATA  # noqa: E402

CLS = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
HM_OF = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
N_HM, N_MK = 4, 5
CROP = 512
ENC = os.environ.get("SP_ENC", "tu-convnext_tiny")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None] * 255
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None] * 255
ROOT = os.path.join(DATA, "..", "..")


def prep_dir(mod):
    return os.path.join(DATA, "..", f"{mod.lower()}_prep")


def load_chips(mod):
    import geopandas as gpd
    from shapely.affinity import affine_transform
    from rasterio.transform import Affine
    ann = json.load(open(os.path.join(prep_dir(mod), "ann.json")))
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    L = L[(L["mod"] == mod) & L.class_name.isin(CLS)].explode(index_parts=False)
    by_region = {r: g for r, g in L.groupby("region")}
    # satellite: reuse the previous team's D-FINE fold assignment so the two model families fuse honestly OOF
    ff = os.path.join(ROOT, "kaggle_import", "out", "koz3-sp-det", "sp_runs", "sp_dfine_l", "folds.json")
    ext = {}
    if mod == "SpOr" and os.path.exists(ff):
        for k, v in json.load(open(ff, encoding="utf-8")).items():
            ext[k.rsplit("__", 1)[0]] = int(v)
    for c in ann:
        if c["region"] in ext:
            c["fold"] = ext[c["region"]]
    if os.environ.get("LORO") == "1":    # leave-one-region-out (data types whose labels sit in one site cluster: Or)
        regs = sorted({c["region"] for c in ann})
        for c in ann:
            c["fold"] = regs.index(c["region"])
    cache_crs = {}
    for c in ann:
        lab = by_region.get(c["region"])
        c["objs"] = []
        if lab is None:
            continue
        key = (c["region"], c["crs"])
        if key not in cache_crs:
            cache_crs[key] = lab.to_crs(c["crs"])
        lab = cache_crs[key]
        inv = ~Affine(*c["transform"])
        m = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]
        from shapely.geometry import box
        win = box(-c["w"] * 0.25, -c["h"] * 0.25, c["w"] * 1.25, c["h"] * 1.25)
        for g, cl in zip(lab.geometry, lab.class_name):
            gp = affine_transform(g, m)
            if gp.intersects(win):
                c["objs"].append((CLS.index(cl), gp))
    return ann


class Sampler:
    def __init__(self, chips, mod, rng):
        self.chips, self.rng, self.dir = chips, rng, os.path.join(prep_dir(mod), "chips")
        n_obj = np.array([len(c["objs"]) for c in chips])
        self.p = (1.0 + np.sqrt(n_obj)) * np.array([c["valid_frac"] for c in chips])
        if os.environ.get("SP_SITE_BAL") == "1":       # every region gets the same total sampling weight
            reg = np.array([c["region"] for c in chips])
            for r_ in np.unique(reg):
                m_ = reg == r_; self.p[m_] /= self.p[m_].sum()
        self.p /= self.p.sum()
        self.img_cache = {}

    def _img(self, f):
        im = self.img_cache.get(f)
        if im is None:
            im = cv2.imread(os.path.join(self.dir, f))[:, :, ::-1]
            if len(self.img_cache) < 3000:
                self.img_cache[f] = im
        return im

    def crop(self):
        c = self.chips[self.rng.choice(len(self.chips), p=self.p)]
        im = self._img(c["file"])
        s = float(np.exp(self.rng.uniform(np.log(0.7), np.log(1.4))))
        h, w = im.shape[:2]
        if c["objs"] and self.rng.random() < 0.6:
            g = c["objs"][self.rng.integers(len(c["objs"]))][1].representative_point()
            cx, cy = g.x + self.rng.normal(0, 80), g.y + self.rng.normal(0, 80)
        else:
            cx, cy = self.rng.uniform(0, w), self.rng.uniform(0, h)
        src = CROP / s
        x0, y0 = cx - src / 2, cy - src / 2
        x0 = float(np.clip(x0, min(0, w - src), max(0, w - src)))   # keep the crop inside the chip when it fits
        y0 = float(np.clip(y0, min(0, h - src), max(0, h - src)))
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        out = cv2.warpAffine(np.ascontiguousarray(im), M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0)
        val = (out.sum(2) > 0).astype(np.float32)
        from shapely.affinity import affine_transform
        objs = [(cl, affine_transform(g, [s, 0, 0, s, -x0 * s, -y0 * s])) for cl, g in c["objs"]]
        hmap, mk = hm.draw_targets((CROP, CROP), objs, len(CLS))
        t_hm = np.stack([np.maximum(hmap[0], hmap[1]), hmap[2], hmap[3], hmap[4]])
        img = out.astype(np.float32).transpose(2, 0, 1)
        # photometric: brightness/contrast/saturation/blur/grey (Ae-like) — style robustness across sensors
        img = img * self.rng.uniform(0.75, 1.25) + self.rng.uniform(-25, 25)
        if self.rng.random() < 0.3:
            g = img.mean(0, keepdims=True); img = g + (img - g) * self.rng.uniform(0.0, 1.3)
        if self.rng.random() < 0.2:
            k = int(self.rng.choice([3, 5])); img = np.stack([cv2.GaussianBlur(ch, (k, k), 0) for ch in img])
        if os.environ.get("SP_QAUG") == "1" and self.rng.random() < 0.5:      # image-quality degradation
            q = np.clip(img.transpose(1, 2, 0), 0, 255).astype(np.uint8)
            if self.rng.random() < 0.4:
                f_ = self.rng.uniform(1.5, 3.2); m_ = max(8, int(CROP / f_))
                q = cv2.resize(cv2.resize(q, (m_, m_), interpolation=cv2.INTER_AREA), (CROP, CROP), interpolation=cv2.INTER_LINEAR)
            if self.rng.random() < 0.35:
                ok, enc = cv2.imencode(".jpg", q[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(self.rng.integers(15, 60))]); q = cv2.imdecode(enc, 1)[:, :, ::-1]
            img = q.astype(np.float32).transpose(2, 0, 1)
            if self.rng.random() < 0.3:
                img = img + self.rng.normal(0, self.rng.uniform(3, 12), img.shape).astype(np.float32)
            if self.rng.random() < 0.2:
                a_ = self.rng.uniform(0.1, 0.4); img = img * (1 - a_) + a_ * self.rng.uniform(150, 230)
        img = (np.clip(img, 0, 255) - MEAN) / STD * val
        k = self.rng.integers(4); fl = self.rng.random() < 0.5
        arrs = [np.rot90(a, k, axes=(-2, -1)) for a in (img, t_hm, mk.astype(np.float32), val)]
        if fl:
            arrs = [a[..., ::-1] for a in arrs]
        return [np.ascontiguousarray(a) for a in arrs]

    def batch(self, n):
        b = [self.crop() for _ in range(n)]
        return [torch.from_numpy(np.stack([x[i] for x in b]).astype(np.float32)) for i in range(4)]


def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(ENC, encoder_weights="imagenet" if pretrained else None, in_channels=3,
                    classes=N_HM + N_MK, decoder_channels=(256, 128, 64, 32, 16))


def train(mod, fold, iters=12000, bs=8, lr=3e-4, seed=0, tag="", extra_mods=()):
    chips = [c for c in load_chips(mod) if c["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    S = Sampler(chips, mod, rng)
    model = build_model().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(6); futs = [ex.submit(S.batch, bs) for _ in range(3)]
    t0 = time.time()
    for i in range(iters):
        img, t_hm, mk, val = [x.cuda(non_blocking=True) for x in futs.pop(0).result()]
        futs.append(ex.submit(S.batch, bs))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img.to(memory_format=torch.channels_last))
        w = val[:, None]
        loss = hm.focal_hm(out[:, :N_HM], t_hm, w) + hm.bce_dice(out[:, N_HM:], mk, w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if i % 500 == 0 or i == iters - 1:
            print(f"{mod} fold {fold} it {i} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    d = os.path.join(ROOT, "work", mod.lower()); os.makedirs(d, exist_ok=True)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(d, f"hm{tag}_f{fold}.pt"))


def load_model(path):
    m = build_model(pretrained=False).cuda().eval()
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cuda").items()})
    return m


def norm_img(rgb):
    x = rgb.astype(np.float32).transpose(2, 0, 1)
    v = (x.sum(0) > 0).astype(np.float32)
    return (x - MEAN) / STD * v, v > 0


def decode_chip(prob, valid, transform, crs, region, markup, thr=0.1, dual=0.35, r_out=0.5, nms=9, scene=None):
    from pyproj import Transformer
    from rasterio.transform import Affine
    from shapely.geometry import Point
    t = Affine(*transform); tr = Transformer.from_crs(crs, 3857, always_xy=True)
    feats = []
    mk = prob[N_HM:]
    for ch, x, y, s in hm.decode(prob[:N_HM], valid, thr=thr, nms=nms):
        yi, xi = int(y), int(x)
        y0, y1, x0, x1 = max(0, yi - 2), yi + 3, max(0, xi - 2), xi + 3
        if ch == 0:
            pd, pi = float(mk[0, y0:y1, x0:x1].mean()), float(mk[1, y0:y1, x0:x1].mean())
            tot = pd + pi + 1e-6
            cand = [(0, pd / tot), (1, pi / tot)]
            best = max(cand, key=lambda c: c[1])
            emit = [best] + [c for c in cand if c is not best and c[1] >= dual]
        else:
            emit = [(ch + 1, 1.0)]
        X, Y = t * (x, y)
        mx, my = tr.transform(X, Y)
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2)))
        for c, pc in emit:
            props = {"class_name": CLS[c], "confidence": float(s * pc), "region_name": region, "markup_type": markup}
            if scene is not None:      # every source image is an independent reference layer (deployment rule)
                props["source_layer"] = scene
            feats.append({"type": "Feature", "geometry": Point(mx, my).buffer(r_out * k, 12).__geo_interface__,
                          "properties": props})
    return feats


def dedupe(feats, dist_m=3.0):
    """Overlapping tiles produce the same peak twice: keep the most confident per class within dist_m.

    Duplicates are removed only within one source image; coincident objects from different scenes stay separate,
    matching what the deployed expert does (and what the organisers require)."""
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    out = []
    keys = {(f["properties"]["class_name"], f["properties"].get("source_layer")) for f in feats}
    for cls, layer in keys:
        fs = sorted([f for f in feats if f["properties"]["class_name"] == cls
                     and f["properties"].get("source_layer") == layer],
                    key=lambda f: -f["properties"]["confidence"])
        pts = [shape(f["geometry"]).centroid for f in fs]
        if not pts:
            continue
        tree = STRtree(pts); taken = np.zeros(len(fs), bool)
        for i, p in enumerate(pts):
            if taken[i]:
                continue
            out.append(fs[i])
            for j in tree.query(p.buffer(dist_m)):
                taken[j] = True
    return out


def oof(mod, tag=""):
    from koz3.labels import to_gt_geojson
    import geopandas as gpd
    chips = load_chips(mod)
    d = os.path.join(ROOT, "work", mod.lower())
    probs = {}
    for fold in sorted({c["fold"] for c in chips}):
        m = load_model(os.path.join(d, f"hm{tag}_f{fold}.pt"))
        for c in [c for c in chips if c["fold"] == fold]:
            rgb = cv2.imread(os.path.join(prep_dir(mod), "chips", c["file"]))[:, :, ::-1]
            x, v = norm_img(rgb)
            probs[c["file"]] = (hm.predict_tiled(m, x, N_HM + N_MK, tile=1024, overlap=0).astype(np.float16), v)
        print("oof fold", fold, flush=True)
    sweep(mod, chips, probs, tag=tag)   # probability maps stay in memory (Ae: ~20 GB as a pickle); candidates are saved


def sweep(mod, chips, probs, thrs=(0.05, 0.1, 0.15, 0.2, 0.3), duals=(0.3, 0.4, 1.1), tag=""):
    """Nested selection of (threshold, dual) over the chip folds; saves low-threshold candidates for fusion studies."""
    import pandas as pd
    from shapely.geometry import shape
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    use = [c for c in chips if c["file"] in probs]
    fold_of = {c["region"]: c["fold"] for c in use}
    folds = sorted(set(fold_of.values()))
    gt_by_fold = {f: to_gt_geojson(L[(L["mod"] == mod) & L.class_name.isin(CLS) & L.region.isin([r for r, g in fold_of.items() if g == f])]) for f in folds}
    cache = {}

    def feats_for(t, dual):
        if (t, dual) not in cache:
            fs = []
            for c in use:
                p, v = probs[c["file"]]
                fs += decode_chip(p.astype(np.float32), v, c["transform"], c["crs"], c["region"], mod.lower(), t, dual,
                                  scene=c["scene"])
            cache[(t, dual)] = dedupe(fs)
        return cache[(t, dual)]
    base = feats_for(min(thrs), 0.3)
    # source_layer keeps the candidate's source image: the OOF harness fuses within an image, like the expert does
    pd.DataFrame([{"region": f["properties"]["region_name"], "cls": f["properties"]["class_name"],
                   "score": f["properties"]["confidence"],
                   "x": shape(f["geometry"]).centroid.x, "y": shape(f["geometry"]).centroid.y,
                   "source_layer": f["properties"].get("source_layer")} for f in base]).to_parquet(
        os.path.join(ROOT, "work", mod.lower(), f"oof_hm{tag}_cands.parquet"))
    make = lambda s, f: {"type": "FeatureCollection", "features": [x for x in feats_for(*s) if fold_of[x["properties"]["region_name"]] == f]}
    C = fold_counts(make, [(t, d) for t in thrs for d in duals], folds, gt_by_fold, CLS)
    h, ch, o, os_, tot = nested(C)
    print(f"{mod} heatmap honest {h:.4f} (oracle {o:.4f} @ {os_}) chosen {ch}")
    for c, (tp, fp, fn) in tot.items():
        if tp + fn:
            print(f"   {c:24s} F1 {2 * tp / max(2 * tp + fn + 0.5 * fp, 1):.3f} tp {tp} fp {fp} fn {fn}")
    return h


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", default="SpOr")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.fold is not None:
        train(a.mod, a.fold, a.iters, bs=a.bs, tag=a.tag, seed=a.seed)
    if a.oof:
        oof(a.mod, a.tag)
