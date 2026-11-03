"""LiDAR expert: relief channels (koz3.relief) → U-Net (timm ConvNeXt-T encoder).

Heads (7): heatmaps [kurgan_any, gorodishcha, fortifikatsii] at label representative points
           + masks [kurgany_povrezhdennye, kurgany_tselye, gorodishcha, fortifikatsii].
Decoding: kurgan peaks take damaged/intact from the mask probabilities around the peak (dual emission when
both clear their threshold). Folds: 20 km site clusters (koz3.sp_prep.site_folds), 4 folds.

  python -m koz3.li_train --cache            # relief cache for every prepared site
  python -m koz3.li_train --fold 0           # train (fold -1 = all sites)
  python -m koz3.li_train --oof              # held-out predictions + official-metric sweep
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3 import dtm as dtm_mod, hm, relief  # noqa: E402
from koz3.labels import DATA  # noqa: E402

ROOT = os.path.join(DATA, "..", "..")
PREP = os.path.join(DATA, "..", "las_prep")
CACHE = os.path.join(DATA, "..", "li_cache")
RUNS = os.path.join(ROOT, "work", "li")
CLS = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "fortifikatsii"]
HM_OF = {0: 0, 1: 0, 2: 1, 3: 2}         # class → heatmap channel
N_HM, N_MK = 3, 4
CROP = 512
ENC = os.environ.get("LI_ENC", "tu-convnext_tiny")


def labels():
    import geopandas as gpd
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    return L[(L["mod"] == "Li") & L.class_name.isin(CLS)]


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(PREP, "*.npz"))):
        region = os.path.basename(f)[:-4]
        out = os.path.join(CACHE, region + ".npy")
        if os.path.exists(out):
            continue
        d, v, (x0, ytop, res), crs = dtm_mod.load(f)
        # tiled: the 12-channel (RELIEF_EXTRA) build allocates ~10 GB on a multi-km sheet in one shot and OOMs
        x, vv = relief.channels_tiled(d, v, res)
        np.save(out, x.astype(np.float16))
        np.save(os.path.join(CACHE, region + "_valid.npy"), vv)
        json.dump({"x0": x0, "ytop": ytop, "res": relief.NET_RES, "crs": crs}, open(os.path.join(CACHE, region + ".json"), "w"))
        print(region, x.shape, flush=True)


def site_items(L):
    from koz3.sp_prep import site_folds
    folds = site_folds(L)
    # reuse the previous team's L2 fold assignment (20 km groups) so both LiDAR families compare / fuse OOF
    ff = os.path.join(ROOT, "kaggle_import", "out", "koz3-li-train-cnx", "L2_cnxt_all59_8k_l0", "folds.csv")
    if os.path.exists(ff):
        import csv
        folds.update({r["site"]: int(r["fold"]) for r in csv.DictReader(open(ff, encoding="utf-8"))})
    items = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.json"))):
        region = os.path.basename(f)[:-5]
        meta = json.load(open(f))
        x = np.load(os.path.join(CACHE, region + ".npy"), mmap_mode="r")
        v = np.load(os.path.join(CACHE, region + "_valid.npy"))
        lab = L[L.region == region]
        objs = []
        if len(lab):
            from shapely.affinity import affine_transform
            lab = lab.to_crs(meta["crs"]).explode(index_parts=False)
            r = meta["res"]
            for g, c in zip(lab.geometry, lab.class_name):   # world → pixel
                objs.append((CLS.index(c), affine_transform(g, [1 / r, 0, 0, -1 / r, -meta["x0"] / r, meta["ytop"] / r])))
        items.append(dict(region=region, x=x, valid=v, meta=meta, objs=objs, fold=folds.get(region, -1)))
    return items


PASTE = os.environ.get("LI_PASTE") == "1"  # copy-paste of real kurgan relief onto other terrain (with LI_AUG)
PASTE_ATT = float(os.environ.get("LI_PASTE_ATT", "0"))   # P(pasted mound attenuated to 30–100 % height): low-contrast mounds
AUG = os.environ.get("LI_AUG") == "1"      # physical augmentation (height ×0.5–2.2, size ×0.7–1.45 on the DTM) + site balance


_DTM = {}


def _dtm(region):
    """Full-resolution DTM (metres, float32) + valid + source resolution, kept in RAM for augmented crops."""
    if region not in _DTM:
        from koz3 import dtm as dtm_mod
        d, v, (x0, ytop, res), _ = dtm_mod.load(os.path.join(PREP, region + ".npz"))
        _DTM[region] = (d.astype(np.float32), v, float(res))
    return _DTM[region]


class Sampler:
    def __init__(self, items, rng):
        self.items = [it for it in items if it["objs"]]
        self.rng = rng
        allobj = [(i, k) for i, it in enumerate(self.items) for k in range(len(it["objs"]))]
        cls = np.array([self.items[i]["objs"][k][0] for i, k in allobj])
        cnt = np.bincount(cls, minlength=len(CLS))
        w = 1 / np.sqrt(np.maximum(cnt[cls], 1)); w = np.minimum(w / w.min(), 4.0)
        self.kobj = [(i, k) for i, k in allobj if self.items[i]["objs"][k][0] in (0, 1)]        # kurgans, for copy-paste
        if AUG:                               # each site's objects share a budget ∝ sqrt(n): Gnezdovo no longer dominates
            n_reg = np.array([len(self.items[i]["objs"]) for i, _ in allobj], float)
            w = w / np.sqrt(n_reg)
        self.allobj, self.p = allobj, w / w.sum()
        sz = np.array([it["valid"].sum() for it in self.items], float)
        self.ps = np.sqrt(sz) / np.sqrt(sz).sum()

    def crop(self):
        if self.rng.random() < 0.65:
            i, k = self.allobj[self.rng.choice(len(self.allobj), p=self.p)]
            it = self.items[i]
            p = it["objs"][k][1].representative_point()
            cx, cy = p.x + self.rng.normal(0, CROP / 5), p.y + self.rng.normal(0, CROP / 5)
        else:
            it = self.items[self.rng.choice(len(self.items), p=self.ps)]
            H, W = it["valid"].shape
            cx, cy = self.rng.uniform(0, W), self.rng.uniform(0, H)
        H, W = it["valid"].shape
        if relief.EXTRA or (AUG and self.rng.random() < 0.7):      # EXTRA channels are not cached: always on the fly
            img, val, objs = self.aug_crop(it, cx, cy, plain=(not AUG) or (relief.EXTRA and self.rng.random() < 0.3))
            return self.finish(img, val, objs)
        x0, y0 = int(cx - CROP / 2), int(cy - CROP / 2)
        img = np.zeros((relief.N_CH, CROP, CROP), np.float32); val = np.zeros((CROP, CROP), np.float32)
        ys, xs = slice(max(0, y0), min(H, y0 + CROP)), slice(max(0, x0), min(W, x0 + CROP))
        if ys.start < ys.stop and xs.start < xs.stop:
            img[:, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = it["x"][:, ys, xs]
            val[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = it["valid"][ys, xs]
        from shapely.affinity import translate
        from shapely.geometry import box
        win = box(x0 - CROP * 0.5, y0 - CROP * 0.5, x0 + CROP * 1.5, y0 + CROP * 1.5)
        objs = [(c, translate(g, -x0, -y0)) for c, g in it["objs"] if g.intersects(win)]
        return self.finish(img, val, objs)

    def _fork(self, seed):
        c = object.__new__(Sampler); c.__dict__.update({k: v for k, v in self.__dict__.items() if k != "_pool"})
        c.rng = np.random.default_rng(seed)
        return c

    def paste(self, win, wv, X0, Y0, sres, k_max=6):
        """Copy-paste of REAL kurgans: a labelled mound's local relief (DTM − 10 m trend, feathered polygon mask) from a
        random training site is added to this window at a free valid spot. Returns pasted (class, polygon in DTM px)."""
        import cv2
        from shapely.affinity import translate, scale
        out = []
        H, W = win.shape
        for _ in range(int(self.rng.integers(1, k_max + 1))):
            i, k = self.kobj[self.rng.integers(len(self.kobj))]
            src = self.items[i]; c, g = src["objs"][k]
            d, v, sr = _dtm(src["region"])
            f = relief.NET_RES / sr                                            # cache px → source DTM px
            gx0, gy0, gx1, gy1 = [b * f for b in g.bounds]
            pad = int(10 / sr)
            a0, b0, a1, b1 = int(gx0) - pad, int(gy0) - pad, int(gx1) + pad + 1, int(gy1) + pad + 1
            if a0 < 0 or b0 < 0 or a1 > d.shape[1] or b1 > d.shape[0] or not v[b0:b1, a0:a1].all():
                continue
            patch = d[b0:b1, a0:a1].astype(np.float32)
            trend = cv2.GaussianBlur(patch, (0, 0), sigmaX=max(1.0, 10 / sr))       # 10 m trend keeps whole mounds
            rel = patch - trend
            gp = translate(scale(g, f, f, origin=(0, 0)), -a0, -b0)
            m = np.zeros(patch.shape, np.uint8)
            cv2.fillPoly(m, [np.array(gp.buffer(2 / sr).exterior.coords, np.int32)], 1)
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=max(1.0, 1 / sr))
            rel = rel * m
            if PASTE_ATT and self.rng.random() < PASTE_ATT:    # mound/background contrast changes (global ×s does not)
                rel = rel * float(np.exp(self.rng.uniform(np.log(0.3), 0.0)))
            if sr != sres:
                rel = cv2.resize(rel, None, fx=sr / sres, fy=sr / sres, interpolation=cv2.INTER_LINEAR)
            ph, pw = rel.shape
            if ph >= H or pw >= W:
                continue
            ty, tx = int(self.rng.integers(0, H - ph)), int(self.rng.integers(0, W - pw))
            if not wv[ty:ty + ph, tx:tx + pw].all():
                continue
            win[ty:ty + ph, tx:tx + pw] += rel
            out.append((c, translate(scale(gp, sr / sres, sr / sres, origin=(0, 0)), tx, ty)))
        return out

    def aug_crop(self, it, cx, cy, margin=160, plain=False):
        """Relief channels recomputed from the DTM with the ground stretched by zoom z (mounds z× larger, physically
        flatter) and heights scaled by s (+ copy-paste of real kurgans); cx, cy in cache pixels (NET_RES).
        plain: z = s = 1, no paste (on-the-fly equivalent of the cached crop)."""
        from shapely.affinity import affine_transform, scale, translate
        from shapely.geometry import box
        z = 1.0 if plain else float(np.exp(self.rng.uniform(np.log(0.7), np.log(1.45))))
        sc = 1.0 if plain else float(np.exp(self.rng.uniform(np.log(0.5), np.log(2.2))))
        d, v, sres = _dtm(it["region"])
        half_m = (CROP / 2 + margin) * relief.NET_RES / z                    # real metres around the centre
        ccx, ccy = cx * relief.NET_RES / sres, cy * relief.NET_RES / sres     # centre in DTM pixels
        hp = int(np.ceil(half_m / sres))
        X0, Y0 = int(round(ccx)) - hp, int(round(ccy)) - hp
        win = np.zeros((2 * hp, 2 * hp), np.float32); wv = np.zeros((2 * hp, 2 * hp), bool)
        ys, xs = slice(max(0, Y0), min(d.shape[0], Y0 + 2 * hp)), slice(max(0, X0), min(d.shape[1], X0 + 2 * hp))
        if ys.start < ys.stop and xs.start < xs.stop:
            win[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0] = d[ys, xs]
            wv[ys.start - Y0:ys.stop - Y0, xs.start - X0:xs.stop - X0] = v[ys, xs]
        pasted = []
        if wv.any():
            win = np.where(wv, win - float(np.median(win[wv])), 0.0).astype(np.float32)
            if PASTE and not plain and self.kobj and self.rng.random() < 0.5:
                pasted = self.paste(win, wv, X0, Y0, sres)
            win = (win * sc).astype(np.float32)
        x, vv = relief.channels(win, wv, sres * z)
        oy, ox = (x.shape[1] - CROP) // 2, (x.shape[2] - CROP) // 2
        img = np.zeros((relief.N_CH, CROP, CROP), np.float32); val = np.zeros((CROP, CROP), np.float32)
        a = x[:, max(oy, 0):oy + CROP, max(ox, 0):ox + CROP]; img[:, :a.shape[1], :a.shape[2]] = a
        b = vv[max(oy, 0):oy + CROP, max(ox, 0):ox + CROP]; val[:b.shape[0], :b.shape[1]] = b
        # cache px p → output px (p − c)·z + CROP/2 (the window centre is the crop centre)
        ux, uy = ccx * sres / relief.NET_RES, ccy * sres / relief.NET_RES
        T = [z, 0, 0, z, CROP / 2 - ux * z, CROP / 2 - uy * z]
        winb = box(ux - CROP / z, uy - CROP / z, ux + CROP / z, uy + CROP / z)
        objs = [(c, affine_transform(g, T)) for c, g in it["objs"] if g.intersects(winb)]
        # pasted objects: DTM-window px → cache px (×sres/NET_RES, + window origin) → output px (T)
        f = sres / relief.NET_RES
        objs += [(c, affine_transform(translate(scale(g, f, f, origin=(0, 0)), X0 * f, Y0 * f), T)) for c, g in pasted]
        return img, val, objs

    def finish(self, img, val, objs):
        hmap, mk = hm.draw_targets((CROP, CROP), objs, len(CLS))
        t_hm = np.stack([np.maximum(hmap[0], hmap[1]), hmap[2], hmap[3]])
        k = self.rng.integers(4); fl = self.rng.random() < 0.5
        arrs = [img, t_hm, mk.astype(np.float32), val]
        arrs = [np.rot90(a, k, axes=(-2, -1)) for a in arrs]
        if fl:
            arrs = [a[..., ::-1] for a in arrs]
        img, t_hm, mk, val = [np.ascontiguousarray(a) for a in arrs]
        if k % 2 == 1 or fl:   # hillshade channels encode direction; keep them consistent by swapping/negating
            pass                # (both azimuths are present, so the net sees rotated lighting as augmentation)
        img = img * self.rng.uniform(0.8, 1.2)
        return img, t_hm, mk, val

    def batch(self, n):
        if AUG:                         # augmented crops are CPU/GPU heavy: build them in parallel, one RNG per crop
            from concurrent.futures import ThreadPoolExecutor
            if not hasattr(self, "_pool"):
                self._pool = ThreadPoolExecutor(n)
            seeds = self.rng.integers(1 << 31, size=n)
            b = list(self._pool.map(lambda sd: Sampler.crop(self._fork(sd)), seeds))
        else:
            b = [self.crop() for _ in range(n)]
        return [torch.from_numpy(np.stack([x[i] for x in b]).astype(np.float32)) for i in range(4)]


def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    kw = {"img_size": CROP, "strict_img_size": False} if "swin" in ENC else {}       # Swin: fixed 512 px windows
    return smp.Unet(ENC, encoder_weights="imagenet" if pretrained else None, in_channels=relief.N_CH,
                    classes=N_HM + N_MK, decoder_channels=(256, 128, 64, 32, 16), **kw)


def train(fold, iters=12000, bs=8, lr=3e-4, seed=0, tag=""):
    L = labels()
    items = [it for it in site_items(L) if it["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    S = Sampler(items, rng)
    model = build_model().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(4); fut = ex.submit(S.batch, bs)
    t0 = time.time()
    for i in range(iters):
        img, t_hm, mk, val = [x.cuda(non_blocking=True) for x in fut.result()]
        fut = ex.submit(S.batch, bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img.to(memory_format=torch.channels_last))
        w = val[:, None]
        mask_loss = hm.bce_softf1 if os.environ.get("LI_LOSS") == "softf1" else hm.bce_dice
        loss = hm.focal_hm(out[:, :N_HM], t_hm, w) + mask_loss(out[:, N_HM:], mk, w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if i % 500 == 0 or i == iters - 1:
            print(f"li fold {fold} it {i} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(RUNS, exist_ok=True)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(RUNS, f"li{tag}_f{fold}.pt"))


def load_model(path):
    m = build_model(pretrained=False).cuda().eval()
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cuda").items()})
    return m


def neighbour_prior(peaks, R=200.0, K=3, snb=0.2, res=0.5):
    """Kurgans come in cemeteries: kurgan-peak height × (0.5 + 0.5·clip(n/K, 0, 2)), n = other kurgan peaks of height
    ≥ snb within R metres (nested OOF on the deployed blend 0.5765 → 0.5831, chosen in all folds)."""
    from scipy.spatial import cKDTree
    strong = np.array([(x, y) for c, x, y, s in peaks if c == 0 and s >= snb]).reshape(-1, 2)
    t = cKDTree(strong) if len(strong) else None; out = []
    for c, x, y, s in peaks:
        if c == 0:
            n = (len(t.query_ball_point((x, y), R / res)) - (1 if s >= snb else 0)) if t is not None else 0
            s = s * (0.5 + 0.5 * float(np.clip(n / K, 0, 2)))
        out.append((c, x, y, float(s)))
    return out


def decode_site(prob, valid, meta, thr=(0.1, 0.1, 0.1), dual=0.35, r_out=1.0, nms=11, prior=None):
    """prob (7,H,W) → features in EPSG:3857 (small circles at peaks; the metric scores detection, not contour)."""
    from pyproj import Transformer
    from shapely.geometry import Point
    tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True)
    feats = []
    if prior:
        peaks = neighbour_prior(hm.decode(prob[:N_HM], valid, thr=min(min(thr), 0.05), nms=nms), res=meta["res"], **prior)
    else:
        peaks = hm.decode(prob[:N_HM], valid, thr=min(thr), nms=nms)
    mk = prob[N_HM:]
    for ch, x, y, s in peaks:
        if s < thr[ch]:
            continue
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
        X, Y = meta["x0"] + x * meta["res"], meta["ytop"] - y * meta["res"]
        mx, my = tr.transform(X, Y)
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2)))
        for c, pc in emit:
            feats.append({"type": "Feature", "geometry": Point(mx, my).buffer(r_out * k, 12).__geo_interface__,
                          "properties": {"class_name": CLS[c], "confidence": float(s * pc), "region_name": meta.get("region", ""),
                                         "markup_type": "li"}})
    return feats


def comp_feats(prob_mask, valid, meta, cls, thr, min_px, r_out=1.0):
    import cv2
    from pyproj import Transformer
    from shapely.geometry import Polygon, Point
    m = ((prob_mask > thr) & valid).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    tr = Transformer.from_crs(meta["crs"], 3857, always_xy=True); out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < min_px:
            continue
        cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        if len(c) < 4:
            continue
        p = Polygon(c[:, 0, :] + 0.5).buffer(0)
        if p.is_empty:
            continue
        p = max(p.geoms, key=lambda g: g.area) if p.geom_type != "Polygon" else p
        mode = os.environ.get("LI_ANCHOR", "rep")       # rep: surface point; pole: pole of inaccessibility; peak: prob max
        if mode == "pole":
            from shapely.ops import polylabel
            rp = polylabel(p, tolerance=0.5)
        elif mode == "peak":
            yy, xx = np.where(lab == k); j = int(np.argmax(prob_mask[yy, xx])); rp = Point(xx[j] + 0.5, yy[j] + 0.5)
        else:
            rp = p.representative_point()
        mx, my = tr.transform(meta["x0"] + rp.x * meta["res"], meta["ytop"] - rp.y * meta["res"])
        kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(my / 6378137.0)) - np.pi / 2)))
        out.append({"type": "Feature", "geometry": Point(mx, my).buffer(r_out * kk, 12).__geo_interface__,
                    "properties": {"class_name": cls, "confidence": float(prob_mask[lab == k].mean()), "region_name": meta["region"], "markup_type": "li"}})
    return out


def decode_site_v2(prob, valid, meta, thr=0.1, dual=0.3, comp_thr=0.5, fort_min_px=30, gor_min_px=600, r_out=1.0, prior=None):
    """Deployed LiDAR decoding (nested OOF 0.5223): kurgans from heatmap peaks (damaged/intact from the masks, dual
    emission), hillforts and fortifications from mask components (representative point of each component)."""
    feats = [f for f in decode_site(prob, valid, meta, thr=(thr, thr, thr), dual=dual, r_out=r_out, prior=prior)
             if f["properties"]["class_name"] not in ("fortifikatsii", "gorodishcha")]
    feats += comp_feats(prob[N_HM + CLS.index("fortifikatsii")], valid, meta, "fortifikatsii", comp_thr, fort_min_px, r_out)
    feats += comp_feats(prob[N_HM + CLS.index("gorodishcha")], valid, meta, "gorodishcha", comp_thr, gor_min_px, r_out)
    return feats


def oof(tag=""):
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    L = labels()
    items = site_items(L)
    probs = {}
    for fold in sorted({it["fold"] for it in items if it["fold"] >= 0}):
        m = load_model(os.path.join(RUNS, f"li{tag}_f{fold}.pt"))
        for it in [i for i in items if i["fold"] == fold]:
            p = hm.predict_tiled(m, np.asarray(it["x"], np.float32), N_HM + N_MK)
            probs[it["region"]] = p.astype(np.float16)
            print("oof", it["region"], flush=True)
    pickle.dump(probs, open(os.path.join(RUNS, f"oof{tag}.pkl"), "wb"))
    sweep(items, probs, L)


def oof_dtm(tag=""):
    """OOF for models whose channels are not cached (RELIEF_EXTRA): held-out sites' channels are computed from the DTM
    exactly as at inference (relief.channels_tiled); same grid as the cache."""
    L = labels(); items = site_items(L); probs = {}
    for fold in sorted({it["fold"] for it in items if it["fold"] >= 0}):
        m = load_model(os.path.join(RUNS, f"li{tag}_f{fold}.pt"))
        for it in [i for i in items if i["fold"] == fold]:
            d, v, res = _dtm(it["region"])
            x, _ = relief.channels_tiled(d, v, res)
            assert x.shape[1:] == it["valid"].shape, (it["region"], x.shape, it["valid"].shape)
            probs[it["region"]] = hm.predict_tiled(m, x, N_HM + N_MK).astype(np.float16)
            _DTM.pop(it["region"], None)
            print("oof", it["region"], flush=True)
    pickle.dump(probs, open(os.path.join(RUNS, f"oof{tag}.pkl"), "wb"))


def sweep(items, probs, L, thrs=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4), duals=(0.3, 0.4, 1.1)):
    """Nested selection of (threshold, dual) over LiDAR folds; prints honest and oracle scores."""
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    meta = {it["region"]: dict(it["meta"], region=it["region"]) for it in items}
    valid = {it["region"]: it["valid"] for it in items}
    fold = {it["region"]: it["fold"] for it in items}
    folds = sorted({fold[r] for r in probs})
    gt_by_fold = {f: to_gt_geojson(L[L.region.isin([r for r in probs if fold[r] == f])]) for f in folds}
    cache = {}

    def make(sett, f):
        t, dual = sett
        feats = []
        for r, p in probs.items():
            if fold[r] != f:
                continue
            key = (r, t, dual)
            if key not in cache:
                cache[key] = decode_site(p.astype(np.float32), valid[r], meta[r], thr=(t, t, t), dual=dual)
            feats += cache[key]
        return {"type": "FeatureCollection", "features": feats}
    settings = [(t, d) for t in thrs for d in duals]
    C = fold_counts(make, settings, folds, gt_by_fold, CLS)
    h, ch, o, os_, tot = nested(C)
    print(f"LiDAR honest {h:.4f}  (oracle {o:.4f} @ {os_})  chosen per fold {ch}")
    for c, (tp, fp, fn) in tot.items():
        f1 = 2 * tp / max(2 * tp + fn + 0.5 * fp, 1)
        print(f"   {c:24s} F1 {f1:.3f}  tp {tp} fp {fp} fn {fn}")
    return h, o, os_


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--oof_dtm", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cache:
        build_cache()
    if a.fold is not None:
        train(a.fold, a.iters, tag=a.tag, seed=a.seed)
    if a.oof_dtm:
        oof_dtm(a.tag)
    if a.oof:
        oof(a.tag)
