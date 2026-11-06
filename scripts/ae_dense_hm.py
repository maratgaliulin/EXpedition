"""Dense-object Ae kurgan heatmap: fixes the two things that made `sp_train --mod Ae` fail.

Diagnosis (docs/EXPERIMENTS.md, 3 Nov): at 0.5 m a kurgan target is a Gaussian of sigma ~2 px, so a 512 crop of a field
sheet holds 1-2 positive pixels; `hm.focal_hm` divides by the positive count, the loss swings 0.68 <-> 5.88 between
checks and the checkpoint ends up either firing everywhere or nowhere. Two independent changes here:

  1. objective: focal normalised by max(positive count, FLOOR) per batch, so an object-poor batch cannot dominate;
  2. an explicit target-sanity gate: before training, the loss at p = t must beat the loss at p = t/2 and at a flat
     0.02 map. Widening sigma to make the "signal" bigger does the opposite — the halo is what the (1-t)^4 background
     term suppresses, so a wide sigma makes underestimation cheaper and the run goes flat. 5 Nov checkpoints:
     sig 6 px -> perfect 0.370 / half 0.213, max heat 0.057 after 12k iterations. Now sigma is tight (2-4 px) and a
     run that fails the gate refuses to start.

plus a dense-first sampler (70 % of crops centred on an object, 10 % of those forced inside a cemetery cluster of >= 6
labels within 60 m — the ledger's "trained on cemetery sheets"), continuous rotation augmentation (Ae_ROT, the D4
group alone is 8 variants for imagery with no canonical orientation) and film photometry / JPEG / resolution-loss.

  python scripts/ae_dense_hm.py --fold 0 --tag _v22       # train one fold
  python scripts/ae_dense_hm.py --oof --tag _v22 --report # candidates + per-sheet / density precision
  python scripts/ae_hm_prior_oof.py --sweep --tag _v22    # framing-prior sweep on those candidates
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3 import hm, sp_train  # noqa: E402
from koz3.labels import DATA  # noqa: E402

WORK = os.path.join(ROOT, "work", "ae")
CROP = 512
# Peak width in PIXELS. A wide sigma does NOT create more positive pixels (pos = t >= 0.999 stays 1 px per object);
# it only adds halo pixels that the (1-t)^4 background term is allowed to suppress, which makes the flat/under-scaled
# prediction cheaper than the correct one. Measured on a fold-0 batch: sig 6 px (r=18) → ~50 000 halo px per batch,
# perfect 0.370 / half 0.213 / flat 3.771, i.e. the objective PREFERS half the target and the 12k run went flat
# (peak 0.057). At 2-4 px the halo cost per object is < 5 % of the peak term and the ordering is correct again.
# Keep it tight: this is the same convention as koz3.hm.draw_targets (sig_frac 0.12 clipped to [1.5, 40] at 0.25 m).
SIG_FRAC = float(os.environ.get("AE_SIG_FRAC", 0.06))
SIG_MIN = float(os.environ.get("AE_SIG_MIN", 2.0))
SIG_MAX = float(os.environ.get("AE_SIG_MAX", 4.0))
FLOOR = 8.0                                      # focal denominator floor (object centres)
ROT = os.environ.get("AE_ROT", "1") == "1"       # continuous rotation aug (aerial film has no canonical orientation)
SCALE_LO = float(os.environ.get("AE_SCALE_LO", 0.7))
SCALE_HI = float(os.environ.get("AE_SCALE_HI", 1.4))
LOSS = os.environ.get("AE_LOSS", "count")        # "count" = weighted MSE (optimum p=t) | "focal" = CenterNet
POS_W = float(os.environ.get("AE_POS_W", 50.0))  # object-neighbourhood weight inside the counting loss
KP, KT = 0, 1                                    # sp_train.CLS indices
MEAN = sp_train.MEAN
STD = sp_train.STD


def targets(shape, objs):
    """objs [(cls, polygon in crop px)] → heat (1,H,W), masks (2,H,W)."""
    from rasterio.features import rasterize
    H, W = shape
    heat = np.zeros((H, W), np.float32)
    kp, kt = [], []
    for c, g in objs:
        if c not in (KP, KT):
            continue
        (kp if c == KP else kt).append(g)
        p = g.representative_point()
        s = float(np.clip(SIG_FRAC * np.sqrt(max(g.area, 1.0)), SIG_MIN, SIG_MAX))
        r = int(3 * s)
        cx, cy = int(round(p.x)), int(round(p.y))
        x0, x1, y0, y1 = max(0, cx - r), min(W, cx + r + 1), max(0, cy - r), min(H, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1],
                                        np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)).astype(np.float32))
    mk = np.zeros((2, H, W), np.float32)
    for i, gs in enumerate((kp, kt)):
        if gs:
            mk[i] = rasterize([(g, 1) for g in gs if not g.is_empty], out_shape=(H, W), dtype="uint8", all_touched=True)
    return heat[None], mk


def focal_stable(logit, t, w):
    """CenterNet focal with a positive-count floor: the denominator cannot collapse to 1-2 pixels."""
    p = torch.sigmoid(logit.float()).clamp(1e-4, 1 - 1e-4)
    pos = t.ge(0.999).float()
    pl = -torch.log(p) * (1 - p) ** 2 * pos
    nl = -torch.log(1 - p) * p ** 2 * (1 - t) ** 4 * (1 - pos)
    return ((pl + nl) * w).sum() / (pos * w).sum().clamp(min=FLOOR)


def count_loss(logit, t, w, pos_w=POS_W):
    """Weighted MSE on the Gaussian density map — the crowd-counting objective (MCNN / CSRNet family).

    Why not focal: the focal minimum is NOT at p = t. Measured on a lone Gaussian, the preferred amplitude is
    alpha* = 0.75 (sig 1.5 px) / 0.70 (2 px) / 0.55 (4 px) / 0.45 (6 px) — widening the target does not add usable
    signal, it moves the optimum toward a flat map, which is exactly the 5 Nov run (peak 0.057 at 12k iterations).
    MSE is separable per pixel, so its optimum is p = t at EVERY pixel regardless of the weights: the weights only
    set the optimisation dynamics. pos_w up-weights the object neighbourhood so a "predict 0 everywhere" map stays
    expensive (that map is the failure mode of an unweighted MSE on a 1e-4-positive background).
    The dense-cemetery task (498 labels in one chip) is a counting task, and this is the standard counting loss.
    """
    p = torch.sigmoid(logit.float())
    wt = w * (1.0 + pos_w * (t > 0.05).float())
    return ((p - t) ** 2 * wt).sum() / wt.sum().clamp(min=1)


class Sampler:
    def __init__(self, chips, rng):
        from scipy.spatial import cKDTree
        self.chips, self.rng, self.dir = chips, rng, os.path.join(sp_train.prep_dir("Ae"), "chips")
        pts, dense = [], []
        for i, c in enumerate(chips):
            if not c["objs"]:
                continue
            p = np.array([[g.representative_point().x, g.representative_point().y] for _, g in c["objs"]])
            pts.append((i, p))
            if len(p) >= 6:                       # cemetery chip: ≥6 labels in one 1024 chip
                dense.append((i, cKDTree(p)))
        self.dense = dense
        self.obj = [(i, k) for i, p in pts for k in range(len(p))]
        self.with_obj = sorted({i for i, _ in pts})
        self.empty = [i for i, c in enumerate(chips) if not c["objs"]]
        self.cache = {}

    def _img(self, f):
        im = self.cache.get(f)
        if im is None:
            im = cv2.imread(os.path.join(self.dir, f))[:, :, ::-1]
            if len(self.cache) < 1500:
                self.cache[f] = im
        return im

    def crop(self):
        r = self.rng.random()
        if r < 0.10 and self.dense:                       # inside a cemetery cluster
            i, tree = self.dense[self.rng.integers(len(self.dense))]
            p = self.chips[i]["objs"][self.rng.integers(len(self.chips[i]["objs"]))][1].representative_point()
            nb = tree.query_ball_point([p.x, p.y], 60.0)
            q = np.array([[self.chips[i]["objs"][k][1].representative_point().x,
                           self.chips[i]["objs"][k][1].representative_point().y] for k in nb])
            c = q[self.rng.integers(len(q))]
            cx, cy = c[0] + self.rng.normal(0, 40), c[1] + self.rng.normal(0, 40)
        elif r < 0.70 and self.with_obj:                   # on an object
            i = self.with_obj[self.rng.integers(len(self.with_obj))]
            g = self.chips[i]["objs"][self.rng.integers(len(self.chips[i]["objs"]))][1].representative_point()
            cx, cy = g.x + self.rng.normal(0, 70), g.y + self.rng.normal(0, 70)
        else:                                              # background (field sheets matter too)
            pool = self.empty or list(range(len(self.chips)))
            i = pool[self.rng.integers(len(pool))]
            h, w = self._img(self.chips[i]["file"]).shape[:2]
            cx, cy = self.rng.uniform(0, w), self.rng.uniform(0, h)
        c = self.chips[i]
        im = self._img(c["file"])
        s = float(np.exp(self.rng.uniform(np.log(SCALE_LO), np.log(SCALE_HI))))
        h, w = im.shape[:2]
        if ROT:
            # Arbitrary rotation about the sampled centre: source → crop is q = s·R(θ)(p − c) + CROP/2.
            # The frame may leave the chip; borderValue 0 plus the validity mask keep that padding out of the loss.
            th = float(self.rng.uniform(-np.pi, np.pi))
            ca, sa = np.cos(th) * s, np.sin(th) * s
            c0 = CROP / 2.0
            M = np.array([[ca, -sa, c0 - ca * cx + sa * cy],
                          [sa, ca, c0 - sa * cx - ca * cy]], np.float32)
            aff = [ca, -sa, sa, ca, float(M[0, 2]), float(M[1, 2])]   # shapely: a, b, d, e, xoff, yoff
        else:
            src = CROP / s
            x0 = float(np.clip(cx - src / 2, min(0, w - src), max(0, w - src)))
            y0 = float(np.clip(cy - src / 2, min(0, h - src), max(0, h - src)))
            M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
            aff = [s, 0, 0, s, -x0 * s, -y0 * s]
        out = cv2.warpAffine(np.ascontiguousarray(im), M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0)
        val = (out.sum(2) > 0).astype(np.float32)
        from shapely.affinity import affine_transform
        objs = [(cl, affine_transform(g, aff)) for cl, g in c["objs"]]
        img = out.astype(np.float32)
        # film photometry + degradation: the public sheets are unseen scans
        img = img * self.rng.uniform(0.7, 1.3) + self.rng.uniform(-30, 30)
        if self.rng.random() < 0.4:
            g = img.mean(2, keepdims=True); img = g + (img - g) * self.rng.uniform(0.0, 0.7)
        if self.rng.random() < 0.3:
            # film response; clip first — x**gamma on a negative pixel is NaN and poisons the whole run
            img = np.clip(img, 0, 255) ** float(self.rng.uniform(0.8, 1.25))
        if self.rng.random() < 0.3:
            k = int(self.rng.choice([3, 5])); img = cv2.GaussianBlur(img, (k, k), 0)
        if self.rng.random() < 0.5:
            if self.rng.random() < 0.4:
                f_ = self.rng.uniform(1.5, 3.0); m_ = max(8, int(CROP / f_))
                img = cv2.resize(cv2.resize(img, (m_, m_), interpolation=cv2.INTER_AREA), (CROP, CROP))
            if self.rng.random() < 0.35:
                ok, enc = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1],
                                       [cv2.IMWRITE_JPEG_QUALITY, int(self.rng.integers(20, 70))])
                img = cv2.imdecode(enc, 1)[:, :, ::-1].astype(np.float32)
            if self.rng.random() < 0.3:
                img = img + self.rng.normal(0, self.rng.uniform(3, 12), img.shape)
        x = ((np.clip(img, 0, 255).transpose(2, 0, 1) - MEAN) / STD * val).astype(np.float32)
        heat, mk = targets((CROP, CROP), objs)
        k = self.rng.integers(4); fl = self.rng.random() < 0.5
        arrs = [np.rot90(a, k, axes=(-2, -1)) for a in (x, heat, mk, val)]
        if fl:
            arrs = [a[..., ::-1] for a in arrs]
        return [np.ascontiguousarray(a) for a in arrs]

    def batch(self, n):
        b = [self.crop() for _ in range(n)]
        return [torch.from_numpy(np.stack([x[i] for x in b])) for i in range(4)]


def build_model(pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet("tu-convnext_tiny", encoder_weights="imagenet" if pretrained else None, in_channels=3,
                    classes=3, decoder_channels=(256, 128, 64, 32, 16))


def train(fold, iters=12000, bs=8, lr=3e-4, seed=0, tag=""):
    chips = [c for c in sp_train.load_chips("Ae") if c["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    S = Sampler(chips, rng)
    m = build_model().cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=iters, pct_start=0.05)
    # fixed dense probe: the 0.026 run died with a healthy-looking loss while peak activation stayed at 0.02-0.08,
    # so log peak probability on the densest chips next to the loss
    pS = Sampler(sorted(chips, key=lambda c: -len(c["objs"]))[:8], np.random.default_rng(12345))
    pimg, pheat, _, pval = [x.cuda() for x in pS.batch(4)]
    pv = pval[:, None] > 0
    pw = pval[:, None]

    # Probe the objective on a fixed dense batch before spending 17 minutes: a run whose optimum is not the target
    # (focal, wide sigma) or whose peaks cannot outweigh the background (count loss, object-poor batches) is doomed.
    with torch.no_grad():
        lg_of = lambda p: torch.logit(p.clamp(1e-4, 1 - 1e-4))
        if LOSS == "focal":
            vals = [(a / 20, focal_stable(lg_of(pheat * (a / 20)), pheat, pw).item()) for a in range(1, 41)]
            a_star, l_best = min(vals, key=lambda q: q[1])
            print(f"target sanity [focal] sig={SIG_FRAC}*sqrtA clip[{SIG_MIN},{SIG_MAX}] alpha*={a_star:.2f} "
                  f"L(alpha*) {l_best:.3f} L(1.0) {focal_stable(lg_of(pheat), pheat, pw):.3f} "
                  f"L(flat .02) {focal_stable(torch.full_like(pheat, 0.02), pheat, pw):.3f}", flush=True)
            if a_star < 0.35 and os.environ.get("AE_FORCE") != "1":
                raise SystemExit("focal optimum is a flat map (alpha* < 0.35): shrink AE_SIG_MAX/AE_SIG_FRAC, "
                                 "or use AE_LOSS=count (default). Override with AE_FORCE=1.")
        else:
            wt = pw * (1.0 + POS_W * (pheat > 0.05).float())
            share = float(wt[pheat > 0.05].sum() / wt.sum())
            print(f"target sanity [count] sig={SIG_FRAC}*sqrtA clip[{SIG_MIN},{SIG_MAX}] "
                  f"peaks {(pheat > 0.05).sum().item()} pos-mass share {share:.3%} "
                  f"L(target) {count_loss(lg_of(pheat), pheat, pw):.5f} "
                  f"L(flat .02) {count_loss(torch.full_like(pheat, 0.02), pheat, pw):.5f}", flush=True)
            if share < 1e-3 and os.environ.get("AE_FORCE") != "1":
                raise SystemExit("positive mass share < 0.1 %: the peak term cannot outweigh the background. "
                                 "Raise AE_POS_W or check the chip labels. Override with AE_FORCE=1.")
    heat_loss = focal_stable if LOSS == "focal" else count_loss

    def peak():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(pimg.to(memory_format=torch.channels_last))
        return torch.sigmoid(o[:, :1].float())[pv].max().item()

    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(6); futs = [ex.submit(S.batch, bs) for _ in range(3)]
    t0 = time.time()
    for i in range(iters):
        img, heat, mk, val = [x.cuda(non_blocking=True) for x in futs.pop(0).result()]
        futs.append(ex.submit(S.batch, bs))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(img.to(memory_format=torch.channels_last))
        w = val[:, None]
        loss = heat_loss(out[:, :1], heat, w) + hm.bce_dice(out[:, 1:], mk, w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step(); sched.step()
        if i % 250 == 0 or i == iters - 1:
            print(f"dense fold {fold} it {i} loss {loss.item():.4f} obj {int((heat > 0.999).sum())} "
                  f"peak {peak():.3f} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, f"dense{tag}_f{fold}.pt")
    torch.save({k: v.half() for k, v in m.state_dict().items()}, out)
    print("saved", out, flush=True)


def load(path):
    import segmentation_models_pytorch as smp  # noqa: F401
    m = build_model(pretrained=False).cuda().eval()
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cuda").items()})
    return m


def oof(tag="", scales=None, report=False):
    """Candidates with their distance to the sheet centre — the input of the framing-prior sweep."""
    import pandas as pd
    from pyproj import Transformer
    from rasterio.transform import Affine
    from shapely.geometry import Point, shape
    import importlib.util as u
    spec = u.spec_from_file_location("aep", os.path.join(ROOT, "scripts", "ae_hm_prior_oof.py"))
    aep = u.module_from_spec(spec); spec.loader.exec_module(aep)
    geom = aep.sheet_geometry()
    chips = sp_train.load_chips("Ae")
    rows = []
    for fold in sorted({c["fold"] for c in chips}):
        m = load(os.path.join(WORK, f"dense{tag}_f{fold}.pt"))
        feats = []
        for c in [c for c in chips if c["fold"] == fold]:
            rgb = cv2.imread(os.path.join(sp_train.prep_dir("Ae"), "chips", c["file"]))[:, :, ::-1]
            x, v = sp_train.norm_img(rgb)
            p = hm.predict_tiled(m, x, 3, tile=1024, overlap=0, scales=scales)
            t = Affine(*c["transform"]); tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
            for _, px, py, s in hm.decode(p[:1], v, thr=0.05, nms=9):
                yi, xi = int(py), int(px)
                y0, y1, x0, x1 = max(0, yi - 2), yi + 3, max(0, xi - 2), xi + 3
                pd, pi = float(p[1, y0:y1, x0:x1].mean()), float(p[2, y0:y1, x0:x1].mean())
                tot = pd + pi + 1e-6
                cand = [(KP, pd / tot), (KT, pi / tot)]
                best = max(cand, key=lambda q: q[1])
                emit = [best] + [q for q in cand if q is not best and q[1] >= 0.35]
                X, Y = tr.transform(*t * (px, py))
                kk = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(Y / 6378137.0)) - np.pi / 2)))
                for cl, pc in emit:
                    feats.append({"type": "Feature", "geometry": Point(X, Y).buffer(0.5 * kk, 12).__geo_interface__,
                                  "properties": {"class_name": sp_train.CLS[cl], "confidence": float(s * pc),
                                                 "region_name": c["region"], "markup_type": "ae",
                                                 "source_layer": c["scene"]}})
        for f in sp_train.dedupe(feats):
            pr = f["properties"]; g = shape(f["geometry"]).centroid
            key = f"{pr['region_name']}|{pr['source_layer']}"
            if key not in geom:
                continue
            X, Y, half = geom[key]
            rows.append(dict(region=pr["region_name"], scene=pr["source_layer"], cls=pr["class_name"],
                             score=pr["confidence"], x=g.x, y=g.y, fold=fold,
                             d=float(np.hypot(g.x - X, g.y - Y)), half=float(half)))
        print("oof fold", fold, len(rows), flush=True)
    D = pd.DataFrame(rows)
    if len(D):
        # local prediction density inside the same source sheet: the decision variable of the density ("carpet") gate.
        # Ae members work on cemetery sheets and fail on open fields, so the emitter must be allowed to speak only
        # where it is dense — this column lets ae_hm_prior_oof / ae_cluster_oof test that honestly.
        from scipy.spatial import cKDTree
        n120 = np.zeros(len(D), np.int32)
        for _, g in D.groupby(["region", "scene"]):
            pts = np.c_[g.x.values, g.y.values]
            nb = cKDTree(pts).query_ball_point(pts, 120.0)
            n120[g.index.values] = [len(t) for t in nb]
        D["n120"] = n120
    D.to_parquet(os.path.join(WORK, f"cands{tag}.parquet"))
    print(len(D), "candidates →", os.path.join(WORK, f"cands{tag}.parquet"))
    if report and len(D):
        sheet_report(D)
    return D


def sheet_report(D):
    """Per-sheet and per-density precision: can the emitter be gated to cemetery sheets?

    A candidate counts as a hit when its emitted point falls inside a same-class GT polygon (the metric's
    representative-point-inside rule, without the coverage term) — an upper bound on true TP, which is what we
    need in order to reject the emitter rather than accept it.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    L = gpd.read_parquet(os.path.join(DATA, "..", "labels_3857.parquet"))
    L = L[(L["mod"] == "Ae") & L.class_name.isin(sp_train.CLS)]
    trees, gt_n = {}, {}
    for reg, g in L.groupby("region"):
        trees[reg] = {cls: STRtree(list(gg.geometry)) for cls, gg in g.groupby("class_name")}
        gt_n[reg] = len(g)
    hit = np.zeros(len(D), bool)
    for i, (reg, cls, x, y) in enumerate(zip(D.region.values, D.cls.values, D.x.values, D.y.values)):
        t = trees.get(reg, {}).get(cls)
        if t is not None:
            hit[i] = bool(t.query(Point(x, y).buffer(0.5)).size)
    D = D.assign(hit=hit)
    print(f"\n=== Ae held-out precision: {int(D.hit.sum())}/{len(D)} = {D.hit.mean():.3f} "
          f"(emitting at the lowest threshold, before any gate) ===")
    if "n120" in D:
        print("\ndensity gate (candidates within 120 m in the same sheet):")
        bins = [0, 1, 2, 5, 11, 24, 49, 10 ** 9]
        cut = pd.cut(D.n120, bins, right=True, include_lowest=True)
        t = D.groupby(cut, observed=True).agg(cands=("hit", "size"), hits=("hit", "sum"))
        t["precision"] = (t.hits / t.cands).round(3)
        t["recall_share"] = (t.hits / max(int(D.hit.sum()), 1)).round(3)
        print(t.to_string())
    print("\nworst / best sheets by emitted count (top 15):")
    s = D.groupby(["region", "scene"]).agg(cands=("hit", "size"), hits=("hit", "sum"))
    s["precision"] = (s.hits / s.cands).round(3)
    s["gt_in_region"] = [gt_n.get(r, 0) for r, _ in s.index]
    print(s.sort_values("cands", ascending=False).head(15).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int)
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="_dense")
    ap.add_argument("--scales", default="", help="inference scale pyramid, e.g. 1.0,0.8,1.25 (OOF only)")
    ap.add_argument("--report", action="store_true", help="print per-sheet / per-density precision after --oof")
    a = ap.parse_args()
    sc = [[float(s), 1.0] for s in a.scales.split(",") if s.strip()] or None
    if a.fold is not None:
        train(a.fold, a.iters, a.bs, a.lr, a.seed, a.tag)
    if a.oof:
        oof(a.tag, scales=sc, report=a.report)
