"""Generic CenterNet-style heads shared by the LiDAR / optical experts.

Targets per class: Gaussian peak at each object's representative point (the point the metric tests) with
sigma ∝ object size, plus the object mask. Decoding: max-pool NMS peaks → score = peak height.
"""
import numpy as np
import torch
import torch.nn.functional as F


def draw_targets(shape, objects, n_cls, sig_frac=0.12, sig_min=1.5, sig_max=40.0):
    """objects: list of (cls, polygon in pixel coords). → hm (n_cls,H,W) float32, mask (n_cls,H,W) uint8."""
    from rasterio.features import rasterize
    H, W = shape
    hm = np.zeros((n_cls, H, W), np.float32)
    mk = np.zeros((n_cls, H, W), np.uint8)
    by_cls = {}
    for c, g in objects:
        by_cls.setdefault(c, []).append(g)
        p = g.representative_point()
        size = np.sqrt(max(g.area, 1.0))
        s = float(np.clip(sig_frac * size, sig_min, sig_max))
        r = int(3 * s)
        cx, cy = int(round(p.x)), int(round(p.y))
        x0, x1, y0, y1 = max(0, cx - r), min(W, cx + r + 1), max(0, cy - r), min(H, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gk = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)).astype(np.float32)
        hm[c, y0:y1, x0:x1] = np.maximum(hm[c, y0:y1, x0:x1], gk)
        if 0 <= cy < H and 0 <= cx < W:
            hm[c, cy, cx] = 1.0
    for c, gs in by_cls.items():
        mk[c] = rasterize([(g, 1) for g in gs if not g.is_empty], out_shape=(H, W), dtype=np.uint8, all_touched=True)
    return hm, mk


def focal_hm(logit, hm, w):
    """CenterNet focal loss; w broadcastable weight mask (valid area)."""
    p = torch.sigmoid(logit.float()).clamp(1e-4, 1 - 1e-4)
    pos = hm.ge(0.999).float()
    pl = -torch.log(p) * (1 - p) ** 2 * pos
    nl = -torch.log(1 - p) * p ** 2 * (1 - hm) ** 4 * (1 - pos)
    return ((pl + nl) * w).sum() / (pos * w).sum().clamp(min=1)


def bce_dice(logit, t, w):
    logit = logit.float()
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * w).sum() / w.expand_as(t).sum().clamp(min=1)
    p = torch.sigmoid(logit) * w
    dice = 1 - (2 * (p * t).sum((0, 2, 3)) + 1) / (p.sum((0, 2, 3)) + (t * w).sum((0, 2, 3)) + 1)
    return bce + dice.mean()


def bce_softf1(logit, t, w, cls_w=(30.0, 20.0, 15.0, 5.0)):
    """Metric-shaped mask loss: BCE + 1 − weighted soft-F1 with the metric's 0.5·FP cost, per class weight."""
    logit = logit.float()
    bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * w).sum() / w.expand_as(t).sum().clamp(min=1)
    p = torch.sigmoid(logit) * w; tw = t * w
    tp = (p * tw).sum((0, 2, 3)); fp = (p * (1 - t) * w).sum((0, 2, 3)); fn = ((1 - p) * tw).sum((0, 2, 3))
    cw = torch.tensor(cls_w[:logit.shape[1]], device=logit.device, dtype=tp.dtype)
    f1 = (2 * (cw * tp).sum() + 1) / (2 * (cw * tp).sum() + (cw * fn).sum() + 0.5 * (cw * fp).sum() + 1)
    return bce + (1 - f1)


def decode(hm, valid=None, thr=0.05, nms=9):
    """hm (C,H,W) numpy → list of (cls, x, y, score)."""
    t = torch.from_numpy(np.ascontiguousarray(hm))[None]
    mx = F.max_pool2d(t, nms, 1, nms // 2)
    keep = ((t == mx) & (t > thr))[0].numpy()
    if valid is not None:
        keep &= valid[None]
    out = []
    for c, y, x in zip(*np.nonzero(keep)):
        out.append((int(c), float(x) + 0.5, float(y) + 0.5, float(hm[c, y, x])))
    return out


@torch.no_grad()
def predict_tiled(model, img, n_out, tile=1024, overlap=192, tta=True, dev="cuda"):
    """img (C,H,W) float32 numpy → (n_out,H,W) sigmoid maps, tiled with flip TTA and linear blending."""
    C, H, W = img.shape
    out = np.zeros((n_out, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    step = tile - overlap
    # overlap=0 (single pass over a chip) would divide by zero here: keep a 1-px ramp so the blend weights stay finite
    ov = max(overlap, 1)
    ramp = np.minimum(np.arange(tile) + 1, ov) / ov
    for y in list(range(0, max(1, H - overlap), step)):
        for x in list(range(0, max(1, W - overlap), step)):
            y1, x1 = min(H, y + tile), min(W, x + tile)
            t = img[:, y:y1, x:x1]
            h, w = t.shape[1:]
            ph, pw = (-h) % 32, (-w) % 32
            tt = torch.from_numpy(np.pad(t, ((0, 0), (0, ph), (0, pw)), mode="reflect" if min(h, w) > max(ph, pw) else "constant")).float()[None].to(dev)
            acc = 0
            if tta == "d8" and tt.shape[-1] == tt.shape[-2]:        # 4 rotations × 2 flips on full square tiles only
                # (padding small sites to a square and rotating them gave no gain: OOF 0.5651 vs 0.5695)
                views = [(k, fl) for k in range(4) for fl in (False, True)]
            else:
                views = [(0, ()), (0, (-1,)), (0, (-2,))] if tta else [(0, ())]
            for k, fl in views:
                z = torch.rot90(tt, k, dims=(-2, -1)) if k else tt
                if fl is True:
                    z = z.flip(-1)
                elif fl:
                    z = z.flip(fl)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = torch.sigmoid(model(z).float())
                if fl is True:
                    o = o.flip(-1)
                elif fl:
                    o = o.flip(fl)
                acc = acc + (torch.rot90(o, -k, dims=(-2, -1)) if k else o)
            o = (acc / len(views))[0, :, :h, :w].cpu().numpy()
            wy = np.minimum(ramp[:h], ramp[:h][::-1]) if (y > 0 or y1 < H) else np.ones(h)
            wx = np.minimum(ramp[:w], ramp[:w][::-1]) if (x > 0 or x1 < W) else np.ones(w)
            if y == 0: wy[:overlap] = 1
            if y1 == H: wy[-overlap:] = 1
            if x == 0: wx[:overlap] = 1
            if x1 == W: wx[-overlap:] = 1
            wgt = np.outer(wy, wx).astype(np.float32)
            out[:, y:y1, x:x1] += o * wgt; cnt[y:y1, x:x1] += wgt
    return out / np.maximum(cnt, 1e-6)
