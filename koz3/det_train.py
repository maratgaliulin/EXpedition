"""Box detector (HF D-FINE / RT-DETRv2, Apache-2.0) fine-tuned on koz3.sp_prep chips.

Differences from the previous team's recipe: small screenshot scenes are seen whole at their native framing
(the site fills the frame — context for hillforts), random scale 0.6-1.3, 768 px crops, folds shared with
D-FINE / heatmap OOF. Output: work/<mod>/det_<tag>_f<fold>.pt and OOF candidates (region, cls, score, x, y in 3857).

  python -m koz3.det_train --mod SpOr --fold 0 --ckpt ustc-community/dfine-large-coco --tag dfl
  python -m koz3.det_train --mod SpOr --oof --tag dfl
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3.sp_train import CLS, load_chips, prep_dir, ROOT  # noqa: E402

CROP = 768
# mixed precision for training: "fp16" (default), "bf16" or "fp32" (env DET_AMP)
AMP_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[os.environ.get("DET_AMP", "fp16")]
# D-FINE denoising queries make local training collapse: the classifier is learned on the GT-derived denoising
# queries, so eval-mode max class score stays ~0.15 and F1 is 0.000 (docs/EXPERIMENTS.md, "Why HF D-FINE did not
# train here"). 0 is the recipe every local satellite member was trained with, and the exported config mirrors it
# (scripts/build_submission.py sets cfg.num_denoising = 0). Keep it at 0 unless re-running that comparison.
NUM_DN = int(os.environ.get("DET_NUM_DN", "0"))
if os.environ.get("NO_TF32") == "1":     # TF32 (Ada default for cuDNN convs) distorts D-FINE gradients by ~22 %
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False


class Sampler:
    def __init__(self, chips, mod, rng):
        self.chips, self.rng, self.dir = chips, rng, os.path.join(prep_dir(mod), "chips")
        n = np.array([len(c["objs"]) for c in chips])
        self.p = (0.5 + np.sqrt(n)) * np.array([c["valid_frac"] for c in chips]); self.p /= self.p.sum()
        self.cache = {}

    def img(self, f):
        im = self.cache.get(f)
        if im is None:
            im = cv2.imread(os.path.join(self.dir, f))[:, :, ::-1]
            if len(self.cache) < 3000:
                self.cache[f] = im
        return im

    def crop(self):
        r = self.rng
        c = self.chips[r.choice(len(self.chips), p=self.p)]
        im = self.img(c["file"]); h, w = im.shape[:2]
        s = float(np.exp(r.uniform(np.log(0.6), np.log(1.3))))
        src = CROP / s
        if c["objs"] and r.random() < 0.6:
            g = c["objs"][r.integers(len(c["objs"]))][1].representative_point()
            cx, cy = g.x + r.uniform(-src / 3, src / 3), g.y + r.uniform(-src / 3, src / 3)
        else:
            cx, cy = r.uniform(0, w), r.uniform(0, h)
        x0 = float(np.clip(cx - src / 2, min(0, w - src), max(0, w - src)))
        y0 = float(np.clip(cy - src / 2, min(0, h - src), max(0, h - src)))
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        out = cv2.warpAffine(np.ascontiguousarray(im), M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderValue=0)
        B = []
        for cl, g in c["objs"]:
            bx0, by0, bx1, by1 = g.bounds
            b = np.array([(bx0 - x0) * s, (by0 - y0) * s, (bx1 - x0) * s, (by1 - y0) * s])
            area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
            bc = np.clip(b, 0, CROP)
            if (bc[2] - bc[0]) * (bc[3] - bc[1]) >= 0.5 * area and bc[2] - bc[0] >= 2 and bc[3] - bc[1] >= 2:
                B.append([cl, *bc])
        B = np.array(B, np.float32).reshape(-1, 5)
        x = out.astype(np.float32).transpose(2, 0, 1)
        vd = (x.sum(0, keepdims=True) > 0)
        x = x * r.uniform(0.75, 1.3, (3, 1, 1)) * r.uniform(0.85, 1.15) + r.uniform(-20, 20, (3, 1, 1))
        if r.random() < 0.25:
            k = int(r.choice([3, 5])); x = np.stack([cv2.GaussianBlur(ch, (k, k), 0) for ch in x])
        if r.random() < 0.15:
            x = np.repeat(x.mean(0, keepdims=True), 3, 0)
        x = np.clip(x, 0, 255) * vd / 255.0
        for _ in range(r.integers(4)):     # rot90 ccw: (x, y) → (y, n - x)
            x = np.rot90(x, 1, axes=(1, 2)); B = B[:, [0, 2, 1, 4, 3]].copy(); B[:, 2], B[:, 4] = CROP - B[:, 4], CROP - B[:, 2]
        if r.random() < 0.5:
            x = x[:, :, ::-1]; B = B.copy(); B[:, 1], B[:, 3] = CROP - B[:, 3], CROP - B[:, 1]
        return np.ascontiguousarray(x, dtype=np.float32), B

    def batch(self, n):
        xs, ts = [], []
        for _ in range(n):
            x, B = self.crop()
            cx, cy = (B[:, 1] + B[:, 3]) / 2 / CROP, (B[:, 2] + B[:, 4]) / 2 / CROP
            bw, bh = (B[:, 3] - B[:, 1]) / CROP, (B[:, 4] - B[:, 2]) / CROP
            xs.append(x)
            ts.append({"class_labels": torch.as_tensor(B[:, 0], dtype=torch.long),
                       "boxes": torch.as_tensor(np.stack([cx, cy, bw, bh], 1), dtype=torch.float32).reshape(-1, 4)})
        return torch.from_numpy(np.stack(xs)), ts


def remap_state(sd, model_keys):
    """transformers 5.x ↔ 4.x key names of D-FINE / RT-DETR (mlp.layers.0/1 ↔ fc1/fc2, o_proj ↔ out_proj,
    encoder.aifi ↔ encoder.encoder): all renames applied together in whichever direction matches the model."""
    if set(sd) <= set(model_keys):
        return sd
    pairs = [(".mlp.layers.0.", ".fc1."), (".mlp.layers.1.", ".fc2."), (".o_proj.", ".out_proj."), (".encoder.aifi.", ".encoder.encoder.")]
    out = {}
    for k, v in sd.items():
        if k not in model_keys:
            fwd, rev = k, k
            for a, b in pairs:
                fwd = fwd.replace(a, b); rev = rev.replace(b, a)
            k = fwd if fwd in model_keys else (rev if rev in model_keys else k)
        out[k] = v
    return out


def make_model(ckpt):
    from transformers import AutoModelForObjectDetection
    return AutoModelForObjectDetection.from_pretrained(ckpt, num_labels=len(CLS), id2label=dict(enumerate(CLS)),
                                                       label2id={c: i for i, c in enumerate(CLS)},
                                                       num_denoising=NUM_DN, ignore_mismatched_sizes=True)


def train(mod, fold, ckpt, tag, iters=10000, bs=4, lr=2e-4, seed=0, init=None, probe_every=0):
    chips = [c for c in load_chips(mod) if c["fold"] != fold]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    S = Sampler(chips, mod, rng)
    model = make_model(ckpt)
    if init:   # start from an existing fine-tuned checkpoint (e.g. the previous team's fold model)
        sd = remap_state(torch.load(init, map_location="cpu"), set(model.state_dict()))
        # non-strict: a checkpoint trained WITH denoising queries carries layers this model does not have (and vice
        # versa); everything else must line up, so report what was skipped instead of failing the whole run
        miss, unexp = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
        print(f"init {init}: {len(miss)} missing, {len(unexp)} unexpected tensors (denoising layers expected)", flush=True)
    model = model.cuda().train()
    probe = [c for c in load_chips(mod) if c["fold"] == fold and len(c["objs"]) >= 3][:8] if probe_every else []
    bb = [p for n, p in model.named_parameters() if "backbone" in n]
    oth = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr * 0.1}, {"params": oth, "lr": lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr * 0.1, lr], total_steps=iters, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=AMP_DTYPE == torch.float16)
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(4); futs = [ex.submit(S.batch, bs) for _ in range(3)]
    t0, run = time.time(), None
    for it in range(iters):
        x, ys = futs.pop(0).result(); futs.append(ex.submit(S.batch, bs))
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_DTYPE is not None):
            out = model(pixel_values=x.cuda(non_blocking=True), labels=[{k: v.cuda() for k, v in y.items()} for y in ys])
        opt.zero_grad(set_to_none=True); scaler.scale(out.loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); scaler.step(opt); scaler.update(); sch.step()
        run = out.loss.item() if run is None else 0.98 * run + 0.02 * out.loss.item()
        if it % 500 == 0 or it == iters - 1:
            print(f"det {tag} fold {fold} it {it} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if probe_every and (it % probe_every == 0 or it == iters - 1):
            model.eval(); mx = []
            with torch.no_grad():
                for c in probe:
                    im = cv2.imread(os.path.join(S.dir, c["file"]))[:, :, ::-1]
                    t = np.zeros((CROP, CROP, 3), np.float32); r = im[:CROP, :CROP]; t[:r.shape[0], :r.shape[1]] = r / 255
                    o = model(pixel_values=torch.from_numpy(t.transpose(2, 0, 1).copy())[None].cuda())
                    mx.append(float(o.logits.sigmoid().max()))
            model.train()
            print(f"   probe it {it}: held-out max class score mean {np.mean(mx):.3f}", flush=True)
    d = os.path.join(ROOT, "work", mod.lower()); os.makedirs(d, exist_ok=True)
    torch.save({k: v.half() for k, v in model.state_dict().items()}, os.path.join(d, f"det_{tag}_f{fold}.pt"))


@torch.no_grad()
def detect_chip(model, rgb, tile=CROP, ov=192, min_score=0.05):
    """rgb (H,W,3) uint8 chip → list of (cls, cx, cy, w, h, score) in chip pixels (flip TTA, class-aware NMS)."""
    import torchvision
    H, W = rgb.shape[:2]
    x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    step = tile - ov
    B, S, C = [], [], []
    for y0 in range(0, max(1, H - ov), step):
        for x0 in range(0, max(1, W - ov), step):
            t = np.zeros((3, tile, tile), np.float32)
            p = x[:, y0:y0 + tile, x0:x0 + tile]; t[:, :p.shape[1], :p.shape[2]] = p
            tt = torch.from_numpy(t)[None].cuda()
            outs = []
            for fl in (False, True):
                z = tt.flip(-1) if fl else tt
                with torch.autocast("cuda", dtype=torch.float16):
                    lg, bx = model(z)
                lg, bx = lg.float().sigmoid()[0], bx.float()[0].clone()
                if fl:
                    bx[:, 0] = 1 - bx[:, 0]
                outs.append((lg, bx))
            lg = torch.cat([o[0] for o in outs]); bx = torch.cat([o[1] for o in outs])
            sc, cl = lg.max(1); m = sc >= min_score
            if not m.any():
                continue
            b = bx[m] * tile
            B.append(torch.stack([b[:, 0] - b[:, 2] / 2 + x0, b[:, 1] - b[:, 3] / 2 + y0, b[:, 0] + b[:, 2] / 2 + x0, b[:, 1] + b[:, 3] / 2 + y0], 1))
            S.append(sc[m]); C.append(cl[m])
    if not B:
        return []
    B, S, C = torch.cat(B), torch.cat(S), torch.cat(C)
    keep = torchvision.ops.batched_nms(B, S, C, 0.5)
    out = []
    for i in keep.tolist():
        x0, y0, x1, y1 = B[i].tolist()
        out.append((int(C[i]), (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, float(S[i])))
    return out


class _Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        o = self.m(pixel_values=x)
        return o.logits, o.pred_boxes


def load_eager(ckpt, path):
    """Eager wrapped model (x → (logits, boxes)); TorchScript traces of D-FINE take > 10 min per call."""
    m = make_model(ckpt)
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    return _Wrap(m.cuda().eval()).eval()


def load_traced(ckpt, path, tile=CROP):
    m = make_model(ckpt)
    m.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
    w = _Wrap(m.cuda().eval()).eval()
    with torch.no_grad():
        return torch.jit.trace(w, torch.rand(1, 3, tile, tile, device="cuda"), check_trace=False, strict=False)


def oof(mod, ckpt, tag):
    import pandas as pd
    from pyproj import Transformer
    from rasterio.transform import Affine
    chips = load_chips(mod)
    rows = []
    for fold in sorted({c["fold"] for c in chips}):
        model = load_eager(ckpt, os.path.join(ROOT, "work", mod.lower(), f"det_{tag}_f{fold}.pt"))
        for c in [c for c in chips if c["fold"] == fold]:
            rgb = cv2.imread(os.path.join(prep_dir(mod), "chips", c["file"]))[:, :, ::-1]
            t = Affine(*c["transform"]); tr = Transformer.from_crs(c["crs"], 3857, always_xy=True)
            for cl, cx, cy, bw, bh, s in detect_chip(model, rgb):
                X, Y = tr.transform(*(t * (cx, cy)))
                rows.append((c["region"], CLS[cl], s, X, Y, bw, bh, c["scene"]))
        print("oof fold", fold, len(rows), flush=True)
    # source_layer = the source image: the OOF harness fuses within an image, exactly like the deployed expert
    df = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y", "w_px", "h_px", "source_layer"])
    df.to_parquet(os.path.join(ROOT, "work", mod.lower(), f"oof_det_{tag}_cands.parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mod", default="SpOr"); ap.add_argument("--fold", type=int); ap.add_argument("--oof", action="store_true")
    ap.add_argument("--ckpt", default="ustc-community/dfine-large-coco"); ap.add_argument("--tag", default="dfl")
    ap.add_argument("--iters", type=int, default=10000); ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--init"); ap.add_argument("--probe_every", type=int, default=0)
    a = ap.parse_args()
    if a.fold is not None:
        train(a.mod, a.fold, a.ckpt, a.tag, a.iters, a.bs, a.lr, init=a.init, probe_every=a.probe_every)
    if a.oof:
        oof(a.mod, a.ckpt, a.tag)
