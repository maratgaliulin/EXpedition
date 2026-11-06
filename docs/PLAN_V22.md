# v22 program — where the remaining points are, and what we change

Baseline: **v20 / 0.50190** (`archive/koz3_submission_v20_layers_ae_fort_20261027`), reproduced bit-identically by
`build/rebuild1.zip`. Platform per-modality P/R: LiDAR 0.596/0.676, satellite 0.4342/0.6393, geophysics 0.754/0.724,
Or 0.706/0.240, Ae 0.103/0.093.

Everything below respects the ledger's acceptance bar (`docs/EXPERIMENTS.md`, 25 Oct):
**ship only if k\* ≥ 5 and bootstrap P ≥ 0.9**, decided on site-disjoint folds, never on the public LB.

## Lever map (why these four, and not the obvious ones)

Public GT composition recovered from integer P/R solutions (ledger, 25 Oct): LiDAR 482, satellite 377, Or 25,
geophysics 1441 (finds-dominated), **Ae ≈ 525–600 polygons — the largest single polygon set, on which we score ~0.10**.

| Weight | Class | Current | Where the loss is |
|---:|---|---|---|
| 30 | damaged kurgans | ~0.45 | Ae missed entirely (~21 % of kp labels), satellite kp precision 0.43, LiDAR blind sites |
| 20 | intact kurgans | ~0.50 | class confusion (hedge deployed), same modality limits |
| 15 | hillforts | ~0.55 | localisation in big scenes, 5 000–10 000 m² recall 0.00 |
| 10 | architecture | ~0.35 | Or recall 0.24 (only 4 training sites) |
| 10 | object_poly | ~0.55 | MagGr precision, TerFl 2 regions |
| 5 | fortifications | ~0.10 | LiDAR trenches 23/264, Ae forts only |
| 10 | finds_points | ~0.74 | at the physics+GBM optimum |

Rejected already (do not retry): rank/WBF stacking, per-region recalibration, model soup, mosaic satellite training,
coarse 2 m branch, zoom TTA for D-FINE, FCOS as a fusion member, the physics/geometry/TDA/scattering candidate
pipelines, learned LiDAR post-processors, LiDAR site self-training, KDE label prior, Ae framing prior and
cemetery-cell clustering on **box** detections.

## P0 — Ae dense film model (the one open lever the ledger itself names)

The ledger twice states the lever verbatim: *"a dense-object (point-heatmap / counting) film model without detection
caps, trained on cemetery sheets"*. Box detectors cap at 100 dets / NMS 0.5, and Тимерево alone has 498 labels in one
chip — that is the whole Ae story.

State on 5 Nov: `scripts/ae_dense_hm.py` trains but goes flat (max heat 0.057 after 12k iters). Mechanism proven by
the script's own target-sanity probe:

```
perfect 0.370  half 0.213  flat0.02 3.771   <-- BAD: half scores better than perfect
```

With σ = 6 px there are ~50 000 halo pixels per batch, each weakly penalised by `(1−t)^4`; predicting *half* the
target everywhere removes more halo cost than it pays at the 1 px peaks. The fix is not a wider Gaussian, it is a
**tight** one (σ = 2–4 px, as in `koz3.hm.draw_targets`) plus the positive-count floor that removes the original
1–2-positive-pixel instability:

1. σ from `SIG_FRAC·sqrt(area)` clipped to [2, 4] px (was [2, 6] with frac 0.10 on a halo that swamps the objective);
2. the sanity probe becomes a **hard gate** — a run whose objective prefers `p = t/2` must not start;
3. continuous rotation augmentation (currently only the D4 group: 4×rot90 + flip) — aerial film has no canonical
   orientation, so this is 8 → ∞ effective augmentation for free;
4. per-sheet precision report and a **prediction-density gate**: single members already work on cemetery sheets
   (Тимерево 90 % in-label) and fail on open-field sheets (1–11 %), so the model must be allowed to speak only where
   it is dense. The old version of this filter was tested on capped box detectors and had no volume; a peak decoder
   has no cap.

Kill criteria: no fold reaches ≥ 0.20 held-out precision on the emitting subset, or the density-gated honest gain
(nested, site folds) is ≤ 0. → Ae stays off and the lever is closed for good.

## P1 — LiDAR scale-pyramid TTA switched on (validated, unshipped)

Four-fold honest result (59 held-out sites, deployed decode): base 0.5730 → **1.0/0.8/1.25 pyramid 0.5889 (+0.0160)**,
positive in 3/4 folds, worst fold −0.003, 25/59 sites better. Already implemented and budget-guarded
(`li.scale_tta` in `koz3_infer/experts/las.py` + `koz3.hm.predict_tiled(scales=…)`), shipped **off**.

Expected platform effect +0.003…+0.004 (kp/kt only). Not worth a submission alone — ship with P0. Cost 1.5–3× LiDAR
inference; the expert already drops extra scales when the budget is short.

## P2 — Ensemble diversity for satellite (new member family, not a 7th correlated one)

Six satellite members are D-FINE/RT-DETR (one ConvNeXt-T heatmap). The ledger's "+0.008 per extra member" applies to
that correlated family; replacing the single heatmap with a retrained one gave +0.003 in the fusion pipeline.
So the ensemble change with real headroom is a **decorrelated heatmap member**:

* different encoder (`SP_ENC=tu-efficientnet_b3`, then `tu-swin_tiny` if it trains), same folds, same site balance;
* continuous rotation + wider scale jitter (`SP_ROT=1`, `SP_SCALELO/SP_SCALEHI`), which is also the augmentation
  change for satellite: the current sampler only sees the D4 group and 0.7–1.4 zoom;
* accept only if the **fused** satellite pipeline (6 members + gate + framing prior + hedge) improves nested;

## P3 — LiDAR ensemble diversity (queued in the ledger, never run)

`LI_ENC=tu-swin_tiny` U-Net (fixed 512 px windows) as a 5th blend member. The v11 blend is 4 correlated ConvNeXt-T
models; a transformer encoder is the cheapest genuinely different inductive bias we can add. Accept only if the
deployed blend improves nested.

## Runbook

```bash
# P0 — Ae dense model (per fold; ~17 min/fold on the RTX 5080)
./myenv/bin/python scripts/ae_dense_hm.py --fold 0 --tag _v22 --iters 12000
./myenv/bin/python scripts/ae_dense_hm.py --fold 1 --tag _v22 --iters 12000
./myenv/bin/python scripts/ae_dense_hm.py --fold 2 --tag _v22 --iters 12000
./myenv/bin/python scripts/ae_dense_hm.py --fold 3 --tag _v22 --iters 12000
# candidates + per-sheet precision / density curve (no training)
./myenv/bin/python scripts/ae_dense_hm.py --oof --tag _v22 --report
# honest framing/density sweep on those candidates
./myenv/bin/python scripts/ae_hm_prior_oof.py --sweep --tag _v22

# P2 — decorrelated satellite heatmap member (fold 2 = deployment fold)
SP_ENC=tu-efficientnet_b3 SP_SITE_BAL=1 SP_QAUG=1 SP_ROT=1 \
  ./myenv/bin/python -m koz3.sp_train --mod SpOr --fold 2 --tag _eb3 --iters 12000

# P3 — Swin LiDAR member
LI_ENC=tu-swin_tiny LI_AUG=1 LI_PASTE=1 RELIEF_EXTRA=1 \
  ./myenv/bin/python -m koz3.li_train --fold 2 --tag _swin --iters 12000

# P1 — config with the pyramid on (base = deployed v20)
./myenv/bin/python scripts/patch_archive.py \
  --base archive/koz3_submission_v20_layers_ae_fort_20261027/../..  # see docs/WORKSPACE_README.md for the zip path
```

## Order of operations

1. P0 fold 0 → check the sanity gate passes and `peak` leaves 0.05. If it does not, stop and re-derive the target —
   do not spend the other three folds.
2. P0 all folds → `--oof --report` → decide the density gate on held-out sheets.
3. In parallel: P2 member (one fold) and P1 config; P3 last (GPU-bound, EV lowest).
4. Promote to a build only what passes k\* ≥ 5 / P ≥ 0.9, then rehearse with
   `scripts/rehearse.py` and patch the archive with `scripts/patch_archive.py` (never re-zip the tree).

---

## Status 2026-11-06 — checkpoint (git `94d2bee`)

**Landed (committed):**
`docs/PLAN_V22.md`, `configs/v22_candidate.json`, `scripts/ae_dense_hm.py` (counting loss, tight σ, hard sanity gate,
continuous-rotation aug, `--report` per-sheet + density precision), `koz3/sp_train.py` (`SP_ROT`, `SP_SCALELO/HI`,
defaults preserve the proven recipe bit-for-bit).

**P0 measurement that changed the design.** The 5 Nov failure was attributed to "target too small → 1–2 positive px".
That is only half of it. Scanning the loss-optimal amplitude α\* (p = α·t) on a lone Gaussian:

| σ (px) | halo px | α\* | L(1.0) | L(α\*) |
|---:|---:|---:|---:|---:|
| 1.5 | 68 | 0.75 | 0.172 | 0.085 |
| 2.0 | 112 | 0.70 | 0.307 | 0.128 |
| 4.0 | 464 | 0.55 | 1.229 | 0.300 |
| 6.0 | 1 040 | **0.45** | 2.765 | 0.457 |

The focal objective's minimum is **never** at p = t, and widening the Gaussian (the previous "fix") moves the optimum
*toward* a flat map — which is the observed outcome (max heat 0.057 after 12k iterations at σ = 6). The first
`half < perfect` gate also mis-fired: it compared only the family {t, t/2, flat}, so it rejected σ = 2–4 as well.

**Replacement:** `AE_LOSS=count` — weighted MSE on the Gaussian density map (MCNN / CSRNet family), the standard
objective for dense counting. MSE is separable per pixel, so its optimum **is** p = t at every pixel regardless of
weights; `AE_POS_W` (default 50) only keeps the "predict 0 everywhere" map expensive. Positive mass share on a real
dense probe batch: **80.7 %**, L(target) 0.00000 vs L(flat 0.02) 0.123.

**Smoke run** (`--fold 0 --iters 600 --tag _smoke2`, 54 s): sanity gate passes, `peak` 0.245 → 0.181 → 0.193 on the
fixed dense probe. Not evidence of a working model (the broken 12k run ended at 0.057, so the floor is 3× higher, but
600 iters is far too early) — the first real run is the decision point.

**Open item for Monday, check this first.** Total loss is still dominated by the mask head: `hm.bce_dice` ≈ 1.0 while
the counting term is ≈ 0.01–0.12. The heat head may therefore be under-trained. Try `loss = 10·count_loss + bce_dice`
(or `count_loss + 0.1·bce_dice`) on a 1k-iteration A/B before spending 4 × 17 min on folds.

**Not started:** no `_v22` fold training, no satellite `_eb3` member, no Swin LiDAR member, no rehearsal of the
pyramid config. GPU idle, `work/ae/dense_smoke2_f0.pt` is a 600-iteration smoke artifact (not usable);
the 5 Nov `work/ae/dense_*` checkpoints are the flat runs.

**Monday, in order:**
```bash
# 1. loss-balance A/B (two 1k runs, ~2 min each), pick the better peak curve
AE_LOSS=count ./myenv/bin/python scripts/ae_dense_hm.py --fold 0 --iters 1000 --tag _bal10
#    (edit the heat term multiplier in scripts/ae_dense_hm.py between the two runs)

# 2. four folds (~17 min each), tag _v22
for f in 0 1 2 3; do ./myenv/bin/python scripts/ae_dense_hm.py --fold $f --tag _v22; done

# 3. candidates + per-sheet / density precision table → decide the carpet gate
./myenv/bin/python scripts/ae_dense_hm.py --oof --tag _v22 --report

# 4. honest sweep on those candidates
./myenv/bin/python scripts/ae_hm_prior_oof.py --sweep --tag _v22
```

Kill the lever if no fold reaches ≥ 0.20 held-out precision on the emitting subset after the density gate.
