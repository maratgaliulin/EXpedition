# Experiments (official metric koz3-1.5.0, out-of-fold, no leakage)

Protocol: folds are geographic (20 km site clusters for LiDAR / optical, region groups or leave-one-region-out
for geophysics); every post-processing choice applied to a held-out fold is selected on the other folds only
(`koz3/nested.py`). "Honest" = nested; "oracle" = best setting picked on all OOF (upper bound, never reported as
the score). Scorer `koz3/score.py` is checked equal to the official script on random trials.

## Geophysics

| Date | Experiment | Finds F1 | Objects F1 | Decision |
|---|---|---:|---:|---|
| 22 Oct | CNN v1 (U-Net R34, raw grey input, 5 region-group folds) | 0.383 | **0.357** | objects: deploy |
| 22 Oct | CNN v2 (detrended input: lum, lum−G(1 m), lum−G(3 m)) | 0.681 | 0.304 | finds +0.30 vs v1; objects worse |
| 22 Oct | previous team's physics candidates + LightGBM (LORO, reproduced locally, exact) | **0.7187** | | finds: deploy |
| 22 Oct | stacking CNN v2 heatmap as a LightGBM feature (LORO) | 0.681 | | **rejected**: CNN confidently wrong on unseen map styles (200: 0.29 vs 0.46, 211: 0.11 vs 0.60) |

| 23 Oct | previous team's pixel-LightGBM objects (LORO, reproduced locally) | | 0.348 | |
| 23 Oct | objects nested on our folds: CNN v1 alone / GBM alone | | 0.327 / 0.321 | (CNN 0.357 above was pooled-oracle) |
| 23 Oct | **objects 0.5·CNN + 0.5·GBM (reprojected), thr / size nested** | | **0.353** | **deploy** |

Per region (finds, LORO): Semibratnee maps 0.73–0.86; Gnezdovo smoothed terraces 0.35–0.60; 212 colour-only map 0.34.

## Satellite (SpOr), 129 regions, 4 shared site folds

| Experiment | Honest | Oracle | Decision |
|---|---:|---:|---|
| D-FINE-L (previous team) | 0.2581 | 0.2581 | |
| RT-DETRv2-R50 (previous team) | — | 0.2397 | |
| our heatmap U-Net (ConvNeXt-T, representative-point heatmaps) | — | 0.2061 | alone weaker |
| **fusion D-FINE + RT-DETRv2 + heatmap, equal weights, thr 0.25** | **0.2729** | 0.2729 | deploy (same threshold chosen in all 4 folds) |
| + hillfort localiser (segmentation on whole scenes) | 0.2707 | | **rejected**: 1–2 hillforts found on held-out sites |
| + centre prior on every small screenshot | 0.2683 | 0.2895 | **rejected**: gain comes from one site cluster (Kislovodsk) only |
| boxes → centre circles (shape) | | +0.0006 | shape is not the bottleneck |

## LiDAR (59 sites, 5 folds = previous team's L2 folds; their L2 ConvNeXt: 0.396)

| Experiment | Honest | Oracle | Decision |
|---|---:|---:|---|
| our U-Net ConvNeXt-T, 9 relief channels from LAS, kurgan-any heatmap + per-class masks, peaks for all classes | 0.4479 | 0.4582 | superseded |
| **+ hillforts & fortifications from mask components (representative point), kurgans from peaks, thr 0.1** | **0.5223** | 0.5223 | **deploy** — same setting chosen in all 5 folds |
| fortification-specific component threshold / size | fort F1 0.096 | 0.103 | no gain (model-limited) |

Per class (deployed): damaged 0.546, intact 0.559, hillforts 0.567 (was 0.288), fortifications 0.100.
Dense cemetery 036 Timerevo: touching mounds separated (see figures/detections/li_oof_036_Тимерево.png).

## Cross-modal fusion (koz3/fuse.py)

Training labels: 54–61 % of LiDAR objects are also labelled at the same spot in SpOr / Ae.

| Experiment | Honest | Decision |
|---|---:|---|
| 7 regions with LiDAR + satellite predictions: no fusion | 0.2876 | |
| LiDAR detections twinned into the optical layer (α = 0.3, no boost) | 0.3271 | superseded by the 16-region test |
| 16 regions (LiDAR + satellite and/or Ae): no fusion | 0.3480 | |
| twins in both directions, α ∈ {0.3, 0.6, 1.0} | 0.3468 | rejected (gain concentrated in one fold) |
| **LiDAR → satellite and LiDAR → Ae twins, α = 0.1** | **0.3654** | **deploy** — every fold chooses a LiDAR-twin variant |

## Ae (4 validation regions)

| Experiment | Honest | Decision |
|---|---:|---|
| heatmap U-Net trained on Ae (chips within 300 m of labels) | 0.026 | **rejected** (fires everywhere on grey film; labels verified aligned) |
| own D-FINE-L (batch 4, lr 2e-4, 768 px) | — | **failed**: classification head collapsed (max score < 0.04 even on training chips) |
| satellite D-FINE (SpOr-trained, held-out fold per site) on Ae sheets | 0.074 | |
| **… damaged kurgans only, conf ≥ 0.7, pooled with LiDAR + satellite experts** | 0.3085 → **0.3103** | **deploy** (same choice in 3/4 folds) |

## Or (1 validation region)

| Experiment | Honest | Decision |
|---|---:|---|
| heatmap U-Net, 20 km cluster folds | 0.000 | architecture labels sit in one site cluster (008/014/093/056 within 8 km) → no learnable signal when held out |
| heatmap U-Net, leave-one-region-out | 0.005 (arch 6 TP / 1166 FP) | rejected |
| **satellite D-FINE on the ortho at 0.5 m, architecture only, conf ≥ 0.5** | arch 11 TP / 71 FP (precision 0.13 > ~0.05 break-even) | **deploy** (weak evidence: 6 regions) |

## Deployment engineering (all verified)

* TorchScript for every U-Net, traced on the GPU (a CPU trace of the timm ConvNeXt bakes a CPU tensor) and checked
  against eager at a different size; fp16 storage → 1.36 GB archive.
* D-FINE / RT-DETRv2 cannot be TorchScript-ed usefully (> 10 min per call; non-profiling executor crashes) → eager
  HF transformers from vendored cp312 wheels (+ pandas: the Ultralytics image may lack it).
* Geophysics finds: scikit-image `peak_local_max` → scipy re-implementation (identical peaks on 6 random maps);
  LightGBM → numpy tree evaluator (|Δp| = 0); ported extraction reproduces cached features exactly.
* Relief channels: tiled computation for multi-km sites; integer-factor pooling (the old full-raster path drifted up to
  ¼ m on odd-sized sites); tiled == full on 99.95 % of pixels.
* LightGBM inference in torch on the GPU: 23× faster than numpy, identical output.
* Our environment reproduces the previous team's D-FINE OOF exactly (same counts ±3, same top scores and positions on
  6 held-out scenes) → their detectors deploy safely. Training D-FINE here does not work: our recipe (L) and their
  exact script with D-FINE-X both collapsed (max class score ≈ 0.2) → new detector training shelved.
* Platform simulation (`scripts/platform_sim`): packages the image may lack behave as not installed; vendored cp312
  wheels verified to import on Python 3.12; vendor prepended only on 3.12.
* **Rehearsal v2** (26 regions, every data type, platform simulation, 100 ms watcher): single atomic write PASS,
  413 s (validation: 51 regions ≈ 15–17 min estimated vs 26 min budget). Archive 1.364 GB.

## Data issues found
* 008 Or orthophoto is in EPSG:4326 (degrees); 014 / 093 Or are in UTM 36N although at 45° E (consistent, far zone).
* Raw Sudbishchi magnetometer files carry no coordinates (format example only).

## Leaderboard feedback and the deployment–validation mismatch (23 Oct)

* v2 (fold ensembles everywhere) scored **0.17023** on the public leaderboard (34th / 59). The validation simulator
  (`scripts/val_sim.py`: validation-shaped draws of training regions, honest OOF predictions, official pooled metric)
  expects **0.34** (5–95 %: 0.28–0.39) for the validated single-model recipe.
* Diagnostic run (fail on purpose to get logs): the platform returns only "container exited (code 3)" — no stdout/stderr.
* **Root cause found:** satellite / Ae detectors pooled 4 (2) fold models + NMS = max score. On imagery none of them saw,
  this emits **3.1–4.0×** the detections of a single model at every threshold, while the fusion threshold was tuned on
  single-model OOF scores → several-fold false positives in the kurgan classes. Score averaging instead under-shoots
  (0.52–0.66×). Neither matches the validated behaviour.
* **v4:** one model per expert (the fold model trained on the most data — it has never seen the validation sites, so it
  behaves exactly as in OOF), region-level satellite fusion and cross-image dedup (organiser rule). Rehearsal 290 s.
* Organiser rule (task page): one object per data type even across overlapping images; validation GT is merged per data
  type. EtGr objects under merged GT: no dedup 0.172 → cross-image dedup 0.248 (the per-map OOF 0.42 was optimistic).

## 23 Oct (cont.): v4 = 0.18455 → the ensemble calibration was real but not the main cause

* Validation simulator ablation (drop one expert's predictions): nothing dropped 0.342; −SpOr 0.286; **−LiDAR 0.200
  (0.15–0.26)**; −geophysics 0.250; −Ae 0.341. Only losing LiDAR matches the leaderboard (0.17–0.18, slightly below →
  misplaced/garbage LiDAR detections, not merely none). Exit-code probe `build/koz3_PROBE_lidar_exitcode.zip` built to
  test ground class / header CRS / UTM.json agreement / UTM-like coordinates / expert run on validation LAS.
* D-FINE training is broken in this environment (not transformers 4.57 vs 5.4, not AMP): their exact script and recipe,
  fold 0 → local 614 candidates, max score 0.13, F1 0.000 vs Kaggle 8 835 candidates, max 0.95, F1 0.285. U-Nets train fine.
* Zoom TTA (QASHI-style, ×1.5) for the 0.5 m-trained D-FINE: base 0.2585 vs zoom 0.2011 / avg 0.2309 / max 0.2272 → rejected.

## ROOT CAUSE of 0.170 / 0.185: laspy is not in the platform image → LiDAR produced nothing

* LiDAR exit-code probe returned exit code 1 = uncaught exception (its only un-guarded statement: `import laspy`).
* The organisers' baseline parses LAS records itself "so as not to pull laspy into the image"; laspy is used only
  for .laz "if installed". Our LiDAR expert raised ImportError in all 13 LAS regions (caught per region → 0 features),
  matching the simulator's "no LiDAR" expectation 0.20 (0.15–0.26).
* Fix (v5): laspy 2.7.0 vendored (pure Python; validation files are .las). The platform simulation now blocks laspy
  unless it comes from inference/vendor — under it v4 fails exactly as on the platform, v5 runs (rehearsal 285 s).
* Also in v5: ground-only clouds (any class code, e.g. training site 101 = all class 8) are no longer passed through
  PMF (which shaves mounds: 101 scored 0.449 with PMF vs 0.478 without).
* Package audit vs the baseline's top-level imports: image has cv2, numpy, rasterio, torch, PIL, affine, pyproj,
  shapely, ultralytics (→ scipy, torchvision). Everything else we use is vendored (pandas, transformers & deps, laspy).

## 23 Oct: v5 = **0.33021** (rank 25) — exactly the simulator's 0.34 (0.28–0.39) → the simulator is a reliable proxy

### Satellite detector training on this machine
* HF D-FINE training fails here whatever we change: fp32, TF32 off, 0 workers, transformers 4.57 (fold-0 F1 0.000 vs
  Kaggle 0.285). Single-step gradients match CPU (TF32 aside), optimizer matches, data/labels identical, head init sane.
* **torchvision FCOS (ResNet-50-FPN, COCO, BSD) trains fine here**: 1.5k iterations → fold-0 F1 0.235 (D-FINE 8k: 0.285).
  Full 4-fold training queued; it joins the nested fusion study as a fourth member.
* LiDAR: second seed per fold training (honest per-fold ensembles: thresholds tuned on the deployed ensemble).

### Intact / damaged (dual-task idea) — 23 Oct
* Morphometric LightGBM on relief features around each LiDAR kurgan peak (radial relief profiles, slope asymmetry,
  summit dent, ring ditch, model mask probabilities), OOF by LiDAR folds: damaged-vs-intact AUC 0.786 (mask rule) → 0.805.
* The larger lever is **dual emission**: LiDAR nested 0.5223 (dual 0.3) → **0.5319** (dual 0.05, mask rule) → 0.5349 with
  the classifier. Deployed: dual 0.05 with the mask rule (classifier +0.003 deferred).
* Satellite dual emission **hurts** (0.2729 → 0.2310): its kurgan precision is too low for a second guess to pay off.
* Papers checked (MDPI 18(20) 3520 Mamba segmentation, 18(19) 3392 few-shot scene classification, 18(18) 3116 review,
  18(17) 2992 optical+nDSM fusion): urban segmentation / scene labels — no direct transfer. NDVI / NDWI / SWIR impossible
  (SpOr is RGB + alpha only). TerraMind: Sentinel-2 10 m multispectral pretraining ≠ 0.1–3 m RGB; licence unverified.

### 23 Oct evening
* **LiDAR honest per-fold ensemble (2 seeds, same split)**, dual 0.05: single 0.5342 → **ensemble 0.5489**
  (damaged 0.569, intact 0.626, hillforts 0.571). Deployed as the fold-1 seed pair, thr 0.15.
* **FCOS (torchvision) full 4 folds: alone 0.188**; added to the fusion it **lowers** it (0.2729 → 0.2611) → rejected.
* Validation simulator: v5 0.342 → **v6 0.354** (0.279–0.407). v6 = build/koz3_submission_v6_li_ensemble_20261023.zip,
  rehearsal under the platform simulation 337 s, single atomic write.
* HMM / particle filters: no temporal axis in the data; only niche = trench tracing for fortifications (weight 5).

## 23 Oct (evening): hillfort scene gate, D-FINE root cause, Ae diagnosis

### Satellite hillfort gate (koz3/scene_cls.py, scripts/sp_scene_gate_oof.py)
* Whole-scene ConvNeXt-T classifier "small screenshot contains a hillfort", OOF on the shared satellite folds: **AUC 0.892**
  (152 small scenes, 49 positive).
* Nested over gate × mode × confidence on top of the deployed fusion (equal weights, t = 0.25):
  fusion 0.2729 → centre prior on every small scene 0.2885 → **gate "swap" (centre point, conf = p, when p ≥ g; drop the
  detectors' hillforts when a small scene has p < g) 0.3012–0.3023**, g = 0.05–0.1 chosen in every fold.
  Placing the hillfort at the detectors' best candidate instead of the centre: 0.275 → rejected.
* Validation simulator (LiDAR seed ensemble, thr 0.15 / dual 0.05): v6 0.354 → **v7 0.368 (0.291–0.427)**; hillfort F1 0.28 → 0.37.
* v7 = v6 package byte-for-byte + models/sp_scene (fold-2 classifier, TorchScript) + sp_or gate (gate_thr 0.1).
  Parity: raw-TIFF platform path reproduces OOF p within 0.03, centres identical.

### Why HF D-FINE did not train here — ROOT CAUSE
* Same loss curve as Kaggle, but eval-mode max score 0.15. The local model scores 0.91 only in *train mode*, where D-FINE
  adds denoising queries built from the ground truth; Kaggle's weights evaluated with the local code show a train-mode
  VFL loss of 67. The local stack trains the classifier on the GT-derived denoising queries, not on the real queries.
* Fix: `DET_NUM_DN=0` (no denoising queries). Fold 0, 3k iterations: **F1 0.221** (was 0.000; Kaggle 8k with DN 0.285).
  cu128 vs cu130 made no difference (cu128 venv deleted).

### Per-data-type OOF (simulator predictions, all regions) — where the points are
* SpOr 0.304; Li 0.485; **Ae 0.042** (1083 damaged / 587 intact kurgans / 393 fortifications missed; 4 validation regions);
  MagGr 0.497; EtGr 0.362; TerFl 0.000; **Or 0.000** (178 architecture FN → architecture F1 0.09 overall).
* Next: Ae-trained detector (koz3/ae_det_train.py, FCOS; then D-FINE no-DN / ultralytics — organisers allow AGPL ultralytics).

### Own-data detectors for Ae and Or (koz3/ae_det_train.py, FCOS R50-FPN; scripts/ae_det_oof.py, nested per-class thr)
| Data | Folds | Result (honest) | Decision |
|---|---|---|---|
| Ae, all classes | Ae site folds (4) | kurgans F1 0.08 (4 035 FP: hundreds of m from any label — no registration offset; dense cemeteries recall-limited) | kurgans **rejected** |
| Ae fortifications | same | **F1 0.254** (thr 0.4 in every fold; deployed expert 0.000) | **deploy** |
| Or architecture (stone ruins, 0.1 m) | leave-one-region-out (6) | **F1 0.216** (thr 0.4–0.5; previous 0.000 OOF) | **deploy** (thr 0.45) |
* "fast" inference (one pass per 1024 px tile, no flip) validated separately: equal or better (fort 0.229 → 0.254, arch 0.201 → 0.216), ~8× faster.
* Ae sheet-centre prior is weak (labels at median 0.43 of the sheet half-size vs predictions 0.61) → not used.
* Simulator: v7 0.368 → **v8 0.381 (0.307–0.442)**; architecture F1 0.094 → 0.172, fortifications 0.028 → 0.144.
* v8 = v7 + models/ae_det (Ae fold-2 FCOS) + models/or_det (Or fold-5 FCOS), Or old path off, per-region caps (Ae 180 s,
  Or 420 s); expert order now geo, las, **sp_or**, ae, or_ (validation Or sheet is 2.7 GB; satellite must not be starved).
  Rehearsal: PASS, Ae 63 MB sheet 23 s.
* Next: satellite D-FINE without denoising (seed 1, 4 folds × 8k) as a 4th fusion member — kurgans carry 50 % of the weight.

## 24 Oct: v9 = v8 + local satellite D-FINE (no denoising) as a 4th fusion member
* D-FINE-L, DET_NUM_DN=0, seed 1, 4 folds × 8k, folds identical to the Kaggle ones (175/175 scenes). Alone 0.2095 honest.
* Nested fusion (all with the hillfort gate): D+R+H 0.3023 → **D+R+H+N 0.3068** (t 0.25) / D+R+N 0.3081; set nested 0.3055.
  Simulator: v8 0.381 → **v9 0.385 (0.306–0.448)** — consistent but small.
* Deployed as its own member (models/sp_dfine2, fold 2) — pooling it with the Kaggle D-FINE would miscalibrate.
  Parity check caught a bug: the exported config needs num_denoising = 0 (no denoising embedding) — now automatic in
  build_submission.export_det. Deployed detections match OOF (counts / max score within fp16 noise).
* Ae/Or detector now reads the sheet once and slices tiles in memory (sp_io.read_grid_raw; pixel parity 0.1/255).
* Rehearsal PASS. Timings were inflated by another agent's GPU job (Codex Work/ae_yolo_pilot, 97 % GPU); the per-region
  caps (Ae 180 s, Or 420 s) and the global budget guard bound the worst case.

## 24 Oct: hunting a bigger kurgan gain
### Satellite context rescoring (scripts/sp_rescore_oof.py) — rejected
* Second-stage LightGBM over fused candidates (member scores, scene geometry, neighbourhood support, region rank).
  With labels "inside a same-class polygon" AUC rose for every class but F1 fell (duplicates near true kurgans promoted).
  With labels "best candidate per GT object": kurgan AUC 0.854 (fused) vs 0.844 (GBM) → no gain; per-class nested
  0.3099 (fused score) vs 0.2573 (GBM) / 0.2891 (blend). Only architecture benefits (0.19 → 0.25). Satellite kurgans
  are limited by the detectors, not by post-processing.
### LiDAR kurgan misses (scripts/li_fn_analysis.py, figures/li_kurgan_misses.jpg, figures/li_sites_relief_heat.jpg)
* 2 147 kurgan labels, recall 0.595. Of 870 misses, **598 get heat < 0.05 (model blind)**, 18 have strong heat lost by
  peak extraction, 110 are found with the wrong class.
* Failure is per SITE: Novenkoe recall 0.14 / Erentuna 0.14 / Timerevo 0.41 vs Gnezdovo 0.70–0.84 — at Novenkoe
  the mounds are obvious in the relief yet the heat is ~0. Causes: local-relief channels clipped at ±2 m (tall mounds
  saturate: channel range 7.7 vs 3.1 at Gnezdovo), weak augmentation (×0.8–1.2 amplitude, no size change), Gnezdovo
  dominates the object sampling.
* Fix under test (LI_AUG=1, koz3/li_train.py): relief recomputed per crop from the DTM with the ground stretched by
  zoom ×0.7–1.45 (mounds larger and physically flatter) and heights ×0.5–2.2; site-balanced object sampling. Parity at
  z = s = 1: channel |diff| 0.02–0.13 (window edges). Baseline single model (same nested decoding): **0.5327**.

### Technical Regulations (approved, 2026-10-24 check) — no change to our pipeline
* 30-min run limit, platform GPU RTX 4090 or better; validation LiDAR = LAS only (we build the DTM / relief ourselves).
* Metric identical to our scorer (cov ≥ 0.5 on the smaller shape + representative point inside GT; points ≤ 2 m;
  FP ×0.5; per-class greedy by confidence; weights 30/20/15/10/5 + 10/10). Areas / distances in UTM.
* LiDAR labels were drawn on hillshades made from LAS by the published baseline algorithm.

### LiDAR morphometric rescoring (scripts/li_rescore_oof.py) — rejected
* GBM on physical peak features (relief rings, asymmetry, dent, ditch, slope, curvature, neighbours), nested:
  AUC 0.917 (peak heat) → 0.926, but metric 0.548–0.551 vs deployed decoding **0.5543** (seed ensemble).

### Calibration
* Confidence only orders the greedy matching; classes are pooled F1 at the emitted set → cross-modal confidence
  calibration barely matters. Joint emission thresholds in the simulator (SP_T × LI_THR): 0.373–0.388 → ±0.005, no lever.

### Queued (honest, nested comparisons)
1. LiDAR LI_AUG (zoom ×0.7–1.45, height ×0.5–2.2, site balance) — vs 0.5327 single.
2. LiDAR LI_AUG + LI_PASTE (copy-paste of real kurgan relief onto other terrain, 10 m trend) + RELIEF_EXTRA
   (positive / negative openness + sky-view factor, 8 directions, 10 m) — figures/li_openness_novenkoe.jpg,
   figures/li_paste_check.jpg; OOF channels computed from the DTM exactly as at inference (li_train.oof_dtm).
3. Satellite D-FINE no-DN + DET_SITE_BAL (site first) + DET_QAUG (resolution loss, JPEG, noise, haze, sharpening —
   low- and high-quality sources) — figures/sp_qaug_samples.jpg; vs the plain no-DN model (0.2095 alone).

### LiDAR augmentation result (LI_AUG, 5 folds × 12k)
* Single model, same nested decoding: **0.5327 → 0.5490** (intact kurgans 0.579 → 0.641, hillforts 0.558 → 0.595,
  fortifications 0.10 → 0.14). Site recall: Esbyu 0.35 → 0.63, Gnezdovo-2 0.70 → 0.76, but Novenkoe 0.14 → 0.12,
  Timerevo 0.41 → 0.30, Gnezdovo-Zaolsh. 0.35 → 0.09.
* Generalisation gap proven: the same architecture finds 98–100 % of Novenkoe's and 93 % of Erentuna's kurgans when
  those sites are IN training, 14–17 % when held out → data and labels are fine; these sites' mounds resemble nothing
  else in training.

### Physics-informed mound detector (scripts/li_blob_oof.py) — rejected
* Candidates = local maxima of the 10 m local-relief model (≥ 7.5 cm), 146 677 candidates covering 72 % of the kurgan
  labels (CNN recall 60 %); physical morphometry → LightGBM, nested. Candidate AUC 0.957 vs CNN heat 0.891, but
  per-site the probability scale shifts (site-relative "test-time adaptation" features did not fix it).
* Adding physics finds where the CNN has none: 0.5538 vs 0.5543 (nested picks the strictest threshold).
  Replacing the CNN kurgans: 0.47–0.48. The high candidate AUC came from easy flat-ground negatives.
* LiDAR ensembles, same nested decoding: deployed s0+s1 0.5489 → s0+s1+aug (equal) **0.5573** / (s0+s1)/2 + aug 1:1
  **0.5581** (intact 0.626 → 0.66, fort 0.05 → 0.10–0.12). Held for the next build together with the queued runs.

### LiDAR test-time self-training (scripts/li_ttt_oof.py) — rejected
* Per held-out site: fold model (aug) predicts → confident kurgan peaks (≥ 0.5) and hillfort / fortification mask
  components (≥ 0.6) as pseudo-labels, uncertain areas zero-weighted → fine-tune on that site alone → predict again.
  Fixed config, no labels. Pilot (8 sites, 300 it): pooled kurgan F1 0.575 → 0.504; variant with every ≥ 15 cm bump
  marked uncertain (150 it, 6 sites): 0.564 → 0.498. The missed kurgans get heat < 0.05, so they cannot become pseudo-
  positives; adapting to a handful of confident examples narrows the model (recall falls at every site).

## 24 Oct: v9 = **0.41693** on the platform (simulator 0.385 — it under-predicts now). Platform per-type P / R:
LiDAR 0.61 / 0.56, satellite 0.38 / 0.54, **Ae 0.045 / 0.093**, Or 0.71 / 0.24, geophysics 0.68 / 0.72; run 13 min / 30.
* Ae: OOF precision 0.13–0.19 (kp ≥ 0.7 / FCOS fort), platform 0.045 — validation Ae regions are Ae-only (no LiDAR
  twins), so the Ae detectors fire on unlabelled features of the big sheets. Ae recall is only 9 % → switched off.
* Or: kept at 0.45 (OOF F1 0.226 at 0.45, 0.190 at 0.30 — the validation ortho's precision is not tuned to).
* **v10** = v9 + LiDAR 3-model ensemble (s0, s1, aug fold 1; honest 0.5489 → 0.5573) + Ae off (config experts_off);
  rehearsal PASS (li: 3 models, ae disabled).

### Foundation backbones, quick honest test (scripts/foundation_probe.py)
* Frozen backbone + linear probe, same 152 satellite small scenes and 4 site folds as the deployed gate (fine-tuned
  ConvNeXt-T OOF AUC 0.892). Hillfort / kurgan scene AUC: ConvNeXt-T IN-22k 0.751 / 0.798; DINOv2 ViT-S 0.859 / 0.735;
  **DINOv2 ViT-B 0.881–0.898** (C tie-break) / 0.767. Self-supervised features carry far better hillfort signal than
  ImageNet features (frozen); no extra kurgan signal at scene level.
* Gate = mean(fine-tuned ConvNeXt, DINOv2-B probe): AUC 0.892 → 0.916, but nested satellite score 0.3012 → 0.3037 only
  (+1 hillfort): localisation (scene centre inside the hillfort in ~40 % of scenes) is the bottleneck, not recognition.
  Not deployed (350 MB ViT for +0.0025).

### LiDAR copy-paste + openness/SVF (LI_AUG + LI_PASTE + RELIEF_EXTRA, tag _augx) — validated
* Single model, same nested decoding: **0.5688** (aug 0.5490, s0 0.5327) — hillforts 0.633, intact 0.645.
* Ensembles: s0+s1+aug (v10) 0.5573; + augx (4 equal) 0.5571; aug+augx 0.5758; **(s0+s1)/2 + aug + augx in thirds
  0.5765** (oracle 0.5775 @ thr 0.15 → stable) — deployed.
* Deployment: li_x_*.ts take the 12-channel relief computed from the DTM in las.py (relief EXTRA toggled per call);
  per-model weights in config li.weights. Parity on Timerevo (held-out fold 1, raw LAS → platform path vs OOF blend):
  149 vs 149 features, damaged kurgans 131 vs 129 TP.
* Simulator (Ae off): LiDAR 2-seed 0.3781 → augx single 0.3873.
* **v11** = v10 + 4-model weighted LiDAR blend (thr 0.15). Rehearsals PASS (satellite / LAS set; geophysics set 0.4678
  unchanged). Expected platform ≈ 0.44–0.45 (v9 0.417; Ae off ≈ +0.015–0.02; LiDAR ≈ +0.01).
* OOF pickles are now zstd-compressed (work/li/oof_*.pkl.zst; `zstd -d` before use).

### Satellite D-FINE no-DN + site balance + image-quality augmentation (sp_dfine_l_balq, seed 2) — small gain
* Same folds (175/175). Alone 0.2211 vs plain no-DN 0.2095. Fusion (+ hillfort gate): 4 members 0.3068 →
  **5 members (+balq) 0.3143** (t 0.25 in 3/4 folds); replacing dfnodn 0.3137; member set nested 0.2985 (selection noise)
  → candidate for v12 with the set fixed in advance (all five).

## 24 Oct: v11 = **0.44032** (LiDAR P/R 0.60/0.64, satellite 0.38/0.60, Ae 0/0 (off), Or 0.71/0.24, geo 0.68/0.72; 10.5 min)
### Towards v12 (all nested / honest)
* LiDAR veto of satellite kurgans (9 LiDAR+SpOr regions): 0.3683 → 0.3700 — too small, not deployed.
* Satellite per-class fused thresholds (5-member fusion + gate): architecture 0.15 in 3–4/4 folds → arch F1 0.189 →
  0.28–0.32; other classes keep the global 0.25. Simulator: v11-like 0.3873 → +balq 0.3888 → **+arch 0.15 0.3943**.
* Geophysics objects per-method thresholds (deployed 50/50 CNN+GBM blend, 5 region-group folds): object_poly
  **0.3706 → 0.4158**; MagGr 0.75 (all folds, interior after extending the grid), EtGr 0.3, TerFl 0.6 (4/5).
* LiDAR hedge (dual) on augx single: 0.05 → 0.15 gives 0.5688 → 0.5714 (re-check on the final blend).
* sp_dfine3 (balq fold 2) parity: counts / max scores equal OOF within fp16 noise.
* Thermography has only 2 training regions → per-method TerFl threshold is fragile (rehearsal: 0 polygons at 0.6).
  **MagGr-only re-threshold (EtGr / TerFl 0.3): object_poly 0.3706 → 0.4197** (0.75 in 4/5 folds, oracle 0.4205) — deployed.
* Satellite kurgan scene gate (frozen ConvNeXt-T IN-22k probe, kurgan-scene AUC 0.798; DINOv2 0.767; both 0.739):
  every gate threshold lowers the score (v12 satellite pipeline 0.3267 → nested 0.3074) — rejected.
* Geophysics finds: deployed p_base @ 0.25 already best (F1 0.736 pooled; ensemble 0.736, aug 0.70, no-dipole 0.725).
* Hillfort localisation with DINOv2-B patch tokens (linear probe, other folds): peak inside the hillfort 0.47 vs scene
  centre 0.39; pooled 0.3267 → 0.3328 (peak) / 0.3352 (both points) but nested only 0.3272 (mode flips between folds)
  → not deployed (350 MB ViT for an unstable gain).
* Drawn annotations in satellite screenshots: 2 of 517 chips → no usable cue.
* LiDAR LI_FORT per-class component threshold: 0.130 → 0.134; r_out for thin trenches: no effect (trenches ~5 m wide).
* v12 satellite pipeline honest (5 members, arch 0.15, hillfort gate): **0.3267** (v11 0.3068).
* LiDAR d8 TTA (augx fold models, channels from the DTM, nested): 3 flips 0.5651 → **d8 on full 1024 px tiles 0.5695**
  (kp 0.553 → 0.563); d8 with small sites padded to squares 0.5651 (no gain) → deployed: d8 on full tiles only (small,
  near-noise gain; costs only inference time).
* Codex's analysis (shared by the user): agree on LiDAR blind sites → **unclipped relief** (RELIEF_UNCLIP: asinh instead
  of the ±2 m clip on the 4 local-relief channels) training as _augu (LI_AUG + LI_PASTE + RELIEF_EXTRA), concurrent with
  the augx seed; deploy path ready (li_u_*.ts → unclipped 12-channel relief; refactored las.py reproduces v11: 149 feats).
  Ae: own-data detector already failed for kurgans (F1 0.08) and Ae ran at 4.5 % public precision → only a pilot beating
  ~20 % held-out precision would bring it back. Kaggle with competition data: not without organiser approval.

### Post-processing round (all nested; no gains → post-processing is saturated)
* Satellite (v12 pipeline 0.3267): WBF-style score-weighted member positions 0.3271; per-member rank calibration
  0.2940; logistic stacking of the 5 member scores 0.2821.
* LiDAR peaks (augx OOF 0.5690): NMS window 7–19 (11 best); sub-pixel heat-centroid refinement pooled +0.001, nested 0.5690.
* Geophysics objects (MagGr 0.75 blend 0.4205 pooled): Gaussian denoising σ 1–2 px + min area 0.25–2 m² → nested 0.4136.
* Swin Transformer, frozen linear probe (same 152 scenes / 4 site folds): Swin-T IN-22k 0.683 hillfort / 0.762 kurgans;
  Swin-B IN-22k 384 0.846 / 0.761 — below DINOv2-B (hillfort 0.88–0.90) and ConvNeXt-T IN-22k (kurgans 0.798).
* Swin-T U-Net encoder for LiDAR works at a fixed 512 px (img_size, strict_img_size=False; inference tiles 512);
  single-fold pilot (fold 2, contains Novenkoe; augx recipe) queued after the unclipped run — GPU memory is full until then.
* Second seed of the copy-paste/openness LiDAR model (augx_s1) — no gain, large seed variance: v11 blend 0.5738 (dual
  grid incl. 0.15; 0.5765 with dual 0.05) → + augx_s1 0.5678 / 0.5681; augx + augx_s1 alone 0.5450 (hillforts 0.505).
  v12 keeps the v11 LiDAR blend (dual 0.05).

### Weakness analysis (v12 config, simulator per class; weighted loss = w·(1−F1))
* kp 18.0, kt 10.9, gor 8.9, arch 7.4, obj ~5.8 (deployed blend), fort 4.6, finds 3.4. Damaged kurgans dominate.
  Validation-shaped kp labels ≈ satellite 250 / LiDAR 300 / **Ae 145 (≈21 %, all missed with Ae off)**.
* Ae cemetery-cluster prior (keep a kurgan only with ≥ k strong neighbours within r): SpOr-D-FINE-on-Ae precision
  0.16 → 0.26 at recall ~8 %; nested F1 0.099 (FCOS 0.054). OOF→public precision ratio for Ae was ~3.5× worse →
  Ae stays off; it needs a fundamentally better detector.
* Satellite kurgan precision falls with distance from the scene centre in BIG scenes (framed on the site): 0.38 / 0.32 /
  0.20 / 0.12 / 0.05 for d_rel 0–0.25 / –0.5 / –0.75 / –1 / >1; neighbour density barely matters (0.16–0.27).
* **Framing prior** (big scenes: d_rel > x needs score ≥ t2, all classes): coarse grid nested 0.3267 → **0.3423**,
  (0.5, 0.35) in all 4 folds; fine grid plateau 0.339–0.343 for x 0.35–0.5, t2 0.3–0.4 (nested 0.3339 from picking
  among near-equal options) → deployed x 0.5, t2 0.35. Deployment geometry from the image grid matches the study exactly
  (8/8 scenes). Simulator 0.3943 → **0.3991**.

### Ae rebuild (24 Oct evening)
* Ae framing prior (sheet centre distance + far-threshold, optional soft cluster condition), nested: SpOr-D-FINE-on-Ae
  0.120 → 0.108, FCOS 0.065 → 0.069 — Ae's core problem is recall (14 % at 4 % precision even at t 0.3), not location.
* Joint detector training (DET_ARCH=dfine no-DN, DET_JOINT: Ae tiles + grey satellite tiles farther than 25 km from every
  held-out Ae site, site-balanced positives, DET_QAUG resolution loss / JPEG + film photometry), 4 Ae folds × 5k → ae_dfj.
* Stopped the satellite seed-3 run (6th correlated member, ~+0.003 expected) to give the GPU to Ae.
* Gemini suggestions: cross-modal Ae confirmation not applicable (validation Ae regions are Ae-only); rank-average
  satellite fusion already rejected (0.294 vs 0.327); unclipped LiDAR only enters the blend if nested blend+unclipped beats
  the proven blend.
* **Joint satellite+Ae D-FINE (ae_dfj) — Ae still does not validate**: per-class nested 0.0568 (deployed path 0.0419);
  damaged kurgans F1 0.112 (179 TP / 3 728 FP / 983 FN, P ≈ 0.05), framing prior nested 0.106. Kurgans on 1940s film
  remain largely undetectable at unseen sites → Ae stays off in v12.

### Damaged / intact confusion (Codex's Ae observation, checked on satellite)
* v12 satellite kurgan predictions @0.25: damaged 416 same-class / 55 inside an INTACT label / 1 077 on nothing;
  intact 44 same / 42 inside a DAMAGED label / 162 nothing.
* **Disagreement hedge** (emit the other kurgan class too when the members' best score for it ≥ tau), nested on the v12
  pipeline incl. framing prior: 0.3423 → **0.3620**, tau 0.4 in all 4 folds (interior: 0.5 → 0.3586); kp TP 384 → 418,
  kt TP 50 → 77. Deployed: koz3.sp_fuse returns s_kp / s_kt; sp_or emits the hedge with confidence ×0.99; each fused
  object (and its hedge) carries a shared "image" tag so region dedupe keeps the pair (scenes are already fused per region).
* **Unclipped relief (augu) — no gain**: single 0.5611 (augx 0.5688; best fortifications 0.157); v11 blend + augu 0.5729,
  (s0+s1)/2 + aug + augu 0.5669 vs v11 blend 0.5738–0.5765; blind sites unchanged (Novenkoe 0.19, Erentuna 0.28).
  Saturation was not the cause of site blindness. v12 keeps the v11 LiDAR blend.

### CPU geomorphometric pipeline (Gemini's CPU-only blueprint; scripts/li_cpu_geo_oof.py, 12 parallel workers) — rejected
* Candidates: multi-scale LoG on the positive 10 m local-relief model → 195 219 candidates covering 1 690 / 2 147 kurgan
  labels (79 %; CNN recall ~60 %). Features: relief rings / slope / curvature, LoG scale + response, rotation-invariant
  angular-harmonic magnitudes (Zernike-style; skimage has no moments_zernike), Gaussian profile-fit R² (N–S, E–W), radial
  gradient symmetry, half-height compactness / eccentricity / solidity, neighbour counts. Tier 1 LightGBM (other folds),
  tier 2 damaged / intact from the CNN masks with a 0.15 hedge.
* Tier-1 AUC: geometry 0.946 vs CNN heat 0.965 on the same candidates (geometry + CNN 0.965).
* Nested metric (augx OOF, deployed 0.5690): ADD 0.5691 / 0.5674; REPLACE 0.4234 (geometry) / 0.5122 (+CNN).
* Hand-crafted geometry does not separate the CNN-missed kurgans from natural bumps either.

### "CPU paradigm shift" round (Gemini) — persistent homology, component trees, wavelet scattering, DINOv2 + k-NN — rejected
* Persistent homology (GUDHI cubical complexes on 64×64 local-relief patches: H0 persistence of the candidate component
  = mound prominence, top-3 H0, H1 count / total / max = ring ditches; component-tree "dynamics" is the same H0
  persistence) + wavelet scattering (Kymatio J 3, L 8, 217 coefficients) on the LoG candidates (top 3 000 / site,
  123 316 candidates, 1 641 kurgan labels covered): tier-1 AUC TDA 0.925, scattering 0.944, TDA + scattering 0.964,
  + geometry 0.966, + CNN 0.971 vs CNN heat 0.961 — but nested metric ADD 0.5691 / 0.5686 vs 0.5690, REPLACE 0.4686 /
  0.5176. AUC is dominated by easy flat-ground negatives; at the decision boundary nothing beyond the CNN.
* DINOv2-B patch tokens (cached, satellite): patch-level kurgan AUC linear probe 0.819, faiss k-NN (cosine, k 20) 0.750,
  sparse random projection 768→96 + k-NN 0.740; scene level 0.70–0.74 (ConvNeXt scene probe 0.798). Patches (14–37 m)
  are larger than kurgans — no localisation gain.

### Metric-engineering round (Gemini) — nothing robust
* LiDAR per-class kurgan thresholds (nested): 0.5654 → 0.5645 (kt 0.628 → 0.640, kp 0.562 → 0.552) — no gain.
* Regional cross-modal context ("MIL"): lower LiDAR's threshold where the satellite fusion has ≥ n kurgans in the region
  (10 LiDAR + satellite regions). Single augx model: 0.4299 → 0.4583, (3, 0.1) in all folds; **deployed blend**: pooled
  best +0.014 (0.4845 → 0.4982) but nested 0.4634 < 0.4845 (choices disagree) → not deployed.
* Not run: KDE of training-label locations (cannot be validated with geographically separated folds; a bet on validation
  sites lying near training sites); centroid-displacement regression for large objects (needs a new trained model —
  the DINOv2 hillfort localiser was unstable).

## v12 final (24 Oct 23:05) — build/koz3_submission_v12_final_20261024.zip (873 MB)
* Sixth satellite member (balq seed 3) in the full v12 satellite pipeline (arch 0.15, framing prior, hedge 0.4, gate):
  5 members 0.3620 → **6 members 0.3667**, hedge tau 0.4 in all folds; parity of sp_dfine4 OK.
* v12 = v11 + satellite: 6 members (sp_dfine2 / 3 / 4 separate), architecture 0.15, framing prior (x 0.5, t2 0.35),
  damaged / intact hedge 0.4; geophysics: MagGr objects 0.75; LiDAR: d8 TTA on full tiles. LiDAR blend, Ae off, Or,
  finds, EtGr / TerFl unchanged.
* Satellite honest: v11 0.3068 → v12 **0.3667**. Simulator (no objects change modelled): 0.387 → ~0.40.
* Rehearsals PASS: satellite / LiDAR set 0.1265 → 0.1580 (4 LiDAR models, 6 satellite members, ae disabled);
  geophysics set 0.4678 → 0.4827 (object_poly 0.294 → 0.331). Single atomic write; zip integrity OK.
* Expected public ≈ 0.46–0.47 (v11 0.44032).

## 25 Oct: v12 = **0.48874** (satellite P/R 0.381/0.599 → 0.442/0.642; geophysics P 0.680 → 0.760; LiDAR 0.60/0.64; 11.5 min)
### Deepseek "interior anchor patches" — already our representation (1 m circles: BIoU ~0.001) except geophysics objects
* Geophysics objects as small patches (≤ 1 m) at an interior anchor vs full polygons (nested): polygons 0.4197,
  representative point 0.4231, **pole of inaccessibility 0.4306** (MagGr 0.6 in 4/5 folds) → staged for v13 (build/sub_v13).
* LiDAR hillfort / fortification anchors (augx OOF): surface point vs pole of inaccessibility vs probability peak —
  identical (gor 19/22/11, fort 23/133/241) → losses are detection misses, not placement.
* Large satellite objects (5 000–10 000 m² public recall 0): mostly satellite hillforts (27) and large ploughed-out
  damaged kurgans (27, faint soil marks 80–110 m; only 20 / 47 have a fused detection inside) — figures/sp_large_kurgan_labels.jpg.
* **Coarse satellite branch** (D-FINE no-DN, site-balanced, quality aug, scenes ×4 → 2 m/px, targets ≥ 40 m only, folds
  identical) added beside the v12 pipeline where it has no same-family detection within d: t_c 0.3 → kp TP 414 → 456 but
  +321 FP (0.348–0.351); t_c 0.6 → 0.3677 pooled; nested 0.3646 < 0.3667 → rejected (similar soil patterns abound).
* **v13** = v12 + geophysics pole-of-inaccessibility object patches (MagGr 0.6): geophysics rehearsal 0.4827 → 0.4913;
  expected public gain ≈ +0.001–0.003 (objects weight 10). build/koz3_submission_v13_geoanchor_20261025.zip.
* Big-scene hillfort gate: not needed — the small-screenshot gate already covers the regions of 63 / 65 satellite hillfort labels.
* Scene-type-specific satellite thresholds (small vs big, v12 pipeline): deployed 0.25 / 0.25 optimal (nested 0.3641 vs 0.3667).

### "Paradigm-shift plan" (user-shared) — assessment and Exp 3 run
* Claims checked against our record: SHAD ≈ tonight's joint SpOr-grey + Ae detector with film photometry / resolution
  loss / JPEG / noise (held-out Ae F1 0.112, P ≈ 0.05); site self-training already rejected (0.575 → ~0.50); DT head
  targets placement, but LiDAR anchors are not the bottleneck (identical hits for 3 anchors). FiLM / cross-attention /
  Siamese / e2cnn need multi-hour GPU training (queue by evidence).
* Exp 3 neighbourhood prior (confidence × (0.5 + 0.5·clip(n/K, 0, 2)), plan constants, threshold applied after reweighting):
  satellite v12 pipeline 0.3667 → 0.31–0.34 (nested keeps no prior) — rejected; LiDAR blend, filter on output confidence:
  0.5648 → nested 0.5812 (prior in all folds) — re-testing inside the exact deployed decoder (peak-height reweighting).
* **LiDAR neighbourhood prior inside the exact deployed decoder** (peak height × (0.5 + 0.5·clip(n/K, 0, 2)), n = kurgan
  peaks ≥ snb within R, before the deployed threshold): deployed nested 0.5765 → **0.5831**; prior chosen in all 5 folds,
  snb 0.2 in all; every snb-0.2 setting at thr 0.15 beats the baseline pooled (0.5874–0.5884 vs 0.5775). Deployed
  (R 200, K 3, snb 0.2, thr 0.15) via li_train.neighbour_prior / decode_site_v2(prior=…); parity with the study exact
  (3 sites, identical confidences). Staged in v13.

### Deepseek round 2 (CPU) — mostly closed
* A (class-specific NMS) → our equivalent is the satellite fusion cluster radius: kurgans 8 m optimal (4–16 m tested);
  other classes 16 m vs 8 m +0.003 pooled at every kurgan radius (16 m in all folds) — small candidate.
* B (mask-IoU NMS) not applicable (point outputs). C (Ae diagnostic) already done earlier (predictions hundreds of m from
  labels; no offset; dense-site recall limited).
* E systematic offset: satellite hits vs label centroids mean (+0.04, −0.16) m → no bias, shift-TTA pointless.
* H pooled-kurgan diagnostic (satellite, no hedge): damaged 0.413 / intact 0.299 / any-kurgan 0.476 → class confusion
  costs real points → damaged/intact crop classifier (ConvNeXt-T, 48 m crops, site folds) queued (GPU full until ~03:00).
* Satellite damaged / intact crop classifier (ConvNeXt-T, 48 m crops, 1 294 labelled kurgans, site folds): OOF AUC 0.558
  (near chance) → class assignment / hedge from it 0.30–0.33 vs member vote + hedge 0.3667 (kept in all folds) — rejected.
  Damaged vs intact is not learnable from 0.5 m appearance across sites.
* v13 (v12 + geophysics pole anchors + LiDAR neighbourhood prior) clean rehearsal: PASS (single atomic write).
* Satellite heatmap retrain (site balance + quality aug, tag _bq): OOF step crashed (tag not passed to sweep) — fixed, rerunning.
* **Satellite heatmap retrain (site balance + quality aug, _bq)**: alone 0.2017 honest (old heatmap ≈ 0.13); replacing
  the old heatmap member in the 6-member v12 pipeline 0.3667 → **0.3696** (hedge 0.4 all folds); as a 7th member 0.3681.
  Deployed as the sp_or heatmap (fold 2, TorchScript); candidates at thr 0.05 / dual 0.3 as in the study; deployed vs OOF
  candidate counts close (cross-scene dedupe / JPEG chips explain small differences).
* **v13** = v12 + geophysics pole anchors (MagGr 0.6) + LiDAR neighbourhood prior (R 200, K 3, snb 0.2) + retrained
  satellite heatmap member. Rehearsals PASS (satellite/LiDAR set 0.1552 vs 0.1580; geophysics 0.4827 → 0.4913).
  Expected public ≈ +0.005–0.01 over v12. build/koz3_submission_v13_20261025.zip.

### "Grandmaster assessment" plan (user-shared) — run / checked
* Premise checks: LiDAR is not 65–70 % of the metric (classes pool over LiDAR / satellite / Ae; 30 satellite vs 13 LiDAR
  validation regions); kurgans come from a focal heatmap, only class / hillfort / fortification masks use BCE+Dice;
  no landscape metadata exists (UTM.json = zone + CRS only) → FiLM would need a learned terrain descriptor.
* Exp 1 running: LiDAR with metric-weighted soft-F1 mask loss (BCE + 1 − soft-F1 with 0.5·FP and class weights
  30/20/15/5; LI_LOSS=softf1, augx recipe), 5 folds, tag _sf1.
* ConvNeXt-Small LiDAR (augx recipe, tag _cnxs) finished; scoring alone and in the blend.
* **ConvNeXt-Small LiDAR (_cnxs) — rejected for the blend**: single 0.5469 (damaged 0.581 = best single, Timerevo
  recall 0.51, but hillforts 0.496); v11 blend + cnxs nested 0.5622 vs 0.5765 (unstable choices).
* **Learned LiDAR post-processor** (LightGBM rank-to-F1 on blend / per-model peak heights, agreement, spread, mask
  shares, neighbour counts 25–500 m, strongest neighbour, relief / slope, site statistics; inside the exact decoder):
  AUC 0.886 → 0.892, but best pooled 0.5834 (nested 0.5765) vs the hand neighbourhood prior 0.5889 → rejected.

## 25 Oct: v13 = **0.48634** (v12 0.48874). Per-type attribution of the three v13 changes:
* LiDAR P/R 0.603/0.637 → 0.596/**0.676** — neighbourhood prior: recall +0.039 for −0.007 precision (worked).
* Satellite 0.442/0.642 → 0.433/0.634 — retrained heatmap member did not transfer → reverted.
* Geophysics precision 0.760 → 0.720 — MagGr threshold 0.6 (with pole anchors) added FPs → back to 0.75.
* Unconditional damaged/intact hedge (Deepseek), satellite v12 pipeline: no hedge 0.3451, **τ 0 (unconditional) 0.3374**
  (intact FP 130 → 1 103), deployed τ 0.4 0.3667 → rejected; LiDAR already hedges near-unconditionally (dual 0.05).
* **v14** = v12 + LiDAR neighbourhood prior + geophysics pole anchors at MagGr 0.75 (v12 satellite unchanged).
  Rehearsals PASS: satellite/LiDAR set 0.1580; geophysics set 0.4978 (v12 0.4827; object_poly 0.368, FP 390 ≈ v12 386).
  build/koz3_submission_v14_20261025.zip. Expected ≈ 0.492–0.497.

## 25 Oct: v14 = **0.48906** (best; v12 0.48874). LiDAR 0.596/0.676, satellite 0.434/0.639, geophysics 0.758/0.703.
* ROGII Wellbore Geology top solutions (2D alignment U-Nets, HMM / particle-filter decoding, physics candidates + learned
  selector, synthetic-data pretraining, seed blending, CV-first selection): candidate + selector and synthetic data
  (copy-paste) already covered; continuity decoding only maps to fortification lines (weight 5).
* Deepseek round 3: rank averaging (0.294 vs 0.327) and WBF positions (+0.0004) already rejected; model soup does not apply
  to independently trained seeds / recipes; satellite edge duplicates are merged by cross-tile NMS + 8 m fusion + 3 m dedupe.
* New (Global Wheat / YOLO): **satellite D-FINE with mosaic augmentation** (4 crops from different scenes per image, 50 %),
  no-DN + site balance + quality aug, 4 folds × 8k (sp_dfine_l_mos) — fusion study chained.
* ROGII 2nd–5th place writeups read. Transferable: (a) **leave-largest-contribution-out acceptance** (2nd) — implemented as
  scripts/robust_accept.py on per-site counts (greedy removal of the sites carrying the gain → k*, site bootstrap);
  (b) multi-split / multi-resolution ensembles (3rd, 2nd) — runtime-bound for us; (c) synthetic pretraining → short real
  finetune (2nd/4th/5th) — our copy-paste covers part; procedural mounds = future multi-hour job; (d) public LB ≈ noise,
  CV rank-correlates with private (5th: public/private anti-correlated) → keep nested-CV-first selection.
* Calibration of the acceptance test on known changes: **LiDAR neighbourhood prior** (platform +): gain +0.0104, k*=7,
  17/7 sites up/down, bootstrap P 0.97 → systematic. **Satellite balq3 as 6th member** (v12): +0.0047, k*=4, 32/22,
  P 0.82 → fragile (3 sites carry it). Acceptance bar from now on: k* ≥ 5 and P ≥ 0.9.
* Platform v14: Ae 0/0 (off by design — held-out Ae precision ≈ 0.05), Or P/R 0.71/0.24 at thr 0.45 (OOF-chosen on 6
  regions; not re-tuned to the single public region).

## 25 Oct (morning): data-driven CPU studies (competition page / S3 / organiser docs unchanged; 1 success slot left today)
Ideas from ROGII (robust acceptance), DOE GEMS (incomplete labels, asymmetric metric), Vesuvius / ScrollPrize
(per-fragment normalisation, supervision masks, label snapping, nnU-Net long training + multi-patch ensembles).
* **Joint pooled-metric thresholds** (scripts/joint_thr_oof.py; validation-shaped draws of per-region counts, thresholds
  chosen on half the regions, scored on the other half): marginals sat×1.0 best, LiDAR 0.10 +0.0013, Ae≥0.7 +0.001
  (Ae OOF→public precision 3.5× worse → no); honest split results below deployed → **rejected** (curves flat).
* **Region-majority kurgan class** (66 % of satellite kurgans in ≥ 90 % single-class regions): satellite nested 0.3565 vs
  0.3664, LiDAR prune/relabel 0.5861 vs 0.5889 → **rejected** (the disagreement hedge already captures it).
* **Satellite per-region score recalibration** (Vesuvius per-fragment normalisation): base chosen in every fold →
  **rejected** (boosting weak regions adds FPs).
* **Hillfort snap** (data: satellite hillforts lie at median 0.085·half from the scene centre, but the centre is inside
  only 29 %): the gate's centre point moved to the best fused hillfort candidate (score ≥ 0.02) within 0.2·half —
  satellite nested **0.3667 → 0.3797** (+0.013), same setting chosen in all 4 folds, plateau over (ρ 0.2/0.35, s 0.02/0.1);
  hillfort TP 17 → 21, FP 46 → 42; robust k*=4, P 0.89 (borderline, mechanism-driven). Implemented in build/sub_v15
  (sp_or gor_snap {rho 0.2, s_min 0.02}); rehearsal pending a free GPU (training holds 13 of 16 GB locally).
* **LiDAR per-site heat recalibration** (Vesuvius per-fragment percentile normalisation; topk / 99.95th-pct statistic,
  scale clip(c/m, lo, cap)): every variant below deployed (0.5889 → 0.5871 … 0.456) → **rejected**: blind sites are not
  uniformly weak copies of good sites — the heat at real mounds is absent, boosting only amplifies FPs (model-side fix needed).
* **LiDAR framing prior / supervision-mask check** (scripts/li_frame_oof.py): TP rate of deployed detections is flat in
  the distance from the tile centre (0.81 / 0.86 / 0.79 / 0.75 for d_rel ≤ 0.25 … ≤ 1.0) → LiDAR labels cover the whole
  tile (not framed like satellite), unlabelled-negative contamination is small; framing prior nested 0.5879 vs 0.5889,
  k*=2 → **rejected**. LiDAR's loss is recall at blind sites, not edge FPs.
* Robust-acceptance status of today's candidates: hillfort snap is the only nested gain (+0.013 satellite pipeline).
* **Satellite D-FINE + mosaic** (sp_dfine_l_mos): 7th member 0.3604, replacing balq3 0.3595, alone 0.2110 vs deployed 0.3667 → **rejected**.
* **v15** = v14 + hillfort snap (sp_or gor_snap rho 0.2, s_min 0.02). Rehearsals PASS (single atomic write):
  satellite/LiDAR set 0.1580 (= v14), geophysics set 0.4978 (= v14); hillfort-scene set (7 training regions, in-sample
  parity check only) v14 0.772 → v15 0.880 (hillfort FP 13 → 6), every gated scene logged its snap (27–299 m).
  build/koz3_submission_v15_20261025.zip. Expected public ≈ +0.003–0.006 over v14 (hillforts weight 15).
* **LiDAR soft-F1 loss** (_sf1, BCE + weighted soft-F1 with 0.5·FP): alone 0.5310; replacing augx 0.5713 vs deployed blend 0.5765; as 5th member 0.5655 → **rejected**. v15 stays the build for today.
* **Classical site-agnostic mound detection for blind LiDAR sites** (scripts/li_relief_signal.py, li_blob_gbm_oof.py,
  li_blob_geo_oof.py): 63 % of CNN-missed kurgans have ≥ 0.3 m prominence in the 10 m detrended DTM (median 0.37 vs 0.63
  detected), but background bumps ≥ 0.3 m are ~5 200/km² (≈ 600 per missed kurgan). Smooth-mound candidates (1 m Gaussian,
  6 m maxima) + robust shape features (scale ratios, radial drops / circularity, ranks, context) + LightGBM: with CNN
  features the model copies the CNN (uncovered hit rate ≤ 6.6 %); trained on CNN-uncovered candidates without CNN /
  site-identity features: OOF AUC 0.73, top-100 hit rate 8 % (break-even ≈ 12 %), geometry-only 0 hits in top 400 →
  **rejected**: missed mounds are physically ambiguous, the CNN already extracts the geometric signal. Site fallback
  ("CNN blind but many mounds") rejected by inspection: 18 of ~25 zero-coverage sites have no kurgans at all.
* **LiDAR → Ae twins with the Ae expert off** — hidden regression found: the 16-region fusion study deployed LiDAR→Ae twins,
  but fuse only twins into a data type that produced detections, so they vanished when Ae was switched off (v11).
  Public set: region 789Q = LAS (205 MB) + German film. Training Li+Ae regions: 46–52 % of Ae kurgan labels contain a
  Li label's representative point (cemeteries 65–71 %). Nested on 18 Li+Ae sites (GT = Li + Ae): no twins 0.4409;
  twins only where the Ae coverage is a site crop (sheet footprint ≤ 6 km², known at test time) and LiDAR conf ≥ 0.4:
  pooled 0.4695 (+0.026), nested honest 0.4548, robust k*=4, P 0.96, 90 % CI [+0.003, +0.038]. Implemented in
  koz3_infer/run.py (_ae_twins, after dedupe/fuse; config ae_twin {max_km2 6, c_min 0.4}). Rehearsal (036 / 180 / 041):
  Timerevo 92 twins → damaged kurgans TP 182 → 236, FP 23 → 61 (F1 0.278 → 0.342); 041 gate silent (74.7 km²);
  standard sets unchanged (0.1580 / 0.4978, PASS single atomic write).
* **v16** = v14 + hillfort snap + LiDAR→Ae twins: build/koz3_submission_v16_20261025.zip (supersedes v15).
* **Satellite kurgan FP anatomy** (scripts/sp_fp_anatomy.py, deployed v14, 1 215 kurgan points): 409 hits, 96 inside a
  kurgan of the other class, only 9 within 8 m of a missed polygon (no positional lever), **671 > 30 m from any kurgan
  label** (median score 0.36 vs hits 0.49); 269 of them in 52 regions without kurgan labels.
* **Region-level kurgan gate** (scripts/sp_region_gate_oof.py): drop all kurgan points of a region whose best kurgan
  < 0.45 → satellite nested 0.3664 → **0.3705** (oracle 0.3716; (99, 0.45/0.5) in all folds), robust **k*=15, P 1.00,
  CI [+0.002, +0.009]** (kp FP 752 → 668, kt FP 307 → 282, 4 TP lost). Relative threshold on top (score ≥ ρ·region max)
  → rejected (0.3705 at ρ 0.4, worse beyond). Implemented: sp_or kurgan_region_gate 0.45 → **v17**; rehearsal on gated
  regions: 3 regions gated, 7 FP removed, 0 TP lost.
* Per-class satellite region gates on top of the kurgan gate (architecture / hillfort / fortification, g 0.2–0.5): all ≤ kurgan gate alone → rejected. LiDAR site-level kurgan gate: base chosen in every fold → rejected.

## 25 Oct: v17 = **0.48678** (v14 0.48906, −0.0023) — platform attribution
* Satellite counts reconstructed exactly from P/R (only integer solution): **377 GT; v14 241 TP / 628 FP → v17 236 TP /
  628 FP** → the two satellite changes together lost **5 TP at unchanged FP** (snap moving ~5 centre hits out while the
  gate removed ~5 FPs, or the gate removing ~5 TP-bearing kurgans — not separable from one log).
* Ae recall still 0 and total FP unchanged (overall FP ratio moves exactly as 5 fewer TPs) → **LiDAR→Ae twins did not
  fire** (789Q Ae footprint > 6 km² or no LiDAR kurgan ≥ 0.4 inside) — neutral.
* LiDAR / geophysics / Or identical; runtime 11 min 26 s.
* Lesson: both satellite changes were positive on 129 OOF regions (snap +4 TP / 64 hillforts, k*=4; gate −109 FP / −4 TP,
  k*=15) but the public 30-region satellite set moved −5 TP: small-count effects are within public noise. Next: isolate
  (v14 + gate) vs (v14 + snap) with tomorrow's slots before choosing the final configuration.

## 25 Oct (evening): public composition recovered from platform P/R (integer GT solutions across v9–v17)
* GT: LiDAR **482**, satellite **377**, Or **25**, geophysics **1 441** (finds-dominated); size bins: ≤100 m² 1 228,
  ≤500 248, ≤1000 92, ≤5000 84, >10000 17 (5000–10000 unknown, recall always 0).
* **Ae ≈ 525–600 GT** (polygons ≥ 1 669 − Li/Sat/Or 884 − geophysics objects ≲ 280; v9 Ae recall 0.0933 = k/75·…):
  Ae is the LARGEST polygon set of the public data and we score 0 on it (v9 Ae on: ~49 TP for ~2 000 FP, P 0.045).
* **Ae fusion + cemetery-cluster prior** (scripts/ae_cluster_oof.py; spdet + dfj + FCOS + hm OOF, 8 m fusion, prior R 100/250 K 3/8): kurgan TP share 2–6 % in every setting (break-even ≈ 12 %), prior worse (film FPs cluster too); validation-shaped honest effect −0.0011, 0/20 wins → **rejected**.
* **LiDAR attenuated copy-paste** (_augxa, pasted mounds at 30–100 % height, p 0.6): alone 0.5444; replacing augx 0.5667 vs deployed 0.5765; 5th member 0.5686 → **rejected** (recall up at 102 / 030, precision cost larger).
* **Ae per-member anatomy** (key finding): single members DO work on cemetery sheets — FCOS ≥ 0.2: Тимерево 193 dets
  90 % in labels, Гнёздово Лесная 2/3 62–74 %, Днепровская/Ольшанская 20–29 %; open-field sheets (Гочево, Миловидово)
  1–11 %. The 4-member mean fusion (hm scores ≤ 0.17, satellite D-FINE blind on Тимерево) had destroyed this signal.
  figures/ae_labels_*.jpg: Тимерево = carpet of touching mounds (498 labels in one chip; box detectors cap 100 / NMS);
  Гочево labels on ploughed fields barely visible on film.
* **Ae member + local-density ("carpet") filter** (scripts/ae_member_oof.py; FCOS / D-FINE / max; keep if ≥ m dets ≥ t
  within R): high-precision subsets (FCOS t 0.4, R 60, m 30: share 0.48; t 0.3: 0.35) but tiny volume (28–55 of 974
  Ae kurgans) → best pooled +0.0018, honest −0.0062 (6/20 wins, unstable choices) → **rejected**. Lever for the future:
  a dense-object (point-heatmap / counting) film model without detection caps, trained on cemetery sheets.

## 2 Nov 2026 — LiDAR scale-pyramid TTA (after the outage; local data rebuilt)

Context: `data/las_prep` / `data/li_cache` were empty, so the 59 on-disk LAS were re-prepared with the new
`scripts/las_prep_local.py` (the shipped `las_stream_prep.py` needs `docs/download_lists/4_lidar_LAS.urls.txt`, which is
not in the repo) and `python -m koz3.li_train --cache` under `RELIEF_EXTRA=1`. `build_cache` used the untiled
`relief.channels` and OOMed on 3–6 GB sheets → switched to `channels_tiled` (same output, bounded memory).

Recipe: fold 0 of the site-disjoint split (`sp_prep.site_folds`, `kaggle_import` absent so the ledger's L2 folds are not
available), `LI_AUG=1 LI_PASTE=1 RELIEF_EXTRA=1`, 12k it, 67 min. 20 held-out sites / 530 objects
(131 kp, 171 kt, 10 gor, 218 fort) — a larger honest set than the 13-region public LiDAR sample.

Decode-only lever, no retraining: average the sigmoid maps of the same model at several resamplings (scale pyramid),
fixed deployed decode (`decode_site_v2`, thr 0.15, dual 0.05, neighbour prior), 3-flip TTA in every arm.

| arm | thr 0.1 | thr 0.15 | thr 0.2 | thr 0.3 |
|---|---:|---:|---:|---:|
| base (1.0) | 0.4495 | 0.4719 | 0.4803 | 0.4890 |
| pyramid 1.0 / 0.8 / 1.25 | **0.4767** | **0.5046** | **0.5253** | **0.5319** |
| Δ | +0.0272 | +0.0326 | +0.0450 | +0.0428 |

At thr 0.15: kp F1 0.411 → 0.447 (fp 341 → 294), kt 0.633 → 0.704 (tp 97 → 125), gor 0.522 → 0.558, fort 0.044 → 0.035.
13/20 sites better; the two-scale variant (1.0/0.8) gives only +0.0147 → the wide scale matters.

**Caveat (important).** The same probe with the SHIPPED weights on local sites goes the other way (011_Эрентуна 0.790 →
0.510, 004_Боровичи_1 0.848 → 0.777, 189_Макеево 0.620 → 0.602, 104_Шетнево 0.800 → 0.789). Those sites are in the
deployed members' training set, where the model has memorised their exact scale appearance and averaging with perturbed
resamplings dilutes the memorised response — so that number is not evidence against the lever, but it does mean the
deployed blend cannot be validated locally. Cost: 1.5–3× LiDAR inference (011: 12 → 15 s, 004: 12.6 → 35.8 s).
Status: implementation shipped **config-gated and off** (`li.scale_tta`, `koz3.hm.predict_tiled(scales=...)`,
`tests/test_predict_scales.py`); promote only after a full fold-OOF of the deployed blend or a platform A/B.

## 3 Nov 2026 — why the Ae heatmap fails (mechanism), and LiDAR folds 1–2

Rebuilt `data/ae_prep` (`python -m koz3.sp_prep --mod Ae`: 36 sheets → 6 803 chips @0.5 m, 4 site folds of
9/8/3/12 regions) and trained the same `koz3/sp_train --mod Ae` heatmap that the ledger rejects at 0.026.

**Diagnosis (new).** The failure is in the training target/loss, not the representation: at 0.5 m a kurgan is a Gaussian
of σ ≈ 2 px, so a 512 crop of a *field* sheet contains **1–2 positive pixels per batch**. `hm.focal_hm` divides by
`(pos·w).sum()`, so the reported loss swings 0.68 ↔ 5.88 between consecutive checks (it 11500 = 5.88, it 11999 = 0.68) and
the gradient is dominated by single pixels. Measured on the saved checkpoints: focal ≈ 5–7 on fresh training batches,
max heat 0.02–0.08 — the model predicts *nothing*, and the only way this loss can be reduced is to raise probability
everywhere, which is exactly the "fires everywhere on grey film" the ledger recorded. Two runs (lr 3e-4 @ bs 8-scale,
lr 1.3e-4 @ bs 4) bracket the pathology: under-fire and over-fire.

Consequence: the sheet-centre framing prior (labels sit at median d90/half-side < 0.6 on every sheet — measured over all
36 Ae sheets, labels are never outside their sheet) cannot rescue a model that does not fire. The prior is still the
right decode-side rule; it needs a model trained with a stable objective on **dense cemetery crops** (many objects per
crop, e.g. Тимерево 498 labels in one chip), i.e. change the *sampler* and the *loss normalisation*, not the decoder.

**LiDAR scale pyramid, folds 1–2** (same recipe/folds as fold 0):

| fold | sites | base | pyramid 1.0/0.8/1.25 | Δ | better |
|---|---:|---:|---:|---:|---:|
| 0 | 20 | 0.4719 | 0.5046 | **+0.0327** | 13/20 |
| 1 | 10 | 0.5495 | 0.5747 | **+0.0252** | 5/10 (2 ties) |
| 2 | 20 | 0.5431 | 0.5405 | **−0.0026** | 2/20 |

Fold 1 stays positive at every threshold (+0.015…+0.026). Fold 2 shows the failure mode: kp improves (0.518 → 0.535)
but kt degrades (0.575 → 0.517, fp 65 → 113) — the wide scale adds intact-kurgan FPs on that fold. Verdict pending fold 3.

### LiDAR scale pyramid — four-fold verdict (59 held-out sites, deployed decode)

| arm | pooled | Δ vs base | sites better |
|---|---:|---:|---:|
| base 1.0 | 0.5730 | — | — |
| **1.0 / 0.8 / 1.25** | **0.5889** | **+0.0160** | 25/59 |
| 1.0 / 1.25 | 0.5677 | −0.0053 | 20/59 |
| 1.0 / 0.8 | 0.5739 | +0.0010 | 17/59 |

Per fold (base → pyramid): f0 0.4719 → 0.5046 (+0.033), f1 0.5488 → 0.5743 (+0.026), f2 0.5431 → 0.5404 (−0.003),
f3 0.6480 → 0.6593 (+0.011). Positive in 3/4 folds, worst fold −0.003; fold 1 stays positive at every threshold
(+0.015…+0.026). Neither single extra scale reproduces the gain — only the 3-point pyramid does, so part of the effect may
be seed noise (one seed per fold, 12k iters). Expected platform effect ≈ +0.003…0.004 final via kp/kt, i.e. **not** enough
to justify a submission slot on its own; keep it gated off and ship it alongside a bigger change.
