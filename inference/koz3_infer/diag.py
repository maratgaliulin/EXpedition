"""Diagnostics printed to the run log (metadata only: environment, input file names / sizes / CRS / LAS headers,
per-expert output counts).

UNUSED — do not wire this back into `main.py`. It used to be called by a "diagnostic build" that wrote no result
and exited 3, on the assumption that the platform returns the run log of a failed run. It does not: a failed run
yields one line, "Контейнер завершился (exit code=3), результат не получен", and nothing else (verified on
2026-11-03 and 2026-11-05, two wasted submissions). The only signal the platform returns for a non-scoring run is
the exit code, so remote diagnostics must use `koz3_infer/probe.py` (bitmask in the exit code).
"""
import json
import os
import sys
import traceback
from pathlib import Path


def p(*a):
    print("[DIAG]", *a, file=sys.stderr, flush=True)


def environment():
    p("python", sys.version.replace("\n", " "))
    for mod in ("numpy", "torch", "torchvision", "pandas", "transformers", "rasterio", "shapely", "pyproj", "laspy",
                "cv2", "scipy", "huggingface_hub", "safetensors", "tokenizers", "PIL", "yaml", "packaging"):
        try:
            m = __import__(mod)
            p(f"pkg {mod:16s} {getattr(m, '__version__', '?'):12s} {getattr(m, '__file__', '')}")
        except Exception as e:
            p(f"pkg {mod:16s} IMPORT FAILED: {type(e).__name__}: {str(e)[:200]}")
    try:
        import torch
        p("cuda", torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    except Exception as e:
        p("cuda check failed", e)


def inputs(in_dir):
    in_dir = Path(in_dir)
    regions = sorted(d for d in in_dir.iterdir() if d.is_dir())
    p(f"{len(regions)} region folders in {in_dir}")
    for r in regions:
        files = sorted(f for f in r.rglob("*") if f.is_file())
        p(f"region {r.name}: " + "; ".join(f"{f.relative_to(r)} ({f.stat().st_size / 1e6:.1f} MB)" for f in files))
        u = r / "UTM.json"
        if u.exists():
            p('  UTM.json ' + ' '.join(u.read_text(encoding='utf-8', errors='replace').split())[:200])
        for f in files:
            try:
                if f.suffix.lower() in (".tif", ".tiff"):
                    import rasterio
                    with rasterio.open(f) as s:
                        p(f"  tif {f.name}: {s.width}x{s.height}x{s.count} {s.dtypes[0]} res {s.res[0]:.4g} crs {s.crs.to_string() if s.crs else None}")
                elif f.suffix.lower() in (".las", ".laz"):
                    import laspy
                    import numpy as np
                    with laspy.open(f) as rd:
                        h = rd.header
                        crs = None
                        try:
                            crs = h.parse_crs()
                        except Exception:
                            pass
                        chunk = next(rd.chunk_iterator(1_000_000))
                        cls = np.asarray(chunk.classification)
                        vals, cnt = np.unique(cls, return_counts=True)
                        area = max((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1]), 1)
                        p(f"  las {f.name}: v{h.version} pf{h.point_format.id} n={h.point_count} density {h.point_count / area:.1f}/m2 "
                          f"crs {crs.to_epsg() if crs else None} mins {[round(x) for x in h.mins]} classes(first 1M) "
                          + ",".join(f"{v}:{c}" for v, c in zip(vals.tolist(), cnt.tolist())))
            except Exception as e:
                p(f"  meta failed {f.name}: {type(e).__name__}: {e}")


def summary(features):
    from collections import Counter
    c = Counter((f["properties"]["region_name"], f["properties"]["markup_type"], f["properties"]["class_name"]) for f in features)
    p(f"TOTAL {len(features)} features")
    for (r, m, k), n in sorted(c.items()):
        p(f"out {r} {m} {k} {n}")
