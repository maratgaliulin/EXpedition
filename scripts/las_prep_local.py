"""Local twin of scripts/las_stream_prep.py: build data/las_prep/<region>.npz from LAS files already on disk.

`las_stream_prep.py` needs `docs/download_lists/4_lidar_LAS.urls.txt` (S3 URLs + sizes) to stream the clouds; when the
dataset is already downloaded under data/train-expds-3 (or the URL list is gone) that path cannot run. This script walks
the downloaded `<region>/01_*_La_точки/*.las` tree instead, builds the DTM with koz3.dtm (identical parameters) and
writes the same npz, so koz3.li_train --cache / --oof work unchanged.

  python scripts/las_prep_local.py [workers=2]        # skips regions that are already prepared
"""
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from koz3 import dtm  # noqa: E402
from koz3.labels import DATA, region_utm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data", "las_prep")


def one(las):
    region = os.path.relpath(las, DATA).split(os.sep)[0]
    out = os.path.join(OUT, region + ".npz")
    if os.path.exists(out):
        return f"skip {region}"
    t = time.time()
    d, v, origin = dtm.build(las)
    dtm.save(out, d, v, origin, region_utm(os.path.join(DATA, region)))
    return (f"{region:36s} {os.path.getsize(las) / 1e6:8.1f} MB → {d.shape} res {origin[2]} "
            f"valid {v.mean():.2f} {time.time() - t:.0f}s")


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    os.makedirs(OUT, exist_ok=True)
    las = sorted(glob.glob(os.path.join(DATA, "*", "01_*", "*.las")))
    print(len(las), "LAS clouds", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(one, las):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
