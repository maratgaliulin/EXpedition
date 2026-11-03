"""Invariants of the scale-pyramid TTA added to `koz3.hm.predict_tiled` (LiDAR decode lever).

The deployment must be bit-identical to the previous behaviour when `li.scale_tta` is absent, and the pyramid must be a
plain weighted average of its single-scale runs, upsampled back to the native grid.

    pytest tests/test_predict_scales.py      (or: python tests/test_predict_scales.py)
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from koz3 import hm  # noqa: E402


class Stub(torch.nn.Module):
    """Deterministic 'net': mean over channels, clipped. Keeps the test independent of any weights."""

    def forward(self, x):
        return x.mean(1, keepdim=True).repeat(1, 3, 1, 1).clamp(-8, 8)


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.3, (12, 96, 128)).astype(np.float32)


def test_no_scales_is_unchanged():
    m = Stub(); x = _img()
    a = hm.predict_tiled(m, x, 3, tile=64, overlap=16, tta=False, dev="cpu")
    b = hm.predict_tiled(m, x, 3, tile=64, overlap=16, tta=False, dev="cpu", scales=None)
    c = hm.predict_tiled(m, x, 3, tile=64, overlap=16, tta=False, dev="cpu", scales=[(1.0, 1.0)])
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_pyramid_is_the_weighted_average():
    m = Stub(); x = _img()
    p = {s: hm.predict_tiled(m, x, 3, tile=64, overlap=16, tta=False, dev="cpu", scales=[(s, 1.0)])
         for s in (0.8, 1.25)}
    got = hm.predict_tiled(m, x, 3, tile=64, overlap=16, tta=False, dev="cpu",
                           scales=[(0.8, 3.0), (1.25, 1.0)])
    want = (3.0 * p[0.8] + 1.0 * p[1.25]) / 4.0
    assert got.shape == (3,) + x.shape[1:] and np.allclose(got, want, atol=1e-5)


if __name__ == "__main__":
    test_no_scales_is_unchanged()
    test_pyramid_is_the_weighted_average()
    print("ok  scale-pyramid TTA: off == old behaviour; pyramid == weighted average")
