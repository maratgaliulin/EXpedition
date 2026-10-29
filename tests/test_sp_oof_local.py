"""Invariants of the local satellite OOF assembler (`scripts/sp_oof_local.py`).

The three post-processing rules are the ones that ship in `inference/koz3_infer/experts/sp_or.py`; these tests pin
their semantics so an "improvement" cannot silently change what the OOF number means.

    pytest tests/test_sp_oof_local.py        (or: python tests/test_sp_oof_local.py)
"""
import os
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from sp_oof_local import Spec, decode  # noqa: E402

KP, KT, GOR = "kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha"
COLS = ["region", "cls", "score", "x", "y", "s_kp", "s_kt"]


def _fused(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_hedge_emits_the_other_kurgan_class_above_tau():
    F = _fused([("r", KP, 0.60, 0, 0, 0.60, 0.50)])
    out = decode(F, Spec(0.25, 0.40, None, None), {}, {})
    assert len(out) == 2, "hedge must emit the intact twin"
    assert set(out.cls) == {KP, KT}
    assert abs(out[out.cls == KT].score.iloc[0] - 0.60 * 0.99) < 1e-9, "twin is emitted slightly discounted"
    # below tau the twin is not emitted
    assert len(decode(F, Spec(0.25, 0.60, None, None), {}, {})) == 1
    # only the kurgan family hedges
    G = _fused([("r", GOR, 0.60, 0, 0, 0.60, 0.60)])
    assert len(decode(G, Spec(0.25, 0.40, None, None), {}, {})) == 1


def test_region_gate_drops_only_that_regions_kurgans():
    F = _fused([("weak", KP, 0.30, 0, 0, 0.30, 0.0),
                ("weak", GOR, 0.30, 10, 0, 0.0, 0.0),
                ("strong", KP, 0.80, 0, 0, 0.80, 0.0)])
    out = decode(F, Spec(0.25, None, 0.45, None), {}, {})
    assert set(out[out.cls.isin([KP, KT])].region) == {"strong"}
    assert set(out[out.cls == GOR].region) == {"weak"}, "the gate is kurgan-only, other classes survive"
    assert len(out) == 2


def test_framing_prior_spares_small_scenes_and_confident_far_objects():
    geom = {"big": [(0.0, 0.0, 1000.0, False)], "small": [(0.0, 0.0, 100.0, True)]}
    F = _fused([("big", KP, 0.30, 900, 0, 0.30, 0.0),      # far (d_rel 0.9) and weak → dropped
                ("big", KP, 0.50, 900, 0, 0.50, 0.0),      # far but confident → kept
                ("small", KP, 0.30, 900, 0, 0.30, 0.0)])   # small scene → prior never applies
    out = decode(F, Spec(0.25, None, None, (0.5, 0.35)), geom, {})
    assert out.region.value_counts().to_dict() == {"big": 1, "small": 1}
    assert out[out.region == "big"].score.iloc[0] == 0.50


def test_threshold_is_per_class():
    F = _fused([("r", KP, 0.20, 0, 0, 0.2, 0.0), ("r", "arkhitektury", 0.20, 5, 0, 0.0, 0.0)])
    out = decode(F, Spec(0.25, None, None, None), {}, {"arkhitektury": 0.15})
    assert set(out.cls) == {"arkhitektury"}


def test_hillfort_gate_swap_drops_and_adds():
    scenes = pd.DataFrame({"region": ["conf", "quiet"], "file": ["a", "b"], "fold": [0, 0],
                           "y": [1, 0], "p": [0.90, 0.05], "x": [100.0, 200.0], "y3857": [0.0, 0.0]})
    F = _fused([("conf", GOR, 0.30, 1000, 0, 0.0, 0.0), ("quiet", GOR, 0.60, 2000, 0, 0.0, 0.0)])
    # swap: the confident scene contributes a centre point, the quiet region loses the detector's hillfort
    out = decode(F, Spec(0.25, None, None, None, ("swap", 0.1)), {}, {}, scenes)
    assert set(out.region) == {"conf"}
    assert len(out) == 2, "detector hillfort in 'conf' + the scene-centre point"
    # add never drops anything
    out = decode(F, Spec(0.25, None, None, None, ("add", 0.1)), {}, {}, scenes)
    assert set(out.region) == {"conf", "quiet"} and len(out) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
