"""Regression check: EtGr layers survive final dedupe, satellite overlaps do not."""
from pathlib import Path
import json
import sys
from shapely.geometry import Point, mapping
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT))


def feature(markup, image, confidence, layer=None):
    props = {"class_name": "object_poly", "region_name": "R", "markup_type": markup,
             "image": image, "confidence": confidence}
    if layer is not None:
        props["source_layer"] = layer          # the source image, tagged by every optical/aerial expert
    return {"type": "Feature", "geometry": mapping(Point(1, 1).buffer(1)), "properties": props}


def apply_final_dedupe(features, dd):
    from koz3.dedupe import dedupe_region
    skip = set(dd.get("skip_markups", ["li"])) | {"etgr"}
    skip_cls = set(dd.get("skip_classes", ["finds_points"]))
    outside = lambda f: (f["properties"].get("markup_type") in skip or
                         f["properties"]["class_name"] in skip_cls)
    return ([f for f in features if outside(f)] +
            dedupe_region([f for f in features if not outside(f)], dd.get("dist_m", 3.0)))


if __name__ == "__main__":
    dd = {"enabled": True, "dist_m": 3.0, "skip_markups": ["li"],
          "skip_classes": ["finds_points"]}
    # EtGr: excluded from dedupe entirely by the orchestrator → both layers survive
    et = apply_final_dedupe([feature("etgr", "layer1", .9), feature("etgr", "layer2", .8)], dd)
    assert len(et) == 2, et
    # any modality: two source images are two reference layers → both survive
    sp = apply_final_dedupe([feature("sp_or", "scene1", .9, "s1.tif"), feature("sp_or", "scene2", .8, "s2.tif")], dd)
    assert len(sp) == 2, sp
    # within ONE source image, two fused objects at the same spot are duplicates → 1 survives
    dup = apply_final_dedupe([feature("sp_or", "obj1", .9, "s1.tif"), feature("sp_or", "obj2", .8, "s1.tif")], dd)
    assert len(dup) == 1, dup
    # the damaged/intact hedge pair shares one "image" id inside one layer → both survive
    hedge = apply_final_dedupe([feature("sp_or", "spobj7", .9, "s1.tif"), feature("sp_or", "spobj7", .89, "s1.tif")], dd)
    assert len(hedge) == 2, hedge
    from koz3_infer.run import _jobs_for
    region = Path("region_with_etgr_and_terfl")
    et_path, tf_path = Path("map.tif"), Path("profile.png")
    plan = {"etgr": [(region, [et_path])], "terfl": [(region, [tf_path])]}
    jobs = _jobs_for(SimpleNamespace(FAMILIES=["etgr", "terfl", "maggr"]), plan)
    assert jobs == [(region, [et_path, tf_path])], jobs
    print("PASS: EtGr keeps 2/2 layers; separate source images stay separate; duplicates within one "
          "source layer are merged; the hedge pair survives; multi-family region queues once")
