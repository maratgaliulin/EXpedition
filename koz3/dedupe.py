"""Region-level duplicate removal for imagery where overlapping views share targets.

Every source image is an independent reference layer for ALL modalities (organiser clarification, adopted in the v20
build): two views of the same physical object are two reference objects, so predictions from different source images
must never be merged.  The inference experts tag each feature with `source_layer` = the source image name and this
function only compares features that share it; EtGr is additionally excluded outright by the orchestrator.

Within one source layer, per (region, markup_type, class family) — kurgan classes form one family, so a mound reported
damaged and intact at the same spot is one object — predictions are visited by descending confidence; a prediction is
dropped when it overlaps an already kept one from a DIFFERENT fused object (property "image"; the damaged/intact hedge
pair shares one "image" id and is therefore kept): its representative point lies inside the kept polygon (or vice
versa), or the centroids are closer than dist_m. Objects of different data types are never merged (separate labels).
"""
import numpy as np

KURGAN = ("kurgany_povrezhdennye", "kurgany_tselye")


def _fam(c):
    return "kurgan" if c in KURGAN else c


def dedupe_region(features, dist_m=4.0):
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    groups = {}
    for i, f in enumerate(features):
        p = f["properties"]
        groups.setdefault((p.get("region_name"), p.get("markup_type"),
                           p.get("source_layer"), _fam(p["class_name"])), []).append(i)
    keep = []
    for _, idx in groups.items():
        idx = sorted(idx, key=lambda i: -features[i]["properties"].get("confidence", 0.0))
        geoms = [shape(features[i]["geometry"]) for i in idx]
        reps = [g.representative_point() for g in geoms]
        cents = [g.centroid for g in geoms]
        k = 1 / np.cos(np.radians(np.degrees(2 * np.arctan(np.exp(cents[0].y / 6378137.0)) - np.pi / 2))) if cents else 1.0
        tree = STRtree([g.buffer(dist_m * k) for g in geoms])
        taken = np.zeros(len(idx), bool); kept_local = []
        for a in range(len(idx)):
            if taken[a]:
                continue
            kept_local.append(a)
            for b in tree.query(geoms[a].buffer(dist_m * k)):
                if b <= a or taken[b]:
                    continue
                ia = features[idx[a]]["properties"].get("image"); ib = features[idx[b]]["properties"].get("image")
                if ia is not None and ia == ib:      # same image: distinct objects, never duplicates
                    continue
                if (geoms[a].contains(reps[b]) or geoms[b].contains(reps[a])
                        or cents[a].distance(cents[b]) < dist_m * k):
                    taken[b] = True
        keep += [idx[a] for a in kept_local]
    return [features[i] for i in sorted(keep)]
