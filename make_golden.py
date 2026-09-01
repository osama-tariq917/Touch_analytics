"""Emit golden test vectors: raw touch events plus the feature vector Python computes.

The Kotlin unit test replays these exact events through the on-device extractor
and asserts the vectors agree. Without this the Android port can silently drift
from the pipeline the results were measured with.
"""
import json, sys
sys.path.insert(0, "/home/claude/nbrun")
import pandas as pd
import touchauth as ta

FEATURE_ORDER = None

def main(src, dst, n=60):
    raw = ta.load_touchalytics(src)
    out = []
    for stroke in ta.segment_strokes(raw):
        f = ta.stroke_features(stroke)
        if f is None:
            continue
        global FEATURE_ORDER
        keys = [k for k in f if k not in ta.METADATA_COLUMNS]
        if FEATURE_ORDER is None:
            FEATURE_ORDER = keys
        out.append({
            "events": [
                {"t": int(r.t_ms), "action": int(r.action), "x": float(r.x), "y": float(r.y),
                 "pressure": float(r.pressure), "area": float(r.area),
                 "orientation": int(r.phone_orient)}
                for r in stroke.itertuples()
            ],
            "features": [float(f[k]) for k in FEATURE_ORDER],
        })
        if len(out) >= n:
            break
    json.dump({"feature_order": FEATURE_ORDER, "cases": out}, open(dst, "w"), indent=1)
    print(f"{len(out)} golden cases, {len(FEATURE_ORDER)} features")
    print(FEATURE_ORDER)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
