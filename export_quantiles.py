"""Build the per-handset quantile tables the app needs for raw-sensor calibration.

Produces app/src/main/assets/device_quantiles.json:

    { "n_points": 256,
      "quantiles": [0.001, ...],
      "devices": { "<device_key>": { "x": [...], "y": [...],
                                     "pressure": [...], "area": [...] } } }

At run time the app maps an incoming sample to its percentile on the handset
that produced it, then reads off the value at that percentile on the handset the
template was enrolled on. Tables come from population data, never from the
enrolled user, so they can ship with the app or be fetched per device model.
"""
import json
import sys

import numpy as np
import touchauth as ta

N = 256
Q = np.linspace(0.001, 0.999, N)
CHANNELS = ["x", "y", "pressure", "area"]


def build(raw, key_col="phone_id"):
    devices = {}
    for dev, grp in raw.groupby(key_col):
        if len(grp) < 500:
            continue
        devices[str(int(dev))] = {
            c: [round(float(v), 6) for v in np.quantile(grp[c].to_numpy(float), Q)]
            for c in CHANNELS
        }
    return {"n_points": N, "quantiles": [round(float(q), 6) for q in Q], "devices": devices}


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/bio/rawdata.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "device_quantiles.json"
    raw = ta.load_bioident(src)
    table = build(raw)
    json.dump(table, open(dst, "w"))
    print(f"{len(table['devices'])} handsets, {N} points per channel -> {dst}")
