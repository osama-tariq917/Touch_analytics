"""Per-device analysis requested in review.

E1  k-fold cross-validation per handset, with ROC curves, to see whether some
    devices support authentication better than others
E2  which handset makes the best normalisation template
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

import touchauth as ta

SEED = 42
OUT = "/home/claude/pack"
CHANNELS = ["x", "y", "pressure", "area"]
Q = np.linspace(0.001, 0.999, 256)
MIN_DEV = 40


def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def fit(gen, imp, cols):
    n = min(len(imp), len(gen) * 4)
    imp = imp.sample(n, random_state=SEED)
    X = pd.concat([gen, imp])[cols]
    y = np.r_[np.ones(len(gen)), np.zeros(n)]
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  n_jobs=-1, random_state=SEED).fit(X, y)


# ---------------------------------------------------------------- E1
def per_device_kfold(df, cols, k=5, min_users=3):
    """5-fold CV per user, pooled per handset, with a device-level ROC."""
    rows, curves = [], {}
    for device, grp in df.groupby("phone_id"):
        users = [u for u, g in grp.groupby("user_id") if len(g) >= MIN_DEV]
        if len(users) < min_users:
            continue
        all_y, all_s, per_user = [], [], []
        for user in users:
            gen = grp[grp.user_id == user]
            imp = df[(df.user_id != user)]
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
            ys, ss = [], []
            labels = np.ones(len(gen))
            for tr_idx, te_idx in skf.split(gen, labels):
                model = fit(gen.iloc[tr_idx], imp, cols)
                imp_te = imp.sample(min(len(imp), 400), random_state=SEED)
                X = pd.concat([gen.iloc[te_idx], imp_te])[cols]
                y = np.r_[np.ones(len(te_idx)), np.zeros(len(imp_te))]
                s = model.predict_proba(X)[:, 1]
                ys.append(y); ss.append(s)
            y = np.concatenate(ys); s = np.concatenate(ss)
            per_user.append(eer_of(y, s))
            all_y.append(y); all_s.append(s)
        y = np.concatenate(all_y); s = np.concatenate(all_s)
        fpr, tpr, _ = roc_curve(y, s)
        curves[int(device)] = (fpr.tolist(), tpr.tolist())
        rows.append({"device": int(device), "users": len(users),
                     "strokes": int(len(grp)),
                     "eer_pooled": eer_of(y, s),
                     "eer_median_user": float(np.median(per_user))})
    return pd.DataFrame(rows).sort_values("eer_pooled"), curves


# ---------------------------------------------------------------- E2
def quantiles_for(raw, device, exclude_user):
    sub = raw[(raw.phone_id == device) & (raw.user_id != exclude_user)]
    if len(sub) < 200:
        sub = raw[raw.phone_id == device]
    return {c: np.quantile(sub[c].to_numpy(float), Q) for c in CHANNELS}


def calibrate(events, src, dst):
    out = events.copy()
    for c in CHANNELS:
        pct = np.interp(out[c].to_numpy(float), src[c], Q)
        out[c] = np.interp(pct, Q, dst[c])
    return out


def strokes_of(sub):
    rows = [f for f in (ta.stroke_features(s) for s in ta.segment_strokes(sub)) if f is not None]
    return pd.DataFrame(rows)


def best_template(raw, bio, cols):
    """For each enrolment handset, the cross-device error rate it yields."""
    counts = bio.groupby(["user_id", "phone_id"]).size()
    rows = []
    for user in sorted(bio.user_id.unique()):
        devs = [p for (u, p), n in counts.items() if u == user and n >= MIN_DEV]
        if len(devs) < 2:
            continue
        for home in devs:
            others = [d for d in devs if d != home]
            gh = bio[(bio.user_id == user) & (bio.phone_id == home)]
            if len(gh) < 40:
                continue
            tr = gh.iloc[:len(gh) // 2]
            imp = bio[bio.user_id != user]
            model = fit(tr, imp, cols)
            imp_t = imp.sample(min(len(imp), 500), random_state=SEED)
            for other in others:
                src = quantiles_for(raw, other, user)
                dst = quantiles_for(raw, home, user)
                mapped = strokes_of(calibrate(
                    raw[(raw.user_id == user) & (raw.phone_id == other)], src, dst))
                if len(mapped) < 10:
                    continue
                X = pd.concat([mapped[cols], imp_t[cols]])
                y = np.r_[np.ones(len(mapped)), np.zeros(len(imp_t))]
                rows.append({"user_id": user, "enrol_device": int(home),
                             "test_device": int(other),
                             "eer": eer_of(y, model.predict_proba(X)[:, 1])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    bio = pd.read_csv("strokes_bioident.csv")
    raw = ta.load_bioident("/home/claude/bio/rawdata.csv")
    d = ta.normalise_devices(bio, "rank")
    cols = ta.feature_columns(d)

    print("=== E1: per-device k-fold ===", flush=True)
    table, curves = per_device_kfold(d, cols)
    table.round(3).to_csv(f"{OUT}/per_device_kfold.csv", index=False)
    json.dump(curves, open(f"{OUT}/per_device_roc.json", "w"))
    print(table.round(3).to_string(index=False))

    print("\n=== E2: which handset is the best enrolment template ===", flush=True)
    bt = best_template(raw, bio, cols)
    bt.round(4).to_csv(f"{OUT}/best_template_pairs.csv", index=False)
    summ = (bt.groupby("enrol_device")
              .agg(median_eer=("eer", "median"), pairs=("eer", "size"))
              .sort_values("median_eer"))
    summ.round(3).to_csv(f"{OUT}/best_template_summary.csv")
    print(summ.round(3).to_string())
