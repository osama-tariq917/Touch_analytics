"""Dr Soodamani's suggestion, tested.

Two ideas, distinct from what we did before:

1. Normalise the RAW SENSOR DATA using device metadata, before features are
   computed -- not the feature vectors afterwards.
2. Normalise ASYMMETRICALLY: leave the training device alone and map only the
   test device's samples into the trained device's reference frame.

Implemented as quantile calibration. For each raw channel (x, y, pressure,
area) we build the distribution that channel takes on each handset, then map a
test-device sample to the value at the same percentile on the enrolment
handset. Screen geometry is handled separately by a physical-units variant
using the recorded dpi.

Leakage guard: the calibration distributions for a handset are built from OTHER
users on that handset, never from the enrolled user's own strokes, so nothing
about the target person leaks into the mapping.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

import touchauth as ta

SEED = 42
MIN_DEVICE_STROKES = 40
CHANNELS = ["x", "y", "pressure", "area"]
QUANTILES = np.linspace(0.001, 0.999, 256)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def user_model(gen, imp, cols):
    n = min(len(imp), len(gen) * 4)
    imp = imp.sample(n, random_state=SEED)
    X = pd.concat([gen, imp])[cols]
    y = np.r_[np.ones(len(gen)), np.zeros(n)]
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  n_jobs=-1, random_state=SEED).fit(X, y)


def channel_quantiles(raw, device, exclude_user):
    """Reference distribution for each channel on one handset."""
    sub = raw[(raw.phone_id == device) & (raw.user_id != exclude_user)]
    if len(sub) < 500:
        sub = raw[raw.phone_id == device]
    return {c: np.quantile(sub[c].to_numpy(float), QUANTILES) for c in CHANNELS}


def calibrate(events, source_q, target_q):
    """Map raw samples from the source handset onto the target handset's scale."""
    out = events.copy()
    for c in CHANNELS:
        pct = np.interp(out[c].to_numpy(float), source_q[c], QUANTILES)
        out[c] = np.interp(pct, QUANTILES, target_q[c])
    return out


def strokes_from(raw_subset):
    rows = [f for f in (ta.stroke_features(s) for s in ta.segment_strokes(raw_subset))
            if f is not None]
    return pd.DataFrame(rows)


def main():
    raw = ta.load_bioident("/home/claude/bio/rawdata.csv")
    bio = pd.read_csv("strokes_bioident.csv")

    counts = bio.groupby(["user_id", "phone_id"]).size()
    plain = ta.normalise_devices(bio, "rank")
    cols = ta.feature_columns(plain)

    results = []
    for user in sorted(bio.user_id.unique()):
        devices = [p for (u, p), n in counts.items() if u == user and n >= MIN_DEVICE_STROKES]
        if len(devices) < 2:
            continue
        home, other = devices[0], devices[1]

        # --- baseline: feature-level normalisation (what we reported before)
        gh = plain[(plain.user_id == user) & (plain.phone_id == home)]
        half = len(gh) // 2
        g_train, g_same = gh.iloc[:half], gh.iloc[half:]
        g_cross = plain[(plain.user_id == user) & (plain.phone_id == other)]
        imp = plain[plain.user_id != user]
        if len(g_train) < 20 or len(g_cross) < 10:
            continue

        model = user_model(g_train, imp, cols)
        imp_t = imp.sample(min(len(imp), 600), random_state=SEED)

        def score(genuine, m, columns):
            X = pd.concat([genuine, imp_t])[columns]
            y = np.r_[np.ones(len(genuine)), np.zeros(len(imp_t))]
            return eer(y, m.predict_proba(X)[:, 1])

        row = {"user_id": user, "enrol": home, "test": other,
               "same_device": score(g_same, model, cols),
               "cross_feature_norm": score(g_cross, model, cols)}

        # --- supervisor's approach: calibrate the test handset's RAW samples
        source_q = channel_quantiles(raw, other, exclude_user=user)
        target_q = channel_quantiles(raw, home, exclude_user=user)
        test_raw = raw[(raw.user_id == user) & (raw.phone_id == other)]
        mapped = calibrate(test_raw, source_q, target_q)
        mapped_strokes = strokes_from(mapped)
        if len(mapped_strokes) < 10:
            continue

        # The enrolment side is untouched, so the model is trained on raw-scale
        # features from the home handset only.
        gh_raw = bio[(bio.user_id == user) & (bio.phone_id == home)]
        g_train_raw = gh_raw.iloc[:len(gh_raw) // 2]
        imp_raw = bio[bio.user_id != user]
        model_raw = user_model(g_train_raw, imp_raw, cols)
        imp_raw_t = imp_raw.sample(min(len(imp_raw), 600), random_state=SEED)

        X = pd.concat([mapped_strokes[cols], imp_raw_t[cols]])
        y = np.r_[np.ones(len(mapped_strokes)), np.zeros(len(imp_raw_t))]
        row["cross_raw_calibrated"] = eer(y, model_raw.predict_proba(X)[:, 1])

        # control: same model, uncalibrated test samples
        g_cross_raw = bio[(bio.user_id == user) & (bio.phone_id == other)]
        X = pd.concat([g_cross_raw[cols], imp_raw_t[cols]])
        y = np.r_[np.ones(len(g_cross_raw)), np.zeros(len(imp_raw_t))]
        row["cross_raw_uncalibrated"] = eer(y, model_raw.predict_proba(X)[:, 1])

        results.append(row)
        print(f"user {user:>3}  same {row['same_device']:.3f}  "
              f"feat-norm {row['cross_feature_norm']:.3f}  "
              f"raw-uncal {row['cross_raw_uncalibrated']:.3f}  "
              f"raw-CAL {row['cross_raw_calibrated']:.3f}", flush=True)

    df = pd.DataFrame(results)
    df.round(4).to_csv("/home/claude/pack/calibration_per_user.csv", index=False)
    print("\n=== medians over", len(df), "users ===")
    for c in ["same_device", "cross_feature_norm", "cross_raw_uncalibrated",
              "cross_raw_calibrated"]:
        print(f"{c:<24} {df[c].median():.3f}")
    improved = (df.cross_raw_calibrated < df.cross_raw_uncalibrated).sum()
    print(f"\nusers improved by calibration: {improved} / {len(df)}")


if __name__ == "__main__":
    main()
