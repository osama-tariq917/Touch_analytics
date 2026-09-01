"""Experiments answering the reviewer's specific requests.

R1  cross-device time-to-decision, before and after calibration
R2  strictly inductive normalisation (statistics from enrolment data only)
R3  calibration stability with limited per-device data
R4  device-task confound check in BioIdent
R5  full distributions (IQR), not just medians
R6  classifier sensitivity
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import touchauth as ta

SEED = 42
MIN_DEV = 40
CHANNELS = ["x", "y", "pressure", "area"]
Q = np.linspace(0.001, 0.999, 256)
OUT = "/home/claude/pack"


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def fuse(s, k):
    if k == 1:
        return s
    n = (len(s) // k) * k
    return s[:n].reshape(-1, k).mean(axis=1) if n else np.array([])


def ttd(gen, imp, ks=(1, 3, 5, 10, 20)):
    out = {}
    for k in ks:
        g, i = fuse(gen, k), fuse(imp, k)
        if len(g) < 3 or len(i) < 3:
            out[k] = np.nan
        else:
            out[k] = eer(np.r_[np.ones(len(g)), np.zeros(len(i))], np.r_[g, i])
    return out


def fit_model(gen, imp, cols, kind="rf"):
    n = min(len(imp), len(gen) * 4)
    imp = imp.sample(n, random_state=SEED)
    X = pd.concat([gen, imp])[cols]
    y = np.r_[np.ones(len(gen)), np.zeros(n)]
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    elif kind == "et":
        m = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    else:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return m.fit(X, y)


def quantiles_for(raw, device, exclude_user, cap=None):
    sub = raw[(raw.phone_id == device) & (raw.user_id != exclude_user)]
    if cap is not None and len(sub) > cap:
        sub = sub.sample(cap, random_state=SEED)
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


def inductive_bins(enrol_frame, apply_frames, cols_scale, nbins=16):
    """Bin edges learned ONLY from enrolment strokes, then applied to test data."""
    edges = {}
    for c in cols_scale:
        v = enrol_frame[c].to_numpy(float)
        edges[c] = np.unique(np.quantile(v, np.linspace(0, 1, nbins + 1)[1:-1]))
    out = []
    for f in [enrol_frame] + list(apply_frames):
        g = f.copy()
        for c in cols_scale:
            g[c] = np.searchsorted(edges[c], g[c].to_numpy(float)) / float(nbins)
        out.append(g)
    return out


def main():
    raw = ta.load_bioident("/home/claude/bio/rawdata.csv")
    bio = pd.read_csv("strokes_bioident.csv")

    # ---- R4: device-task confound -------------------------------------
    tab = pd.crosstab(bio.phone_id, bio.doc_id)
    tab.to_csv(f"{OUT}/device_task_crosstab.csv")
    print("R4 device x task strokes\n", tab, "\n")
    both = (tab > 0).all(axis=1).sum()
    print(f"R4: handsets carrying BOTH task types: {both} of {len(tab)}\n")

    plain = ta.normalise_devices(bio, "rank")
    cols = ta.feature_columns(plain)
    scale_cols = [c for c in ta.SCALE_DEPENDENT if c in cols]
    counts = bio.groupby(["user_id", "phone_id"]).size()

    rows, ttd_rows = [], []
    for user in sorted(bio.user_id.unique()):
        devs = [p for (u, p), n in counts.items() if u == user and n >= MIN_DEV]
        if len(devs) < 2:
            continue
        home, other = devs[0], devs[1]

        gh_raw = bio[(bio.user_id == user) & (bio.phone_id == home)]
        half = len(gh_raw) // 2
        tr_raw, same_raw = gh_raw.iloc[:half], gh_raw.iloc[half:]
        cross_raw = bio[(bio.user_id == user) & (bio.phone_id == other)]
        imp_raw = bio[bio.user_id != user]
        if len(tr_raw) < 20 or len(cross_raw) < 10:
            continue
        imp_t = imp_raw.sample(min(len(imp_raw), 600), random_state=SEED)

        row = {"user_id": user}

        # ---- uncalibrated + calibrated (raw scale) ----
        m_raw = fit_model(tr_raw, imp_raw, cols, "rf")
        s_imp = m_raw.predict_proba(imp_t[cols])[:, 1]
        s_same = m_raw.predict_proba(same_raw[cols])[:, 1]
        s_cross = m_raw.predict_proba(cross_raw[cols])[:, 1]

        src = quantiles_for(raw, other, user)
        dst = quantiles_for(raw, home, user)
        mapped = strokes_of(calibrate(raw[(raw.user_id == user) & (raw.phone_id == other)], src, dst))
        if len(mapped) < 10:
            continue
        s_cal = m_raw.predict_proba(mapped[cols])[:, 1]

        row["same_device"] = eer(np.r_[np.ones(len(s_same)), np.zeros(len(s_imp))], np.r_[s_same, s_imp])
        row["cross_uncal"] = eer(np.r_[np.ones(len(s_cross)), np.zeros(len(s_imp))], np.r_[s_cross, s_imp])
        row["cross_cal"] = eer(np.r_[np.ones(len(s_cal)), np.zeros(len(s_imp))], np.r_[s_cal, s_imp])

        # ---- R1: cross-device time to decision ----
        for label, sc in (("uncal", s_cross), ("cal", s_cal), ("same", s_same)):
            for k, v in ttd(sc, s_imp).items():
                ttd_rows.append({"user_id": user, "condition": label, "k": k, "eer": v})

        # ---- R2: inductive vs transductive feature normalisation ----
        tr_n, same_n, cross_n, imp_n = inductive_bins(tr_raw, [same_raw, cross_raw, imp_raw], scale_cols)
        m_ind = fit_model(tr_n, imp_n, cols, "rf")
        imp_n_t = imp_n.loc[imp_t.index]
        s_i = m_ind.predict_proba(imp_n_t[cols])[:, 1]
        row["cross_featnorm_inductive"] = eer(
            np.r_[np.ones(len(cross_n)), np.zeros(len(s_i))],
            np.r_[m_ind.predict_proba(cross_n[cols])[:, 1], s_i])

        gh_t = plain[(plain.user_id == user) & (plain.phone_id == home)]
        m_tr = fit_model(gh_t.iloc[:half], plain[plain.user_id != user], cols, "rf")
        cross_t = plain[(plain.user_id == user) & (plain.phone_id == other)]
        imp_tt = plain.loc[imp_t.index]
        s_it = m_tr.predict_proba(imp_tt[cols])[:, 1]
        row["cross_featnorm_transductive"] = eer(
            np.r_[np.ones(len(cross_t)), np.zeros(len(s_it))],
            np.r_[m_tr.predict_proba(cross_t[cols])[:, 1], s_it])

        # ---- R3: calibration stability with limited per-device data ----
        for cap in (200, 1000, 5000):
            src_c = quantiles_for(raw, other, user, cap=cap)
            dst_c = quantiles_for(raw, home, user, cap=cap)
            mp = strokes_of(calibrate(raw[(raw.user_id == user) & (raw.phone_id == other)], src_c, dst_c))
            if len(mp) < 10:
                row[f"cross_cal_{cap}"] = np.nan
                continue
            sc = m_raw.predict_proba(mp[cols])[:, 1]
            row[f"cross_cal_{cap}"] = eer(np.r_[np.ones(len(sc)), np.zeros(len(s_imp))], np.r_[sc, s_imp])

        # ---- R6: classifier sensitivity ----
        for kind in ("et", "lr"):
            m2 = fit_model(tr_raw, imp_raw, cols, kind)
            si = m2.predict_proba(imp_t[cols])[:, 1]
            sc = m2.predict_proba(mapped[cols])[:, 1]
            row[f"cross_cal_{kind}"] = eer(np.r_[np.ones(len(sc)), np.zeros(len(si))], np.r_[sc, si])

        rows.append(row)
        print(f"user {user:>3} same {row['same_device']:.3f} uncal {row['cross_uncal']:.3f} "
              f"cal {row['cross_cal']:.3f} ind {row['cross_featnorm_inductive']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.round(4).to_csv(f"{OUT}/reviewer_response_per_user.csv", index=False)

    print("\n=== R5 distributions (median [IQR]), n =", len(df), "===")
    summ = []
    for c in [c for c in df.columns if c != "user_id"]:
        v = df[c].dropna()
        summ.append({"condition": c, "median": v.median(),
                     "q1": v.quantile(.25), "q3": v.quantile(.75), "n": len(v)})
        print(f"{c:<30} {v.median():.3f}  [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]")
    pd.DataFrame(summ).round(3).to_csv(f"{OUT}/reviewer_response_summary.csv", index=False)

    t = pd.DataFrame(ttd_rows)
    piv = t.groupby(["condition", "k"]).eer.median().unstack()
    piv.round(3).to_csv(f"{OUT}/cross_device_time_to_decision.csv")
    print("\n=== R1 cross-device time to decision (median EER) ===")
    print(piv.round(3))


if __name__ == "__main__":
    main()
