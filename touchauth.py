"""Shared pipeline for touch-based continuous authentication."""

import numpy as np
import pandas as pd

RAW_COLUMNS = ["phone_id", "user_id", "doc_id", "t_ms", "action",
               "phone_orient", "x", "y", "pressure", "area", "finger_orient"]

ACTION_DOWN, ACTION_UP, ACTION_MOVE = 0, 1, 2
MIN_POINTS = 5
MIN_DISTANCE_PX = 10.0
N_RANK_BINS = 16
METADATA_COLUMNS = ["user_id", "doc_id", "phone_id", "session"]


def load_touchalytics(path):
    """Column 1 is the user. Column 0 is the handset."""
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS)
    df["session"] = df["doc_id"].astype(int)
    return df.sort_values(["user_id", "doc_id", "t_ms"], kind="stable").reset_index(drop=True)


def load_bioident(path):
    """BioIdent ships a header row, a trailing comma and no finger-orientation column.

    A session here is (handset, document type), because the same user visits
    more than one handset and those visits must never be merged.
    """
    df = pd.read_csv(path, low_memory=False)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = ["phone_id", "user_id", "doc_id", "t_ms", "action",
                  "phone_orient", "x", "y", "pressure", "area"]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df["finger_orient"] = 0.0
    df["session"] = (df["phone_id"].astype(int).astype(str) + "_"
                     + df["doc_id"].astype(int).astype(str))
    return df.sort_values(["user_id", "phone_id", "doc_id", "t_ms"],
                          kind="stable").reset_index(drop=True)


def segment_strokes(df):
    """Yield complete down..up sequences, cut inside one (user, session)."""
    for _, session in df.groupby(["user_id", "session"], sort=False):
        session = session.reset_index(drop=True)
        start = None
        for i, action in enumerate(session["action"].to_numpy()):
            if action == ACTION_DOWN:
                start = i
            elif action == ACTION_UP and start is not None:
                stroke = session.iloc[start:i + 1]
                start = None
                if len(stroke) >= MIN_POINTS:
                    yield stroke


def _percentiles(values, prefix):
    if values.size == 0:
        return {f"{prefix}_p{p}": 0.0 for p in (20, 50, 80)}
    return {f"{prefix}_p{p}": float(np.percentile(values, p)) for p in (20, 50, 80)}


def stroke_features(stroke):
    x = stroke["x"].to_numpy(float)
    y = stroke["y"].to_numpy(float)
    t = stroke["t_ms"].to_numpy(float) / 1000.0
    pressure = stroke["pressure"].to_numpy(float)
    area = stroke["area"].to_numpy(float)

    dx, dy, dt = np.diff(x), np.diff(y), np.diff(t)
    dt = np.where(dt <= 0, np.nan, dt)

    step = np.hypot(dx, dy)
    traj_len = float(step.sum())
    direct = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    if direct < MIN_DISTANCE_PX or traj_len == 0:
        return None

    duration = float(t[-1] - t[0])
    if duration <= 0:
        return None

    velocity = step / dt
    velocity = velocity[np.isfinite(velocity)]
    if velocity.size < 2:
        return None
    accel = np.diff(velocity) / np.nanmedian(dt)
    accel = accel[np.isfinite(accel)]

    ux, uy = (x[-1] - x[0]) / direct, (y[-1] - y[0]) / direct
    deviation = np.abs(-uy * (x - x[0]) + ux * (y - y[0]))

    heading = np.arctan2(dy, dx)
    mrl = float(np.hypot(np.mean(np.cos(heading)), np.mean(np.sin(heading))))
    end_angle = float(np.arctan2(y[-1] - y[0], x[-1] - x[0]))

    mid = len(stroke) // 2
    tail = velocity[-3:] if velocity.size >= 3 else velocity

    feats = {
        "user_id": int(stroke["user_id"].iloc[0]),
        "doc_id": int(stroke["doc_id"].iloc[0]),
        "phone_id": int(stroke["phone_id"].iloc[0]),
        "session": stroke["session"].iloc[0],
        "n_points": len(stroke),
        "duration": duration,
        "direct_distance": direct,
        "trajectory_length": traj_len,
        "straightness": direct / traj_len,
        "end_angle": end_angle,
        "direction_flag": int(np.round(end_angle / (np.pi / 2)) % 4),
        "mean_resultant_length": mrl,
        "mean_velocity": traj_len / duration,
        "tail_velocity": float(np.median(tail)),
        "max_deviation": float(deviation.max()),
        "mean_step": float(step.mean()),
        "std_step": float(step.std()),
        "mean_pressure": float(pressure.mean()),
        "std_pressure": float(pressure.std()),
        "mid_pressure": float(pressure[mid]),
        "mean_area": float(area.mean()),
        "std_area": float(area.std()),
        "mid_area": float(area[mid]),
        "phone_orient": int(stroke["phone_orient"].iloc[0]),
        "points_per_second": len(stroke) / duration,
    }
    feats.update(_percentiles(velocity, "velocity"))
    feats.update(_percentiles(accel, "accel"))
    feats.update(_percentiles(deviation, "deviation"))
    feats.update(_percentiles(pressure, "pressure"))
    feats.update(_percentiles(area, "area"))
    return feats


def build_strokes(raw):
    rows = [f for f in (stroke_features(s) for s in segment_strokes(raw)) if f is not None]
    return pd.DataFrame(rows)


# Features carrying handset-specific physical scale or sampling rate.
SCALE_DEPENDENT = [
    "n_points", "points_per_second",
    "direct_distance", "trajectory_length", "mean_velocity", "tail_velocity",
    "max_deviation", "mean_step", "std_step",
    "mean_pressure", "std_pressure", "mid_pressure",
    "mean_area", "std_area", "mid_area",
    *[f"velocity_p{p}" for p in (20, 50, 80)],
    *[f"accel_p{p}" for p in (20, 50, 80)],
    *[f"deviation_p{p}" for p in (20, 50, 80)],
    *[f"pressure_p{p}" for p in (20, 50, 80)],
    *[f"area_p{p}" for p in (20, 50, 80)],
]

PIXEL_FEATURES = [
    "direct_distance", "trajectory_length", "max_deviation", "mean_step",
    "std_step", "mean_velocity", "tail_velocity",
    *[f"velocity_p{p}" for p in (20, 50, 80)],
    *[f"accel_p{p}" for p in (20, 50, 80)],
    *[f"deviation_p{p}" for p in (20, 50, 80)],
]

SENSOR_FEATURES = [
    "mean_pressure", "std_pressure", "mid_pressure",
    "mean_area", "std_area", "mid_area",
    *[f"pressure_p{p}" for p in (20, 50, 80)],
    *[f"area_p{p}" for p in (20, 50, 80)],
]


def normalise_devices(df, method="rank", dpi_map=None):
    """Strip handset-specific scale so features describe behaviour, not hardware.

    none  leave untouched (ablation baseline)
    rank  per-handset equal-frequency binning of every scale-dependent feature
    dpi   pixels to millimetres via dpi_map, plus rank normalisation of the
          pressure and area channels, which have no physical unit
    """
    if method == "none":
        return df.copy()

    out = df.copy()

    def bin_within_device(columns):
        for col in columns:
            ordered = out.groupby("phone_id")[col].rank(method="first")
            out[col] = (ordered.groupby(out["phone_id"])
                        .transform(lambda s: pd.qcut(s, min(N_RANK_BINS, s.nunique()),
                                                     labels=False, duplicates="drop"))
                        .astype(float))
            out[col] = out.groupby("phone_id")[col].transform(
                lambda s: s / max(s.max(), 1))

    if method == "rank":
        bin_within_device(SCALE_DEPENDENT)
        return out

    if method == "dpi":
        if not dpi_map:
            raise ValueError("method='dpi' needs a {phone_id: (xdpi, ydpi)} map")
        missing = set(out["phone_id"].unique()) - set(dpi_map)
        if missing:
            raise ValueError(f"no dpi entry for handsets {sorted(missing)}")
        for phone_id, (xdpi, ydpi) in dpi_map.items():
            mm_per_px = 25.4 / np.mean([xdpi, ydpi])
            mask = out["phone_id"] == phone_id
            out.loc[mask, PIXEL_FEATURES] = out.loc[mask, PIXEL_FEATURES] * mm_per_px
        bin_within_device(SENSOR_FEATURES + ["n_points", "points_per_second"])
        return out

    raise ValueError(f"unknown normalisation method: {method}")


def feature_columns(df):
    return [c for c in df.columns if c not in METADATA_COLUMNS]
