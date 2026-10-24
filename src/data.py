"""Data loading and feature engineering.

Uses the real UCI "Steel Industry Energy Consumption" dataset when
data/Steel_industry_data.csv is present. Otherwise generates a synthetic
dataset with the same schema so the app always runs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Steel_industry_data.csv"

RENAME = {
    "date": "timestamp",
    "Usage_kWh": "usage_kwh",
    "Lagging_Current_Reactive.Power_kVarh": "lagging_kvarh",
    "Leading_Current_Reactive_Power_kVarh": "leading_kvarh",
    "CO2(tCO2)": "co2_t",
    "Lagging_Current_Power_Factor": "lagging_pf",
    "Leading_Current_Power_Factor": "leading_pf",
    "NSM": "nsm",
    "WeekStatus": "week_status",
    "Day_of_week": "day_of_week",
    "Load_Type": "load_type",
}

# Forecast features: the calendar plus usage from at least one day earlier,
# so the model is a genuine day-ahead forecast. One reading = 15 minutes,
# so 96 readings = 1 day and 672 readings = 1 week.
FORECAST_FEATURES = [
    "hour", "minute", "dayofweek", "month", "is_weekend",
    "lag_96", "lag_192", "lag_672", "prev_day_mean", "prev_day_max",
    "lag_96_win", "lag_672_win", "prev_week_same_slot_mean", "hour_x_dow",
]

# names used in the app and in what the agents read
FEATURE_NAMES = {
    "hour": "hour of the day",
    "minute": "minute of the hour",
    "dayofweek": "day of the week",
    "month": "month",
    "is_weekend": "weekend or weekday",
    "lag_96": "usage at the same time yesterday",
    "lag_192": "usage at the same time 2 days ago",
    "lag_672": "usage at the same time last week",
    "prev_day_mean": "yesterday's average usage",
    "prev_day_max": "yesterday's highest usage",
    "lag_96_win": "yesterday's usage around this time (±30 min)",
    "lag_672_win": "last week's usage around this time (±30 min)",
    "prev_week_same_slot_mean": "average usage at this time over the past 7 days",
    "hour_x_dow": "hour of the week (e.g. Monday 9am)",
}

# The anomaly detector looks at how much energy is used and when.
ANOMALY_FEATURES = ["usage_kwh", "hour_sin", "hour_cos", "is_weekend"]


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_label)."""
    if Path(path).exists():
        df = pd.read_csv(path, encoding="utf-8-sig")  # Kaggle/UCI file starts with a BOM
        df.columns = df.columns.str.strip()
        df = df.rename(columns=RENAME)
        raw_ts = df["timestamp"]
        df["timestamp"] = pd.to_datetime(raw_ts, format="%d/%m/%Y %H:%M", errors="coerce")
        if df["timestamp"].isna().mean() > 0.5:  # different date format in some mirrors
            df["timestamp"] = pd.to_datetime(raw_ts, dayfirst=True, errors="coerce")
        # Each day's 00:00 reading closes that day (it follows 23:45), so it belongs to the next date.
        midnight = (df["timestamp"].dt.hour == 0) & (df["timestamp"].dt.minute == 0)
        df.loc[midnight, "timestamp"] += pd.Timedelta(days=1)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
        df = df.reset_index(drop=True)
        return df, "UCI Steel Industry Energy Consumption (real plant data)"
    return generate_synthetic(), "Synthetic steel-plant data (add the real CSV to data/ to switch)"


def generate_synthetic(seed: int = 42) -> pd.DataFrame:
    """Synthetic 15-minute data for one year, same schema as the UCI dataset."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2018-01-01 00:15", "2019-01-01 00:00", freq="15min")
    hour = ts.hour + ts.minute / 60
    dow = ts.dayofweek.values
    weekend = dow >= 5

    # Production profile: two shifts on weekdays, lunch dip, light Saturdays.
    prod = np.where((hour >= 8) & (hour < 20), 1.0, 0.0)
    prod = prod - np.where((hour >= 12) & (hour < 13), 0.5, 0.0)
    prod = np.where(weekend, np.where((dow == 5) & (hour >= 9) & (hour < 14), 0.45, 0.0), prod)
    seasonal = 1 + 0.12 * np.cos(2 * np.pi * (ts.dayofyear.values - 15) / 365)

    usage = 3.5 + prod * rng.normal(75, 18, len(ts)) * seasonal
    usage += rng.normal(0, 2.5, len(ts))

    # Inject realistic waste: equipment left running overnight / over weekends.
    days = pd.Series(ts.normalize()).unique()
    idle_days = rng.choice(days, size=int(len(days) * 0.06), replace=False)
    idle_mask = np.isin(ts.normalize(), idle_days) & ((hour < 6) | (hour >= 21))
    usage[idle_mask] += rng.normal(22, 4, idle_mask.sum())
    spikes = rng.random(len(ts)) < 0.004
    usage[spikes] += rng.normal(60, 15, spikes.sum())
    usage = np.clip(usage, 0, 160).round(2)

    lag_kvarh = np.clip(usage * rng.normal(0.45, 0.12, len(ts)) + rng.normal(0, 1, len(ts)), 0, None).round(2)
    lag_kvarh[idle_mask] *= 1.8  # idling motors run at poor power factor
    lead_kvarh = np.clip(rng.normal(2, 2, len(ts)), 0, None).round(2)
    lag_pf = np.where(usage > 0, 100 * usage / np.sqrt(usage**2 + lag_kvarh**2 + 1e-9), 100).round(2)
    lead_pf = np.where(lead_kvarh > 0, 100 * usage / np.sqrt(usage**2 + lead_kvarh**2 + 1e-9), 100).round(2)

    load_type = np.select([usage < 20, usage < 60], ["Light_Load", "Medium_Load"], "Maximum_Load")
    return pd.DataFrame({
        "timestamp": ts,
        "usage_kwh": usage,
        "lagging_kvarh": lag_kvarh,
        "leading_kvarh": lead_kvarh,
        "co2_t": (usage * 0.0005).round(4),
        "lagging_pf": lag_pf,
        "leading_pf": lead_pf,
        "nsm": (ts.hour * 3600 + ts.minute * 60).values,
        "week_status": np.where(weekend, "Weekend", "Weekday"),
        "day_of_week": ts.day_name(),
        "load_type": load_type,
    })


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df["timestamp"]
    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute
    df["dayofweek"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["date"] = t.dt.date
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (df["hour"] + df["minute"] / 60) / 24)

    u = df["usage_kwh"]
    df["lag_96"] = u.shift(96)        # same time yesterday
    df["lag_192"] = u.shift(192)      # same time 2 days ago
    df["lag_672"] = u.shift(672)      # same time last week
    df["prev_day_mean"] = u.shift(96).rolling(96).mean()
    df["prev_day_max"] = u.shift(96).rolling(96).max()
    df["lag_96_win"] = u.shift(94).rolling(5).mean()    # yesterday, +/-30 min window
    df["lag_672_win"] = u.shift(670).rolling(5).mean()  # last week, +/-30 min window
    df["prev_week_same_slot_mean"] = sum(u.shift(96 * k) for k in range(1, 8)) / 7
    df["hour_x_dow"] = df["hour"] + 24 * df["dayofweek"]
    return df.dropna(subset=FORECAST_FEATURES).reset_index(drop=True)


def emission_factor(df: pd.DataFrame) -> float:
    """tCO2 per kWh, derived from the data (falls back to 0.0005)."""
    kwh = df["usage_kwh"].sum()
    if "co2_t" in df and kwh > 0 and df["co2_t"].sum() > 0:
        return float(df["co2_t"].sum() / kwh)
    return 0.0005
