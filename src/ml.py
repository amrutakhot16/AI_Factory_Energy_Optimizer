"""scikit-learn models and the deterministic analysis the agents reason over.
Design principle: all numbers (forecasts, waste, savings) are computed here,
in code. The LLM agents only interpret and act on these numbers.
"""
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import ANOMALY_FEATURES, FEATURE_NAMES, FORECAST_FEATURES, add_features, emission_factor

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "artifacts.joblib"
TEST_DAYS = 61  # last ~2 months held out (time-based split, no leakage)


@dataclass
class Tariff:
    peak_price: float = 0.18      # $/kWh, weekday peak window
    offpeak_price: float = 0.09   # $/kWh, all other times
    peak_start: int = 10
    peak_end: int = 18

    def price(self, df: pd.DataFrame) -> np.ndarray:
        peak = (df["is_weekend"] == 0) & (df["hour"] >= self.peak_start) & (df["hour"] < self.peak_end)
        return np.where(peak, self.peak_price, self.offpeak_price)

    def is_peak(self, df: pd.DataFrame) -> pd.Series:
        return (df["is_weekend"] == 0) & (df["hour"] >= self.peak_start) & (df["hour"] < self.peak_end)


@dataclass
class Artifacts:
    forecaster: RandomForestRegressor
    anomaly_model: IsolationForest
    metrics: dict
    feature_importance: dict
    idle_baseline_kwh: float
    emission_factor: float
    split_date: str
    data_source: str
    feature_frame: pd.DataFrame = field(repr=False)


def _daily_mean(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return round(float(frame.groupby("date")["usage_kwh"].sum().mean()))


def _scores(y, p) -> dict:
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
    }


def train(raw: pd.DataFrame, data_source: str) -> Artifacts:
    df = add_features(raw)
    split = df["timestamp"].max() - pd.Timedelta(days=TEST_DAYS)
    tr, te = df[df["timestamp"] <= split], df[df["timestamp"] > split]

    forecaster = RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_leaf=5, n_jobs=-1, random_state=42
    ).fit(tr[FORECAST_FEATURES], tr["usage_kwh"])
    pred = forecaster.predict(te[FORECAST_FEATURES])

    metrics = {
        "Random Forest (day-ahead)": _scores(te["usage_kwh"], pred),
        "Baseline: same time yesterday": _scores(te["usage_kwh"], te["lag_96"]),
        "Baseline: same time last week": _scores(te["usage_kwh"], te["lag_672"]),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
    }
    importance = dict(sorted(
        zip(FORECAST_FEATURES, forecaster.feature_importances_.round(4).tolist()),
        key=lambda kv: -kv[1],
    ))

    anomaly_model = IsolationForest(
        n_estimators=200, contamination=0.02, random_state=42
    ).fit(tr[ANOMALY_FEATURES])

    off_hours = (tr["hour"] < 6) | (tr["hour"] >= 22) | (tr["is_weekend"] == 1)
    idle_baseline = float(tr.loc[off_hours, "usage_kwh"].quantile(0.25))

    df["forecast_kwh"] = forecaster.predict(df[FORECAST_FEATURES])
    return Artifacts(
        forecaster=forecaster,
        anomaly_model=anomaly_model,
        metrics=metrics,
        feature_importance=importance,
        idle_baseline_kwh=idle_baseline,
        emission_factor=emission_factor(raw),
        split_date=str(split.date()),
        data_source=data_source,
        feature_frame=df,
    )


def save(art: Artifacts, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path)


def load(path: Path = MODEL_PATH) -> Artifacts | None:
    return joblib.load(path) if Path(path).exists() else None


# names for the three savings opportunities.
OPP_CLOSED = "Switch off machines when the plant is closed"
OPP_SPIKES = "Investigate unusual spikes"
OPP_SHIFT = "Run heavy jobs in cheaper hours"


def analyze(art: Artifacts, start, end, tariff: Tariff, idle_recoverable: float = 0.5,
            peak_shift_share: float = 0.15) -> dict:
    """Run the models over a period and compute the savings opportunities.

    Every number the agents see is calculated here, in code.
    """
    df = art.feature_frame
    p = df[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)].copy()
    if p.empty:
        raise ValueError("No data in the selected period.")

    ef = art.emission_factor  # tonnes of CO2 per kWh
    days = max((p["timestamp"].max() - p["timestamp"].min()).total_seconds() / 86400, 1)
    annualize = 365 / days    # turns "this period" into "per year"

    # Price, cost and flags for every 15-minute reading
    p["price"] = tariff.price(p)
    p["cost"] = p["usage_kwh"] * p["price"]
    p["is_peak"] = tariff.is_peak(p)
    p["closed"] = (p["hour"] < 6) | (p["hour"] >= 22) | (p["is_weekend"] == 1)
    p["anomaly"] = art.anomaly_model.predict(p[ANOMALY_FEATURES]) == -1
    p["excess_kwh"] = np.where(p["anomaly"], (p["usage_kwh"] - p["forecast_kwh"]).clip(lower=0), 0.0)

    total_kwh = float(p["usage_kwh"].sum())
    total_cost = float(p["cost"].sum())

    # Each reading counts toward at most one opportunity (no double counting):
    # unusual spikes belong only to opportunity 2; opportunities 1 and 3 use normal readings.
    normal = ~p["anomaly"]

    # 1. Plant closed but usage above its normal "everything off" level
    idle_threshold = art.idle_baseline_kwh * 1.5
    closed_rows = p[p["closed"] & normal]
    idle_excess = (closed_rows["usage_kwh"] - idle_threshold).clip(lower=0)
    idle_kwh = float(idle_excess.sum() * idle_recoverable)
    idle_cost = float((idle_excess * closed_rows["price"]).sum() * idle_recoverable)

    # 2. Unusual spikes: energy above what the forecast expected
    spike_kwh = float(p["excess_kwh"].sum())
    spike_cost = float((p["excess_kwh"] * p["price"]).sum())

    # 3. Move part of the heaviest work from expensive hours to cheap hours
    heavy_peak = p[p["is_peak"] & (p["load_type"] == "Maximum_Load") & normal]
    shift_kwh = float(heavy_peak["usage_kwh"].sum() * peak_shift_share)
    shift_cost = shift_kwh * (tariff.peak_price - tariff.offpeak_price)

    def opp(name, kwh, cost, how, co2=True):
        return {
            "opportunity": name,
            "annual_kwh": round(kwh * annualize),
            "annual_cost_usd": round(cost * annualize),
            "annual_co2_t": round(kwh * ef * annualize, 2) if co2 else 0.0,
            "how_calculated": how,
        }

    opportunities = [
        opp(OPP_CLOSED, idle_kwh, idle_cost,
            f"Usage above {idle_threshold:.1f} kWh per 15 min at night or on weekends; "
            f"assumes {int(idle_recoverable * 100)}% of it can be switched off"),
        opp(OPP_SPIKES, spike_kwh, spike_cost,
            f"{int(p['anomaly'].sum())} unusual readings; energy above the forecast"),
        opp(OPP_SHIFT, shift_kwh, shift_cost,
            f"Moves {int(peak_shift_share * 100)}% of the heaviest work out of peak-price hours. "
            "Saves money, not energy, so no CO2 saving", co2=False),
    ]

    # Group unusual readings that happen back-to-back into events
    a = p[p["anomaly"]].copy()
    events = []
    if not a.empty:
        a["gap"] = (a["timestamp"].diff() > pd.Timedelta(minutes=15)).cumsum()
        for _, g in a.groupby("gap"):
            events.append({
                "start": g["timestamp"].min().strftime("%a %Y-%m-%d %H:%M"),
                "end": g["timestamp"].max().strftime("%H:%M"),
                "actual_kwh": round(float(g["usage_kwh"].sum()), 1),
                "expected_kwh": round(float(g["forecast_kwh"].sum()), 1),
                "extra_kwh": round(float(g["excess_kwh"].sum()), 1),
                "when": "plant closed" if g["closed"].mean() > 0.5 else "working hours",
            })
        events.sort(key=lambda e: -e["extra_kwh"])

    hourly = p.groupby("hour")["usage_kwh"].mean()
    summary = {
        "period": f"{start} to {end}",
        "days": round(days),
        "total_kwh": round(total_kwh),
        "total_cost_usd": round(total_cost),
        "total_co2_t": round(total_kwh * ef, 2),
        "share_of_energy_in_peak_price_hours": round(float(p.loc[p["is_peak"], "usage_kwh"].sum() / total_kwh), 3),
        "share_of_energy_while_plant_closed": round(float(p.loc[p["closed"], "usage_kwh"].sum() / total_kwh), 3),
        "busiest_hour": int(hourly.idxmax()),
        "quietest_hour": int(hourly.idxmin()),
        "weekday_avg_daily_kwh": _daily_mean(p[p["is_weekend"] == 0]),
        "weekend_avg_daily_kwh": _daily_mean(p[p["is_weekend"] == 1]),
        "unusual_readings": int(p["anomaly"].sum()),
        "unusual_events": len(events),
        "prices": f"${tariff.peak_price}/kWh on weekdays {tariff.peak_start}:00-{tariff.peak_end}:00, "
                  f"${tariff.offpeak_price}/kWh otherwise",
    }
    model = art.metrics["Random Forest (day-ahead)"]
    baseline = art.metrics["Baseline: same time yesterday"]
    forecast = {
        "typical_error_kwh_per_15min": round(model["MAE"], 1),
        "baseline_error_same_time_yesterday": round(baseline["MAE"], 1),
        "error_reduction_vs_baseline": round(1 - model["MAE"] / baseline["MAE"], 2),
        "most_useful_inputs": [FEATURE_NAMES.get(k, k) for k in list(art.feature_importance)[:3]],
    }
    return {
        "frame": p,
        "summary": summary,
        "forecast": forecast,
        "opportunities": opportunities,
        "events": events,
        "hourly_profile": hourly.round(1).to_dict(),
        "total_annual_savings_usd": round(sum(o["annual_cost_usd"] for o in opportunities)),
        "total_annual_co2_t": round(sum(o["annual_co2_t"] for o in opportunities), 2),
    }
