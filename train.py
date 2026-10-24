"""Train the forecaster and anomaly detector, print evaluation, save artifacts.
"""
from src.data import load_data
from src.ml import MODEL_PATH, save, train


def main() -> None:
    raw, source = load_data()
    print(f"Data source: {source}  ({len(raw):,} rows)")
    art = train(raw, source)
    print(f"Train/test split at {art.split_date} (time-based, last 2 months held out)\n")
    print(f"{'Model':35s} {'MAE':>8s} {'RMSE':>8s} {'R2':>7s}")
    for name, m in art.metrics.items():
        if isinstance(m, dict):
            print(f"{name:35s} {m['MAE']:8.2f} {m['RMSE']:8.2f} {m['R2']:7.3f}")
    print("\nTop features:", ", ".join(list(art.feature_importance)[:5]))
    save(art)
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
