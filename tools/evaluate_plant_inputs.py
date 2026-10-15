"""evaluate_plant_inputs.py - the held-out evaluation of the three plant-input forecasts (solar irradiance, ambient temperature, natural inflow)
for the dashboard page "Plant inputs".

The protocol is the one of the training scripts next to the forecasters: chronological split, the last TEST_MONTHS are the test, the model was selected
by walk-forward cross-validation on the development part only (the choice is read from the *_selected_model.json files, not repeated here), and
the test is read once. What this adds is the comparison the page needs: all five algorithms and three simple forecasts (same hour yesterday, same
hour a week earlier, the average of the month) scored on the same test hours, and the test-year series for the charts. Only measured days are scored.

    python tools/evaluate_plant_inputs.py            writes model_development/plant_inputs/{ghi,temperature,inflow}.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "phase_2_da_day_ahead_bidding" / "da_price_pv_inflow_forecasting"
for p in (FOLDER, ROOT):
    sys.path.insert(0, str(p))

from ml_train_val_test_common import MODEL_NAMES, fit_selected, real_rows          # noqa: E402

TEST_MONTHS = 12
OUT = ROOT / "model_development" / "plant_inputs"


def _scores(y, pred) -> dict:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err))}


def _split(data: pd.DataFrame, time_col: str):
    last = data[time_col].max()
    start = last - pd.DateOffset(months=TEST_MONTHS)
    return data[data[time_col] < start], data[data[time_col] >= start]


def _evaluate(key, name, unit, time_col, target, fcols, data, naive: dict, climatology, selected, resolution, features_text) -> dict:
    data = data.sort_values(time_col).reset_index(drop=True)
    dev, test = _split(data, time_col)
    y = test[target].to_numpy(dtype=float)
    models = {}
    preds = {}
    for m in MODEL_NAMES:
        model = fit_selected(m, dev[fcols], dev[target].to_numpy(dtype=float), fcols)
        preds[m] = np.asarray(model.predict(test[fcols]), dtype=float)
        models[m] = _scores(y, preds[m])
    base = {}
    for label, col in naive.items():
        ok = test[col].notna().to_numpy()
        base[label] = _scores(y[ok], test[col].to_numpy(dtype=float)[ok])
        preds[label] = test[col].to_numpy(dtype=float)
    clim = climatology(dev, test)
    base["Average of the month"] = _scores(y, clim)
    preds["Average of the month"] = clim
    chosen = preds[selected]
    persistence = base["Same time yesterday"]["mae"]
    # test-year series for the charts: daily means (actual, chosen model, yesterday) and, for the hourly series, the last 7 days hour by hour
    frame = pd.DataFrame({"t": test[time_col].to_numpy(), "actual": y, "model": chosen, "yesterday": preds["Same time yesterday"]})
    daily = frame.assign(day=frame["t"].dt.normalize()).groupby("day")[["actual", "model", "yesterday"]].mean().reset_index()
    out = {
        "key": key, "name": name, "unit": unit, "resolution": resolution, "features": features_text, "selected": selected,
        "dev_range": [str(dev[time_col].min())[:10], str(dev[time_col].max())[:10]], "test_range": [str(test[time_col].min())[:10], str(test[time_col].max())[:10]],
        "n_dev": int(len(dev)), "n_test": int(len(test)),
        "models": models, "baselines": base,
        "skill_vs_yesterday": float(1 - models[selected]["mae"] / persistence) if persistence > 0 else None,
        "daily": {"day": [str(d)[:10] for d in daily["day"]], "actual": daily["actual"].round(3).tolist(), "model": daily["model"].round(3).tolist(),
                  "yesterday": daily["yesterday"].round(3).tolist()},
    }
    if resolution == "hourly":
        tail = frame[frame["t"] >= frame["t"].max() - pd.Timedelta(days=7)]
        out["week"] = {"t": [str(t)[:16] for t in tail["t"]], "actual": tail["actual"].round(3).tolist(), "model": tail["model"].round(3).tolist(),
                       "yesterday": tail["yesterday"].round(3).tolist()}
    return out


def _pv(target_col: str, key: str, name: str, unit: str, selected: str) -> dict:
    from pv_power_forecaster import _build_features, _feature_cols, _load_history
    hist = _load_history()
    full = _build_features(hist)
    full = full[full["datetime"].isin(hist.loc[real_rows(hist), "datetime"])]          # only measured sensor hours are scored
    fcols = _feature_cols(target_col)
    lag24, lag168 = f"lag_24h_{target_col}", f"lag_168h_{target_col}"
    data = full.dropna(subset=fcols + [target_col, lag24]).copy()
    data["hour"] = data["datetime"].dt.hour
    data["month"] = data["datetime"].dt.month

    def climatology(dev, test):
        table = dev.groupby(["month", "hour"])[target_col].mean()
        return np.array([table.get((m, h), np.nan) for m, h in zip(test["month"], test["hour"])], dtype=float)

    return _evaluate(key, name, unit, "datetime", target_col, fcols, data, {"Same time yesterday": lag24, "Same time a week earlier": lag168}, climatology, selected,
                     "hourly", ", ".join(fcols))


def _inflow(selected: str) -> dict:
    from reservoir_inflow_forecaster import _build_features, _feature_cols, _load_history
    hist = _load_history()
    full = _build_features(hist)
    full = full[full["Date"].isin(hist.loc[real_rows(hist), "Date"])]          # only measured days are scored
    fcols = _feature_cols()
    data = full.dropna(subset=fcols + ["inflow_m3h", "lag_1d"]).copy()
    data["month"] = data["Date"].dt.month

    def climatology(dev, test):
        table = dev.groupby("month")["inflow_m3h"].mean()
        return np.array([table.get(m, np.nan) for m in test["month"]], dtype=float)

    return _evaluate("inflow", "Natural inflow", "m³/h", "Date", "inflow_m3h", fcols, data, {"Same time yesterday": "lag_1d", "Same time a week earlier": "lag_7d"},
                     climatology, selected, "daily", ", ".join(fcols))


def _data_summary(history: pd.DataFrame, time_col: str, value_cols: list, unit: str) -> dict:
    """What the history of a series holds: its dates, how many values were measured and how many were filled."""
    real = real_rows(history)
    measured = history[real]
    return {
        "first": str(history[time_col].min())[:10], "last": str(history[time_col].max())[:10], "rows": int(len(history)), "measured": int(real.sum()),
        "filled": int((~real).sum()), "last_measured": str(measured[time_col].max())[:10],
        "sources": {str(k): int(v) for k, v in history["source"].fillna("").astype(str).value_counts().items()} if "source" in history.columns else {},
        "columns": value_cols, "unit": unit,
    }


def summaries() -> dict:
    from pv_power_forecaster import _load_history as pv_history
    from reservoir_inflow_forecaster import _load_history as inflow_history
    pv = pv_history()
    inflow = inflow_history()
    return {"ghi": _data_summary(pv, "datetime", ["GHI"], "W/m²"), "temperature": _data_summary(pv, "datetime", ["T_amb"], "°C"),
            "inflow": _data_summary(inflow, "Date", ["inflow_m3h"], "m³/h")}


def main() -> None:
    if "--data-only" in sys.argv:                                   # refresh only the description of the data, without training anything
        for key, summary in summaries().items():
            path = OUT / f"{key}.json"
            r = json.loads(path.read_text(encoding="utf-8"))
            r["data"] = summary
            path.write_text(json.dumps(r, indent=1), encoding="utf-8")
            print(key, summary["first"], summary["last"], summary["measured"], summary["filled"])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    pv = json.loads((FOLDER / "pv_selected_model.json").read_text(encoding="utf-8"))
    inflow = json.loads((FOLDER / "inflow_selected_model.json").read_text(encoding="utf-8"))
    results = {
        "ghi": _pv("GHI", "ghi", "Solar irradiance", "W/m²", pv["GHI"]["selected"]),
        "temperature": _pv("T_amb", "temperature", "Air temperature", "°C", pv["T_amb"]["selected"]),
        "inflow": _inflow(inflow["selected"]),
    }
    for key, summary in summaries().items():
        results[key]["data"] = summary
    for key, r in results.items():
        (OUT / f"{key}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        best = min(list(r["models"].items()) + list(r["baselines"].items()), key=lambda kv: kv[1]["mae"])
        print(f"{key}: selected {r['selected']}  MAE {r['models'][r['selected']]['mae']:.3f}  best of all: {best[0]} {best[1]['mae']:.3f}  skill vs yesterday {r['skill_vs_yesterday']:.3f}")


if __name__ == "__main__":
    main()
