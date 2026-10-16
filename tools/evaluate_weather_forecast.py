"""evaluate_weather_forecast.py - does a numerical weather forecast improve the irradiance and temperature forecasts of the plant, judged as of the decision time?

The truth is the ERA5 reanalysis of the Alqueva site (tools/fetch_open_meteo.py --reanalysis); the weather forecast is the two-day-old forecast of
Open-Meteo's weather model as it was issued (tools/fetch_open_meteo.py). Both are hourly, in UTC.

AS OF 11:00 on D-1 (about 09:00 UTC) no observation of the last hours before the delivery day exists. For a target hour h of day D the newest usable
observation is therefore the same hour of D-1 when h <= 09:00 UTC and the same hour of D-2 otherwise. Every lag below respects that, for the
baselines and for the algorithms alike: nothing here reads a value the decision maker would not have.

Scored on the last twelve months of the reanalysis (the test), all hours, MAE / RMSE / bias:
  * simple forecasts: the newest same-hour observation, the same hour two days earlier, the weather forecast alone, the average of the month;
  * the five tree algorithms trained on the whole history WITHOUT the forecast;
  * the same algorithms trained on only the period for which forecasts exist, without the forecast (so a gain is not just a shorter history);
  * the same algorithms trained on that period WITH the forecast as an input.

    python tools/evaluate_weather_forecast.py          writes model_development/plant_inputs/{ghi,temperature}_weather.json
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

from ml_train_val_test_common import MODEL_NAMES, fit_selected          # noqa: E402
from pv_power_forecaster import _clearsky_ghi                           # noqa: E402

DIR = ROOT / "model_development" / "plant_inputs"
TEST_MONTHS = 12
LAST_OBS_HOUR = 9                    # the newest observation of D-1 at the decision time (UTC hour)
TARGETS = {"ghi": ("ghi", "ghi_fc_d2", "W/m²", "Solar irradiance"), "temperature": ("tamb", "tamb_fc_d2", "°C", "Air temperature")}


def _scores(y, pred) -> dict:
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "bias": float(np.mean(err))}


def _frame() -> pd.DataFrame:
    era = pd.read_csv(DIR / "era5_alqueva.csv.gz", parse_dates=["datetime_utc"])
    fc = pd.read_csv(DIR / "open_meteo_forecasts.csv.gz", parse_dates=["datetime_utc"])
    df = era.merge(fc[["datetime_utc", "ghi_fc_d2", "tamb_fc_d2"]], on="datetime_utc", how="left").sort_values("datetime_utc").reset_index(drop=True)
    df["datetime"] = df["datetime_utc"].dt.tz_localize(None)
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["doy"] = df["datetime"].dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df["hour"] / 24), np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * (df["month"] - 1) / 12), np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["clearsky_ghi"] = df["datetime"].apply(_clearsky_ghi)
    for col in ("ghi", "tamb"):
        s = df[col]
        df[f"{col}_newest"] = np.where(df["hour"] <= LAST_OBS_HOUR, s.shift(24), s.shift(48))      # the newest same-hour observation known at 11:00 on D-1
        df[f"{col}_lag48"] = s.shift(48)
        df[f"{col}_lag168"] = s.shift(168)
        df[f"{col}_mean48"] = s.shift(48).rolling(24).mean()                                        # the mean of the 24 hours ending 48 hours earlier
    df["kt_newest"] = np.where(df["clearsky_ghi"] > 10, df["ghi_newest"] / df["clearsky_ghi"].replace(0, np.nan), 0.0)
    df["nwp_kt"] = np.where(df["clearsky_ghi"] > 10, df["ghi_fc_d2"] / df["clearsky_ghi"].replace(0, np.nan), 0.0)
    return df


def _evaluate(df: pd.DataFrame, target: str, nwp: str, unit: str, name: str) -> dict:
    cal = ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy", "clearsky_ghi"]
    base = cal + [f"{target}_newest", f"{target}_lag48", f"{target}_lag168", f"{target}_mean48"] + (["kt_newest"] if target == "ghi" else [])
    with_fc = base + [nwp] + (["nwp_kt"] if target == "ghi" else [])
    need = with_fc + [target]
    d = df.dropna(subset=need).reset_index(drop=True)
    start = d["datetime"].max() - pd.DateOffset(months=TEST_MONTHS)
    dev_all = df.dropna(subset=base + [target])
    dev_all = dev_all[dev_all["datetime"] < start]
    dev_fc, test = d[d["datetime"] < start], d[d["datetime"] >= start]
    y = test[target].to_numpy(dtype=float)
    out = {"target": name, "unit": unit, "test_range": [str(test["datetime"].min())[:10], str(test["datetime"].max())[:10]], "n_test": int(len(test)),
           "n_dev_full": int(len(dev_all)), "n_dev_forecast": int(len(dev_fc)), "dev_forecast_range": [str(dev_fc["datetime"].min())[:10], str(dev_fc["datetime"].max())[:10]]}
    table = dev_all.groupby(["month", "hour"])[target].mean()
    out["baselines"] = {
        "Newest same-hour observation": _scores(y, test[f"{target}_newest"]),
        "Same hour two days earlier": _scores(y, test[f"{target}_lag48"]),
        "Weather forecast alone": _scores(y, test[nwp]),
        "Average of the month": _scores(y, [table.get((m, h), np.nan) for m, h in zip(test["month"], test["hour"])]),
    }
    frame = pd.DataFrame({"t": test["datetime"].to_numpy(), "actual": y, "model": test[nwp].to_numpy(dtype=float), "yesterday": test[f"{target}_newest"].to_numpy(dtype=float)})
    daily = frame.assign(day=frame["t"].dt.normalize()).groupby("day")[["actual", "model", "yesterday"]].mean().reset_index()
    out["daily"] = {"day": [str(d)[:10] for d in daily["day"]], "actual": daily["actual"].round(3).tolist(), "model": daily["model"].round(3).tolist(),
                    "yesterday": daily["yesterday"].round(3).tolist()}
    tail = frame[frame["t"] >= frame["t"].max() - pd.Timedelta(days=7)]
    out["week"] = {"t": [str(x)[:16] for x in tail["t"]], "actual": tail["actual"].round(3).tolist(), "model": tail["model"].round(3).tolist(),
                   "yesterday": tail["yesterday"].round(3).tolist()}
    era = df
    out["data"] = {"source": "ERA5 reanalysis (Open-Meteo), Alqueva site, hourly UTC", "first": str(era["datetime"].min())[:10], "last": str(era["datetime"].max())[:10],
                   "rows": int(len(era)), "forecast_first": str(df.loc[df[nwp].notna(), "datetime"].min())[:10]}
    groups = {"without_forecast_full_history": (base, dev_all), "without_forecast_same_period": (base, dev_fc), "with_forecast": (with_fc, dev_fc)}
    for key, (cols, train) in groups.items():
        out[key] = {}
        for m in MODEL_NAMES:
            model = fit_selected(m, train[cols], train[target].to_numpy(dtype=float), cols)
            out[key][m] = _scores(y, np.asarray(model.predict(test[cols]), dtype=float))
        print(f"{target} {key}: " + ", ".join(f"{m} {v['mae']:.3f}" for m, v in out[key].items()), flush=True)
    return out


def main() -> None:
    df = _frame()
    for key, (target, nwp, unit, name) in TARGETS.items():
        r = _evaluate(df, target, nwp, unit, name)
        (DIR / f"{key}_weather.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(key, "baselines:", {k: round(v["mae"], 3) for k, v in r["baselines"].items()}, flush=True)


if __name__ == "__main__":
    main()
