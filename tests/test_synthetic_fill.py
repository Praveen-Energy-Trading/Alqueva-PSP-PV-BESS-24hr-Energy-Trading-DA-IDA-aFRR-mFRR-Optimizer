"""
test_synthetic_fill.py — artificial outcomes for locked-window days with none (a demonstration choice): confined, labelled, never
overwriting real data, and visible in the preflight and in the locked-test report.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from common_layer.configuration.forecast_formulation import load_forecast_formulation
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.model_development import locked_test as lt, synthetic
from common_layer.forecast_evaluation.tuning import load_tuning_config

from tests.test_forecast_evaluation import PERIODS, synthetic_panel
from tests.test_locked_test import cfg_locked, decision

FORM = load_forecast_formulation()
TCFG = load_tuning_config()
LOCK = (dt.date(2022, 9, 1), dt.date(2022, 10, 20))
FILL_CFG = {"targets": {"ida1_price": {"method": "base_price"}}}
GAP_DAYS = [dt.date(2022, 9, 5), dt.date(2022, 9, 6), dt.date(2022, 9, 20)]      # no IDA1 outcome at all
PARTIAL = dt.date(2022, 9, 10)                                                    # some hours missing: not a gap day
DEV_GAP = dt.date(2022, 7, 10)                                                    # a gap outside the locked window


def cfg():
    c = cfg_locked(LOCK)
    c["locked_test"] = {"start": LOCK[0], "end": LOCK[1], "data_digest": None}
    return c


def gappy_panel():
    p = synthetic_panel(300)
    local = p.index.tz_convert("Europe/Madrid")
    day = np.array([d for d in local.date])
    for d in GAP_DAYS + [DEV_GAP]:
        p.loc[day == d, "ida1_pt"] = np.nan
    p.loc[(day == PARTIAL) & (local.hour < 6), "ida1_pt"] = np.nan
    return p


def test_the_real_configuration_fills_only_idA1_and_only_with_a_known_method(tmp_path):
    c = synthetic.load_fill_config()
    assert list(c["targets"]) == ["ida1_price"] and c["targets"]["ida1_price"]["method"] == "base_price"
    f = tmp_path / "f.yaml"
    f.write_text("targets: {ida1_price: {method: magic}}")
    with pytest.raises(ValueError):
        synthetic.load_fill_config(str(f))


def test_only_locked_window_days_with_no_outcome_are_filled_and_with_the_base_price():
    panel = gappy_panel()
    out, info = synthetic.fill_locked(panel, FORM, "ida1_price", cfg(), FILL_CFG)
    assert info["method"] == "base_price" and info["days"] == [str(d) for d in GAP_DAYS]
    local = panel.index.tz_convert("Europe/Madrid")
    day = np.array([d for d in local.date])
    for d in GAP_DAYS:
        m = day == d
        np.testing.assert_array_equal(out.loc[m, "ida1_pt"].to_numpy(), panel.loc[m, "da_pt"].to_numpy())          # spread of zero
    changed = ~out["ida1_pt"].eq(panel["ida1_pt"]) & ~(out["ida1_pt"].isna() & panel["ida1_pt"].isna())
    assert changed.sum() == 24 * len(GAP_DAYS) and set(day[changed.to_numpy()]) == set(GAP_DAYS)
    assert out.loc[day == DEV_GAP, "ida1_pt"].isna().all()                                                         # development is never filled
    assert out.loc[(day == PARTIAL) & (local.hour < 6), "ida1_pt"].isna().all()                                    # a partly real day is left alone
    real = panel["ida1_pt"].notna()
    np.testing.assert_array_equal(out.loc[real, "ida1_pt"].to_numpy(), panel.loc[real, "ida1_pt"].to_numpy())      # real values never change
    assert panel.equals(gappy_panel())                                                                              # the input is not modified


def test_targets_that_are_not_listed_are_left_exactly_as_they_are():
    panel = gappy_panel()
    out, info = synthetic.fill_locked(panel, FORM, "ida2_price", cfg(), FILL_CFG)
    assert out is panel and info["method"] is None and info["days"] == []
    out, info = synthetic.fill_locked(panel, FORM, "da_price", cfg(), {"targets": {}})
    assert out is panel and info["days"] == []


def test_a_level_target_cannot_use_the_base_price_method():
    with pytest.raises(ValueError, match="spread target"):
        synthetic.fill_locked(gappy_panel(), FORM, "da_price", cfg(), {"targets": {"da_price": {"method": "base_price"}}})


def test_real_only_drops_exactly_the_filled_days():
    pred = pd.DataFrame({"local_date": [d.toordinal() for d in GAP_DAYS + [dt.date(2022, 9, 7)]], "y_pred": 1.0})
    info = {"days": [str(d) for d in GAP_DAYS]}
    assert list(synthetic.real_only(pred, info)["local_date"]) == [dt.date(2022, 9, 7).toordinal()]
    assert synthetic.real_only(pred, None) is pred and synthetic.real_only(pred, {"days": []}) is pred


def _dec():
    d = decision({"type": "single", "model": "lightgbm"})
    d["target"] = "ida1_price"
    return d


def test_the_fill_closes_the_outcome_gap_in_the_preflight_and_is_declared(tmp_path):
    (tmp_path / "config").mkdir()
    root = str(tmp_path)
    panel, c = gappy_panel(), cfg()
    filled, info = synthetic.fill_locked(panel, FORM, "ida1_price", c, FILL_CFG)
    kw = dict(log_root=root, lock_root=root, code_revision="abc", min_truth_share=0.99)
    raw = lt.preflight(_dec(), panel, FORM, c, **kw)
    assert "outcomes_present" in {x["check"] for x in raw["checks"] if not x["ok"]}                       # 47 of 50 days: below 99%
    assert "synthetic_fill_declared" not in {x["check"] for x in raw["checks"]}
    digest_raw = lt.truth_digest(AsOfView(panel, FORM, "ida1_price"), c)[0]
    digest_filled = lt.truth_digest(AsOfView(filled, FORM, "ida1_price"), c)[0]
    assert digest_raw != digest_filled                                                                  # the freeze would see the fill
    frozen_panel = filled
    from common_layer.forecast_evaluation import guards
    guards.write_lock("ida1_price", digest_filled, "w", root)
    ok = lt.preflight(_dec(), frozen_panel, FORM, c, synthetic=info, **kw)
    names = {x["check"]: x for x in ok["checks"]}
    assert names["outcomes_present"]["ok"] and names["synthetic_fill_declared"]["ok"] and "3 locked day(s)" in names["synthetic_fill_declared"]["detail"]
    assert "not real prices" in names["synthetic_fill_declared"]["detail"]


def test_the_locked_report_marks_the_artificial_days_and_gives_real_day_scores(tmp_path):
    (tmp_path / "config").mkdir()
    root = str(tmp_path)
    panel, c = gappy_panel(), cfg()
    filled, info = synthetic.fill_locked(panel, FORM, "ida1_price", c, FILL_CFG)
    rep = lt.run_final(_dec(), filled, FORM, c, PERIODS, TCFG, None, root, True, info)
    n_all, n_real = rep["locked_scores"]["n"], rep["real_only"]["champion"]["n"]
    assert n_all - n_real == 24 * len(GAP_DAYS) and rep["real_only"]["reference"]["n"] == n_real
    assert rep["synthetic_fill"]["days"] == info["days"] and np.isfinite(rep["real_only"]["loss"])
    md = lt.render(rep, _dec())
    assert "3 of the locked days have artificial outcomes, not real prices" in md and "On real days only" in md
    assert "2022-09-05" in md and "2022-09-20" in md
    assert json.loads(json.dumps(lt.to_json(rep)))["synthetic_fill"]["method"] == "base_price"


def test_without_a_fill_the_report_has_no_synthetic_section(tmp_path):
    root = str(tmp_path)
    (tmp_path / "config").mkdir()
    panel, c = synthetic_panel(300), cfg()
    rep = lt.run_final(_dec(), panel, FORM, c, PERIODS, TCFG, None, root, True, None)
    assert "synthetic_fill" not in rep and "real_only" not in rep and "artificial" not in lt.render(rep, _dec())
