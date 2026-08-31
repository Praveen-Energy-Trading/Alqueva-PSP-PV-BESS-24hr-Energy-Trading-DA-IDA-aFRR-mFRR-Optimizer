"""ML Forecasting - graphs-first view of every machine-learning model this
pipeline actually uses. Every number is read straight from the
*_selected_model.json files each forecaster writes after its own real
walk-forward bake-off (see da_price_forecaster.py::_auto_select_model and
friends) - nothing here is illustrative or recomputed."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

import data
import theme

st.title("🤖 ML Forecasting")

st.markdown(
    '<style>'
    'div[data-testid="stSegmentedControl"] div[role="radiogroup"] { flex-wrap: wrap; row-gap: 4px; }'
    '</style>',
    unsafe_allow_html=True,
)


@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()

    ml = data.load_ml_models_overview()
    rows = ml["rows"]
    if not rows:
        st.info("No trained models found yet - the forecasters write "
                "`*_selected_model.json` the first time they run.")
        return

    # -------------------------------------------------------------------
    # Headline numbers only.
    # -------------------------------------------------------------------
    total_rows = sum(r["n_training_rows"] or 0 for r in rows)
    win_counts = ml["win_counts"]
    top_model = max(win_counts, key=win_counts.get) if win_counts else " - "

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecasting targets", ml["n_targets"])
    c2.metric("Candidate algorithms", 4)
    c3.metric("Training rows (combined)", f"{total_rows:,}")
    with c4:
        st.markdown(
            f'<div style="padding-top:2px;">'
            f'<div style="font-size:14px; color:{theme.INK_SECONDARY};">Most-often-winning model</div>'
            f'<div style="font-size:26px; font-weight:500; color:{theme.INK_PRIMARY}; margin-top:2px;">{top_model}</div>'
            f'<div style="font-size:12px; color:{theme.INK_MUTED};">'
            f'wins {win_counts.get(top_model, 0)} of {ml["n_targets"]} bake-offs</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------------
    # One button per forecasting target -- pick one, see its bake-off chart.
    # -------------------------------------------------------------------
    labels = [r["target"] for r in rows]
    state_key = "ml_bakeoff_target"
    if state_key not in st.session_state or st.session_state[state_key] not in labels:
        st.session_state[state_key] = labels[0]
    selected_label = st.segmented_control(
        "Forecasting target", labels, key=state_key, label_visibility="collapsed",
    ) or st.session_state[state_key]
    sel = next(r for r in rows if r["target"] == selected_label)

    # Metric selector: which of the real walk-forward-CV metrics to chart.
    # MAE always exists (the original bake-off metric); RMSE/MAPE/DirAcc
    # were added this session for DA/IDA1/IDA2/IDA3 only -- offered only
    # when actually present for the SELECTED target, never a blank chart.
    metric_options = {"MAE": ("cv_mae", sel["unit"])}
    if sel.get("cv_rmse"):
        metric_options["RMSE"] = ("cv_rmse", sel["unit"])
    if sel.get("cv_mape"):
        metric_options["MAPE"] = ("cv_mape", "%")
    if sel.get("cv_directional_accuracy"):
        metric_options["Directional accuracy"] = ("cv_directional_accuracy", "fraction correct")

    # Keyed per-target (not a single fixed key): each target's available
    # metric set differs (MAE-only vs MAE+RMSE+MAPE+DirAcc), and reusing one
    # widget key across different `options` lists causes Streamlit to reset
    # the stored value to None on target switch -- confirmed by hitting a
    # real KeyError this way when verifying against the live dashboard.
    metric_state_key = f"ml_bakeoff_metric_{selected_label}"
    metric_labels = list(metric_options.keys())
    if metric_state_key not in st.session_state or st.session_state[metric_state_key] not in metric_labels:
        st.session_state[metric_state_key] = metric_labels[0]
    chosen_metric = st.segmented_control(
        "Metric", metric_labels, key=metric_state_key, label_visibility="collapsed",
    )
    if chosen_metric not in metric_options:
        chosen_metric = metric_labels[0]
    metric_key, metric_unit = metric_options[chosen_metric]

    model_names = list(sel[metric_key].keys())
    metric_vals = [sel[metric_key][m] for m in model_names]
    colors = [theme.STATUS_GOOD if m == sel["selected"] else theme.INK_MUTED for m in model_names]
    fig = go.Figure(go.Bar(x=model_names, y=metric_vals, marker_color=colors,
                            text=[f"{v:,.2f}" for v in metric_vals], textposition="outside"))
    theme.style_fig(fig, height=380, yaxis_title=f"{chosen_metric} ({metric_unit})", legend=False)
    st.plotly_chart(fig, width="stretch")
    rows_txt = f"{sel['n_training_rows']:,} rows" if sel["n_training_rows"] else "n/a"
    st.caption(f"Green = winner **{sel['selected']}**, live in production · {rows_txt} trained")

    # Real Wilcoxon paired significance test between the top-2 candidates
    # by MAE (added this session) -- reports honestly when the "winner"
    # isn't actually distinguishable from the runner-up, not just the
    # lowest number.
    sig = sel.get("significance_top2")
    if sig and sig.get("best") and sig.get("runner_up"):
        p = sig.get("p_value")
        if p is not None and p == p:  # not NaN
            verdict = "statistically significant" if sig.get("significant_at_0.05") else "NOT statistically significant"
            st.caption(f"Wilcoxon signed-rank test, {sig['best']} vs {sig['runner_up']}: "
                       f"p = {p:.3f} — the gap is **{verdict}** at the 5% level.")


_render()
