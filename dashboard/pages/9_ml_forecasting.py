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
    c2.metric("Candidate algorithms", 3)
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

    # MAE is the only walk-forward-CV metric this pipeline reports.
    chosen_metric = "MAE"
    metric_key, metric_unit = "cv_mae", sel["unit"]

    model_names = list(sel[metric_key].keys())
    metric_vals = [sel[metric_key][m] for m in model_names]
    is_winner = [m == sel["selected"] for m in model_names]
    colors = [theme.STATUS_GOOD if w else theme.rgba(theme.INK_MUTED, 0.55) for w in is_winner]
    line_colors = [theme.STATUS_GOOD if w else theme.GRIDLINE for w in is_winner]

    fig = go.Figure(go.Bar(
        x=model_names, y=metric_vals,
        marker=dict(
            color=colors,
            line=dict(color=line_colors, width=1.5),
            cornerradius=8,
        ),
        text=[f"{v:,.2f}" for v in metric_vals],
        textposition="outside",
        textfont=dict(size=15, color=theme.INK_PRIMARY, family=theme.FONT_FAMILY),
        cliponaxis=False,
        width=0.4,
        hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
    ))
    theme.style_fig(fig, height=340, yaxis_title=f"{chosen_metric} ({metric_unit})", legend=False)
    fig.update_layout(
        bargap=0.5,
        margin=dict(t=28, b=10, l=10, r=10),
    )
    fig.update_xaxes(
        range=[-0.7, len(model_names) - 0.3],
        tickfont=dict(size=13, color=theme.INK_SECONDARY),
    )
    fig.update_yaxes(rangemode="tozero")

    _l, chart_col, _r = st.columns([1, 2, 1])
    with chart_col:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    rows_txt = f"{sel['n_training_rows']:,} rows" if sel["n_training_rows"] else "n/a"
    st.caption(f"🟢 Winner **{sel['selected']}**, live in production · {rows_txt} trained")


_render()
