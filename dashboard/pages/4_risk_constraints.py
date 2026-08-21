"""Risk & Constraints — how close to physical/regulatory limits the plant
ran, and a live check of the invariants the test suite defines as
"correct" (tests/test_bug_regressions.py BUG-1..8 and friends), computed
against real production output instead of only at test time."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import data
import theme

st.title("⚠️ Risk & Constraints")

@st.fragment(run_every=theme.auto_refresh_interval())
def _render() -> None:
    theme.inject_scroll_restore()
    selected_date = st.session_state.get("selected_date")
    report_ready = st.session_state.get("report_ready", False)

    if not selected_date:
        st.warning("No runs found yet — visit Run & Monitor to start one.")
        return
    if not report_ready:
        st.info(data.no_report_message(selected_date))
        return

    report = data.load_daily_report(selected_date)
    dispatch, isp, kpis = report["dispatch"], report["isp"], report["kpis"]

    plant_cfg = data.load_plant_config()
    market_cfg = data.load_market_config()
    fcr_mw = plant_cfg["fcr"]["mandatory_headroom_mw"]
    gen_cap_mw = market_cfg["bid_limits"]["max_generation_mw"]
    pump_cap_mw = market_cfg["bid_limits"]["max_pump_mw"]
    gen_cap_fcr_mw = gen_cap_mw - fcr_mw
    pump_cap_fcr_mw = pump_cap_mw - fcr_mw

    # ---------------------------------------------------------------------------
    # Overall status banner -- computed from BOTH check sources below (the
    # exported Constraint Verification flags and the invariant checklist)
    # before anything else renders, so "is today clean" doesn't require
    # scrolling the whole page to piece together from 12+ separate checks.
    # ---------------------------------------------------------------------------

    verify_rows = kpis[kpis["Section"].astype(str).str.contains("Constraint", case=False, na=False)]
    constraint_flags = {
        str(row["Metric"]).strip()[:-3].strip().lower(): str(row["Value"]).strip().upper()
        for _, row in verify_rows.iterrows() if str(row["Metric"]).strip().endswith("OK?")
    }
    n_constraint_fail = sum(1 for v in constraint_flags.values() if v == "NO")

    # (plain-English label for the visual chip, technical label + detail
    # for the tooltip -- a non-technical reader gets "No conflicting
    # commands", an engineer hovering gets the exact invariant and numbers)
    checks: list[tuple[str, bool, str, str]] = []
    if {"ISP", "Market", "Up_MW", "Dn_MW"}.issubset(isp.columns):
        simultaneous = isp[(isp["Up_MW"] > 0) & (isp["Dn_MW"] > 0)]
        checks.append(("No conflicting reserve commands", simultaneous.empty,
                        "No simultaneous up+down activation in the same ISP",
                        f"{len(simultaneous)} ISP(s) with both directions active" if not simultaneous.empty else "Never happened today"))
    if "BESS_SOC_pct" in dispatch.columns:
        soc = dispatch["BESS_SOC_pct"]
        breach = dispatch[(soc < 9.99) | (soc > 95.01)]
        checks.append(("Battery stayed in safe range", breach.empty,
                        "BESS SOC stays within 10-95% bounds",
                        f"{len(breach)} hour(s) outside bounds" if not breach.empty else "10-95% all day"))
    if {"Gen_headroom_MW", "Pump_headroom_MW"}.issubset(dispatch.columns):
        env_breach = dispatch[(dispatch["Gen_headroom_MW"] < -0.01) | (dispatch["Pump_headroom_MW"] < -0.01)]
        checks.append(("Plant stayed within its limits", env_breach.empty,
                        "Physical + FCR envelope never breached",
                        f"{len(env_breach)} hour(s) breached" if not env_breach.empty else "No breach"))
    if "Mass_balance_error_hm3" in dispatch.columns:
        mb_breach = dispatch[dispatch["Mass_balance_error_hm3"].abs() > 1e-4]
        checks.append(("Water accounting checks out", mb_breach.empty,
                        "Reservoir mass balance error within tolerance (1e-4 hm3)",
                        f"max error {dispatch['Mass_balance_error_hm3'].abs().max():.2e} hm3" if not mb_breach.empty else "Within tolerance"))
    if "Energy_balance_check_MW" in dispatch.columns:
        eb_breach = dispatch[dispatch["Energy_balance_check_MW"].abs() > 0.1]
        checks.append(("Power accounting checks out", eb_breach.empty,
                        "Hourly energy balance within tolerance (0.1 MW)",
                        f"max error {dispatch['Energy_balance_check_MW'].abs().max():.2f} MW" if not eb_breach.empty else "Within tolerance"))
    n_invariant_fail = sum(1 for _, ok, _, _ in checks if not ok)

    total_checks = len(constraint_flags) + len(checks)
    total_fail = n_constraint_fail + n_invariant_fail
    if total_checks:
        if total_fail == 0:
            st.success(f"🟢 All {total_checks} risk checks passed for **{selected_date}** "
                       f"({len(constraint_flags)} constraint + {len(checks)} invariant)")
        else:
            st.error(f"🔴 {total_fail} of {total_checks} risk checks FAILED for **{selected_date}** "
                     f"— see Constraint verification / Production health checklist below")

    # One-line summary instead of a jargon-heavy paragraph -- the bar below
    # shows the actual split visually, so the text only needs to anchor it.
    st.caption(f"Plant can generate up to {gen_cap_mw:.0f} MW or pump up to {pump_cap_mw:.0f} MW. "
               f"{fcr_mw:.0f} MW of that is always held back for grid safety (FCR) and never traded.")

    envelope_fig = go.Figure()
    for label, cap, tradable, color in [
        ("Generation", gen_cap_mw, gen_cap_fcr_mw, theme.COLOR_GEN),
        ("Pumping",    pump_cap_mw, pump_cap_fcr_mw, theme.COLOR_PUMP),
    ]:
        envelope_fig.add_trace(go.Bar(y=[label], x=[tradable], orientation="h", name="Tradable",
                                       marker_color=color, showlegend=label == "Generation",
                                       text=f"{tradable:.0f} MW tradable", textposition="inside"))
        envelope_fig.add_trace(go.Bar(y=[label], x=[cap - tradable], orientation="h", name="Reserved (FCR)",
                                       marker_color=theme.STATUS_NEUTRAL, showlegend=label == "Generation",
                                       text=f"{cap - tradable:.0f} MW reserved", textposition="inside"))
    theme.style_fig(envelope_fig, height=160, barmode="stack", xaxis_title="MW")
    envelope_fig.update_layout(legend=dict(orientation="h", y=1.25))
    st.plotly_chart(envelope_fig, width="stretch")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Utilization % — computed here, not stored anywhere in the pipeline output
    # ---------------------------------------------------------------------------

    st.subheader("Envelope utilization")
    st.caption("How close the plant ran to its tradable limit each hour.")
    net = dispatch["Plant_net_final_MW"]
    gen_util_pct = (net.clip(lower=0) / gen_cap_fcr_mw * 100)
    pump_util_pct = ((-net).clip(lower=0) / pump_cap_fcr_mw * 100)

    # Two upright panels (both 0-100%, nothing negative) instead of one
    # chart with pump values flipped below zero -- a mirrored "-90%" reads
    # as a below-zero number to a non-technical reader, when it's really
    # just "90% of the pump limit", not a negative quantity.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                         subplot_titles=("Generating", "Pumping"))
    fig.add_trace(go.Bar(x=dispatch["Hour"], y=gen_util_pct, name="Generation %", marker_color=theme.COLOR_GEN), row=1, col=1)
    fig.add_trace(go.Bar(x=dispatch["Hour"], y=pump_util_pct, name="Pump %", marker_color=theme.COLOR_PUMP), row=2, col=1)
    fig.add_hline(y=100, line_dash="dot", line_color=theme.STATUS_CRITICAL, row=1, col=1)
    fig.add_hline(y=100, line_dash="dot", line_color=theme.STATUS_CRITICAL, row=2, col=1)
    theme.style_fig(fig, height=380, legend=False)
    fig.update_layout(margin=dict(t=32))  # subplot title ("Generating") otherwise sits under the modebar icons
    fig.update_yaxes(title_text="% of limit", range=[0, 110], row=1, col=1)
    fig.update_yaxes(title_text="% of limit", range=[0, 110], row=2, col=1)
    st.plotly_chart(fig, width="stretch")
    peak_gen = gen_util_pct.max()
    peak_pump = pump_util_pct.max()
    # Peak alone doesn't say how often the plant actually ran hot -- one
    # spiky hour reads identically to a whole afternoon near the limit
    # unless the hour-count is shown alongside it.
    hrs_gen_90 = int((gen_util_pct >= 90).sum())
    hrs_pump_90 = int((pump_util_pct >= 90).sum())
    theme.metric_cards([
        ("Peak generation utilization",  f"{peak_gen:.1f} %"),
        ("Peak pump utilization",        f"{peak_pump:.1f} %"),
        ("Hours ≥90% generation limit",  str(hrs_gen_90)),
        ("Hours ≥90% pump limit",        str(hrs_pump_90)),
    ], ncols=4, pre_formatted=True)

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # FCR headroom margin
    # ---------------------------------------------------------------------------

    st.subheader("FCR headroom margin")
    if {"Gen_headroom_MW", "Pump_headroom_MW"}.issubset(dispatch.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["Gen_headroom_MW"], name="Gen headroom MW", line=dict(color=theme.COLOR_GEN)))
        fig.add_trace(go.Scatter(x=dispatch["Hour"], y=dispatch["Pump_headroom_MW"], name="Pump headroom MW", line=dict(color=theme.COLOR_PUMP)))
        fig.add_hline(y=0, line_dash="dot", line_color=theme.STATUS_CRITICAL, annotation_text="breach")
        theme.style_fig(fig, height=320, yaxis_title="MW")
        st.plotly_chart(fig, width="stretch")
        min_gen_hr = dispatch["Gen_headroom_MW"].min()
        min_pump_hr = dispatch["Pump_headroom_MW"].min()
        if min_gen_hr < -0.01 or min_pump_hr < -0.01:
            st.error(f"🔴 Headroom breached — min gen headroom {min_gen_hr:.2f} MW, min pump headroom {min_pump_hr:.2f} MW")
        else:
            st.success(f"🟢 Headroom never breached — min gen {min_gen_hr:.2f} MW, min pump {min_pump_hr:.2f} MW")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Constraint verification panel — promoted from the bottom of Summary_KPIs
    # ---------------------------------------------------------------------------

    st.subheader("Constraint verification (from Summary_KPIs)")
    if verify_rows.empty:
        st.info("No 'Constraint Verification' section found in this report's Summary_KPIs sheet.")
    else:
        # Summary_KPIs exports each check as TWO rows -- "Max energy balance
        # error" (the number) and "Energy balance OK?" (its own YES/NO flag)
        # -- which used to render as two separate cards for one fact. Fold
        # the flag (computed above, for the banner) into the value card it
        # belongs to instead.
        values: list[tuple[str, object, object]] = [
            (str(row["Metric"]).strip(), row["Value"], row.get("Unit"))
            for _, row in verify_rows.iterrows() if not str(row["Metric"]).strip().endswith("OK?")
        ]

        def _match_flag(metric_label: str) -> str | None:
            low = metric_label.lower()
            for key, flag in constraint_flags.items():
                if all(word in low for word in key.split()):
                    return flag
            return None

        ncols = min(4, len(values)) or 1
        cols = st.columns(ncols)
        for i, (metric, val, unit) in enumerate(values):
            flag = _match_flag(metric)
            badge_color = theme.STATUS_GOOD if flag == "YES" else theme.STATUS_CRITICAL if flag == "NO" else None
            badge_text = "PASS" if flag == "YES" else "FAIL" if flag == "NO" else None
            display_val = theme.fmt_metric(val) if isinstance(val, (int, float)) else str(val)
            unit_txt = f" {unit}" if isinstance(unit, str) and unit.strip() and unit.strip().lower() != "nan" else ""
            badge_html = (
                f'<span style="font-size:11px; font-weight:600; padding:2px 8px; border-radius:10px; '
                f'background:{badge_color}22; color:{badge_color};">{badge_text}</span>'
            ) if badge_text else ""
            with cols[i % ncols]:
                st.markdown(
                    f'<div style="background:{theme.SURFACE}; border:1px solid {theme.GRIDLINE}; '
                    f'border-radius:10px; padding:0.7rem 0.9rem; margin-bottom:8px; min-height:84px;">'
                    f'<div style="font-size:12.5px; color:{theme.INK_SECONDARY}; margin-bottom:4px;">{metric}</div>'
                    f'<div style="display:flex; align-items:center; justify-content:space-between;">'
                    f'<div style="font-size:22px; font-weight:500; color:{theme.INK_PRIMARY};">{display_val}{unit_txt}</div>'
                    f'{badge_html}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Invariant health checklist — computed from stored data, not re-derived
    # from the test suite itself. Documents what's checkable vs not.
    # ---------------------------------------------------------------------------

    st.subheader("Production health checklist")

    # Colored chips, not a bullet list of technical sentences -- color
    # alone answers "is this OK", and the plain-English label says what it
    # is; hover any chip for the exact invariant an engineer would want.
    # `checks` was already computed above (for the top banner) -- reused
    # here, not recomputed.
    cols = st.columns(len(checks) or 1)
    for col, (plain_label, ok, technical_label, detail) in zip(cols, checks):
        color = theme.STATUS_GOOD if ok else theme.STATUS_CRITICAL
        with col:
            st.markdown(
                f'<div title="{technical_label} — {detail}" style="background:{color}14; '
                f'border:1px solid {color}55; border-radius:10px; padding:0.7rem 0.6rem; '
                f'text-align:center; min-height:70px;">'
                f'<div style="font-size:20px;">{"🟢" if ok else "🔴"}</div>'
                f'<div style="font-size:12px; color:{theme.INK_PRIMARY}; margin-top:2px;">{plain_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.caption("2 further invariants (per-unit delivery mode, settlement reconciliation) "
               "aren't checkable from the current exports — not silently skipped, just not yet wired up.")


_render()
