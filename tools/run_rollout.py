"""
run_rollout.py — run the model-development process for every price target (stage 9 of docs/Model_Development_Plan.pdf).

    python tools/run_rollout.py --status                     what each target's data allows and how far each has got
    python tools/run_rollout.py --run ida1_price             tuned evaluation, algorithm comparison, champion decision
    python tools/run_rollout.py --run all-ready              the same for every target that is ready and not yet done

Treatments per target: config/rollout.yaml. The steps run one after another (they share the machine) and are resumable: a
step whose output exists is skipped unless --force. The rollout stops at the champion decision. The remaining steps are
deliberate, separate actions: freeze the locked outcomes (tools/evaluate_forecaster.py --lock), the one locked test
(tools/run_locked_test.py --confirm-locked), registering the champion (tools/champion.py init) and calibrating its
monitoring (tools/monitor_champion.py calibrate).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.configuration.forecast_formulation import load_forecast_formulation  # noqa: E402
from common_layer.data_layer.clean_store import CleanStore  # noqa: E402
from common_layer.forecast_evaluation import guards  # noqa: E402
from common_layer.forecast_evaluation.asof import AsOfView  # noqa: E402
from common_layer.market_analysis import panel as pnl  # noqa: E402
from common_layer.model_development import champions, monitoring, rollout  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PHASE9_DA = os.path.join(_REPO, "runtime", "analysis", "phase9", "da_stage2_tpe_v2")


def run_dir(target: str) -> str:
    """Folder of the tuned evaluation of a target (the day-ahead price uses the stage 2 run)."""
    if target == "da_price" and os.path.exists(PHASE9_DA):
        return PHASE9_DA
    return os.path.join(_REPO, "runtime", "analysis", "rollout", target, "tuned")


def paths(target: str) -> dict:
    d = run_dir(target)
    return {"run": d, "run_done": os.path.join(d, "run.json"), "comparison_dir": os.path.join(_REPO, "model_development", "rollout", target),
            "comparison": os.path.join(_REPO, "model_development", "rollout", target, f"{target}_comparison.md"),
            "decision": os.path.join(_REPO, "model_development", "selection", f"{target}.json"),
            "locked": os.path.join(_REPO, "model_development", "locked_test", f"{target}.json")}


def progress(target: str) -> dict:
    p = paths(target)
    champ = champions.active(target)
    base = monitoring.load_baseline(target, champ["version"]) if champ else None
    yes = lambda b: "yes" if b else "-"                                                          # noqa: E731
    return {"tuned_run": yes(os.path.exists(p["run_done"])), "comparison": yes(os.path.exists(p["comparison"])),
            "decision": yes(os.path.exists(p["decision"])), "frozen": yes(target in guards.read_locks()),
            "locked_test": yes(os.path.exists(p["locked"])), "champion": f"v{champ['version']}" if champ else "-",
            "monitoring": yes(base)}


def steps(target: str, py: str = sys.executable) -> list:
    p = paths(target)
    tool = lambda name: os.path.join(_REPO, "tools", name)                                       # noqa: E731
    return [("tuned evaluation", p["run_done"], [py, tool("evaluate_forecaster.py"), "--target", target, "--tuned", "--search-method", "tpe",
                                                "--out", p["run"]]),
            ("algorithm comparison", p["comparison"], [py, tool("compare_algorithms.py"), "--run", p["run"], "--suffix", "_tuned", "--target", target,
                                                       "--label", target, "--out", p["comparison_dir"]]),
            ("champion decision", p["decision"], [py, tool("select_champion.py"), "--run", p["run"], "--suffix", "_tuned", "--target", target])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--run", help="a target id, or all-ready")
    ap.add_argument("--force", action="store_true", help="repeat steps whose output exists")
    a = ap.parse_args()
    form, panel = load_forecast_formulation(), pnl.build_panel(CleanStore())
    st = rollout.status(view_for=lambda t: AsOfView(panel, form, t))
    pr = pd.DataFrame([{"target": t, **progress(t)} for t in st["target"]])
    if a.status or not a.run:
        pd.set_option("display.width", 250)
        print(st.merge(pr, on="target").to_string(index=False))
        return 0
    ready = dict(zip(st["target"], st["ready"]))
    todo = [t for t in st["target"] if ready[t] == "yes" and not os.path.exists(paths(t)["decision"])] if a.run == "all-ready" else [a.run]
    for t in todo:
        if ready.get(t) != "yes":
            print(f"{t}: not ready ({ready.get(t)}); skipped")
            continue
        for name, out, cmd in steps(t):
            if os.path.exists(out) and not a.force:
                print(f"{t}: {name}: done, skipped")
                continue
            print(f"{t}: {name} ...", flush=True)
            t0 = time.time()
            r = subprocess.run(cmd, cwd=_REPO, stdin=subprocess.DEVNULL)
            if r.returncode != 0:
                print(f"{t}: {name} FAILED (exit {r.returncode}); stopping")
                return r.returncode
            print(f"{t}: {name} done in {time.time() - t0:.0f} s")
        print(f"{t}: champion decision ready: {paths(t)['decision']}. Next: freeze outcomes, locked test, register the champion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
