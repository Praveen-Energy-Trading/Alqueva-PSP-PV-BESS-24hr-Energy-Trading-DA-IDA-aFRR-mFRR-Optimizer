"""
make_results_summary.py — one generated table of where every price target stands (stage 9 of docs/Model_Development_Plan.pdf).

    python tools/make_results_summary.py

Reads model_development/selection/*.json (champion decisions), model_development/locked_test/*.json and config/rollout.yaml, and
asks the registry which champions are active. Writes model_development/RESULTS.md.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from common_layer.model_development import champions, summary  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main() -> int:
    md = os.path.join(_REPO, "model_development")
    df = summary.collect(os.path.join(md, "selection"), os.path.join(md, "locked_test"),
                         active_version=lambda t: (champions.active(t) or {}).get("version"))
    with open(os.path.join(md, "RESULTS.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(summary.render(df))
    print(f"wrote {os.path.join(md, 'RESULTS.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
