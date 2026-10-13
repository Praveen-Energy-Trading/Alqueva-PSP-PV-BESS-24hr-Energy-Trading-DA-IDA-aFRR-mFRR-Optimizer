"""The exploratory mFRR result is labelled as such and never claims a decision."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_mfrr_exploration_is_labelled_and_below_the_evidence_rule():
    path = REPO / "model_development" / "exploratory" / "mfrr_price.json"
    if not path.exists():
        return
    r = json.loads(path.read_text(encoding="utf-8"))
    assert r["status"] == "exploratory" and r["scored_days"] < r["required_days"]
    assert "champion" not in r and "locked" in r["note"]
    assert r["first_test_day"] < "2025-10-12" and r["last_test_day"] <= "2025-10-11"          # the development window only, never the locked year
