"""
data_quality — Phase 3: automated checks, flags, imputation rules, provenance, daily score,
report and the pass/fail gate that must be green before any model run.

    config.py       loads and validates config/data_quality.yaml
    checks.py       gap, flat-run, DST, spike/negative, cross-source and level checks (tidy issue rows)
    scoring.py      daily score per (dataset, day) and the gate decision
    imputation.py   per-variable fill rules that can only look backwards in time
    provenance.py   live / synthetic labels and the training-eligibility rule
    report.py       Markdown report
    runner.py       run everything on the clean zone, store results, enforce the gate
"""
