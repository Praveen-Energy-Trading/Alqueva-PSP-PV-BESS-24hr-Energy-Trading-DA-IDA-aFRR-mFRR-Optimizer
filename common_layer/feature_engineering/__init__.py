"""
feature_engineering — Phase 7: features with a written availability time, one definition for training,
evaluation and live prediction, a versioned store, and stability-checked selection.

    calendar.py     holidays (Portugal), bridge days, daylight-saving days, cyclical encodings, market rules
    catalogue.py    config/feature_catalogue.yaml: every feature, what it reads, when it is known (derived, checked)
    builder.py      builds the catalogue's features for one delivery day from an as-of view
    store.py        SQLite feature store, one table per (target, feature set), versioned by definition hash
    selection.py    permutation importance per period and the stability of the ranking across periods
"""
