"""
data_layer — the data foundation of the forecasting programme (Phase 2).

Three zones, each versioned and rebuildable:
    raw      what was downloaded, byte for byte, never edited (runtime/data/raw)
    clean    parsed, validated, true-time tables in UTC (runtime/data/clean)
    features model-ready views (Phase 7)

Modules:
    calendar   true market time: UTC period starts, 23/25-hour and 92/100-quarter days
    regimes    dated market-rule changes, read from config/market_regimes.yaml
    raw_store  immutable raw captures with checksum manifest
    parsers    raw bytes -> tidy tables (pure functions)
    contracts  data contracts: shape, range, completeness, freshness
    clean_store  SQLite store of the clean tables
    fetchers   polite HTTP download with retries
    ingest     incremental, idempotent loading with a load log, and rebuild from raw
"""
