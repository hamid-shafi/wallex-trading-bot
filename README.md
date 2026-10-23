# Wallex market research and paper trading

[راهنمای کامل فارسی، نصب و داشبورد](README_FA.md)

A Python research and execution engine for Wallex TMN markets, with persistent session state, fee accounting, public market/news inputs, and a Persian dashboard. The documented local setup runs a simulated portfolio using public data; it requires no exchange account key.

## Start here

Use the [Persian setup guide](README_FA.md) for the verified local worker and loopback dashboard commands. It explains the fixed seed deadline, model validity, restart behavior, and separate local state directory. See [FEE_FIX_FA.md](FEE_FIX_FA.md) for the fee-accounting repair and its operational limits.

The bundled model is a fixed research rule policy: `fit=null`, `research_only=true`, and `prospective_validated=false`. It is not evidence of trained ML, an LLM trading agent, or verified investment performance. The bundled 721 hourly bars per market provide warmup, not an independent performance evaluation. The adaptive paper policy only reduces entry risk; historical backtests do not automatically reproduce that policy.

The operational spot adapter targets `main_spot`, not a credit, margin, or competition wallet. No live activation commands are included in the setup guide. Seed dates belong to the original snapshot and do not roll forward automatically.

## Verification

Python 3.12 was used for the reviewed release. Install `requirements.txt` in a virtual environment, then run:

```bash
python -B -m unittest discover -s tests -v
node tests/test_observer_frontend.js
```

The underlying release passed 132 Python tests and the Node.js frontend check. Order tests use synthetic fixtures; these are engineering checks, not backtest or profitability results.

Engine identity, computed over sorted direct `wallex_engine/*.py` filenames and bytes:

```text
59be964cc3730cc67ce36626c8ab2cb5a14883b9de7fc3c682da1980a4a716a9
```

Both bundled model files have SHA-256:

```text
13fe05459e702a1488f36734251a740f4da061132cb873cbba16233e6e8115cb
```

This public distribution replaces private deployment notes while retaining the engine and model bytes. It excludes account credentials, private trading ledgers, and personal server snapshots. Package metadata records release identity, not the current state of any deployed account or server.
