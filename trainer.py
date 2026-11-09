"""
Weekly study: finds the best 5 shares and the best rule set for each.

Runs every Saturday (and by hand from the Actions tab). For each stock in the
pool it replays about a year of 5-minute candles through all the rule sets in
engine.py (5-min entries with 15-min and daily trend filters), after Groww
charges and slippage, split into three periods:

  TRAIN    (oldest half)  : pick the best rule set for each stock
  VALIDATE (next quarter) : pick the 5 stocks whose rule set kept working
  TEST     (newest quarter, never used for any choice): the honest check

A stock is "✅ tested" only if its rule set made money in all three periods.
The result goes to live_config.json, which live_bot.py reads every morning.

Past results do not predict future results. Not financial advice.
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

import engine as E
from sail_intraday_bot import IST, send

CAPITAL = float(os.getenv("BACKTEST_CAPITAL", "20000") or "20000")
DAYS = int(os.getenv("TRAIN_DAYS", "380") or "380")        # calendar days of history
PICK = 5
MIN_TRAIN_TRADES = 20
MIN_VAL_TRADES = 6
MIN_TEST_TRADES = 5
MIN_ACCURACY = 45.0          # % of train trades in profit after charges
WARMUP_DAYS = 25             # first days only warm up the indicators (daily trend)
CONFIG_FILE = "live_config.json"

POOL = ["SAIL", "TATASTEEL", "SBIN", "ICICIBANK", "HDFCBANK", "AXISBANK",
        "RELIANCE", "INFY", "TATAPOWER", "BANKBARODA", "PNB", "ONGC", "NTPC",
        "COALINDIA", "VEDL", "HINDALCO", "JSWSTEEL", "BEL", "BHEL", "NMDC",
        "LT", "KOTAKBANK", "MARUTI", "SUNPHARMA", "BHARTIARTL", "ITC",
        "HINDUNILVR", "BAJFINANCE", "TITAN", "ASIANPAINT", "ADANIPORTS",
        "ADANIENT", "TECHM", "WIPRO", "HCLTECH", "DLF", "GAIL", "IOC", "BPCL",
        "CANBK"]


def stats(trades):
    n = len(trades)
    net = sum(t[1] for t in trades)
    wins = sum(1 for t in trades if t[1] > 0)
    return dict(n=n, net=round(net, 2), acc=round(wins / n * 100, 1) if n else 0.0)


def study_stock(ind, cut1, cut2, start):
    """Best rule set on TRAIN, then its VALIDATE and TEST results."""
    rows = []
    for p in E.combos():
        tr = E.backtest(ind, p, CAPITAL, day_from=start)
        parts = ([t for t in tr if t[0] < cut1],
                 [t for t in tr if cut1 <= t[0] < cut2],
                 [t for t in tr if t[0] >= cut2])
        rows.append((p, *(stats(x) for x in parts)))
    ok = [r for r in rows if r[1]["n"] >= MIN_TRAIN_TRADES and r[1]["acc"] >= MIN_ACCURACY]
    if not ok:
        return None, rows
    best = max(ok, key=lambda r: r[1]["net"])
    return best, rows


def main() -> None:
    g = E.login()
    now = datetime.now(IST)
    start_date = (now - timedelta(days=DAYS)).date()
    results, failed = [], []
    idea_net, idea_n = defaultdict(float), defaultdict(int)
    tdays = []
    for sym in POOL:
        try:
            df = E.fetch_5min(g, sym, start_date, now)
            if len(df) < 75 * 60:
                failed.append(sym)
                continue
            ind = E.indicators(df)
            days = [d for d, _, _ in ind["days"]][WARMUP_DAYS:]
            n = len(days)
            cut1, cut2 = days[n // 2], days[3 * n // 4]
            best, rows = study_stock(ind, cut1, cut2, days[0])
            for p, a, b, c in rows:                  # which ideas work, across stocks
                key = (p["kind"], p.get("tf"))
                idea_net[key] += a["net"]
                idea_n[key] += a["n"]
            tdays = max(tdays, days, key=len)
            if best:
                results.append(dict(sym=sym, params=best[0], label=E.label(best[0]),
                                    train=best[1], val=best[2], test=best[3],
                                    test_days=n - 3 * n // 4))
            print(sym, "->", E.label(best[0]) if best else "nothing usable")
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
            failed.append(sym)

    passed = [r for r in results if r["train"]["net"] > 0 and r["val"]["net"] > 0
              and r["val"]["n"] >= MIN_VAL_TRADES]
    passed.sort(key=lambda r: -r["val"]["net"])
    chosen = passed[:PICK]
    for r in chosen:
        t = r["test"]
        r["status"] = "tested" if t["net"] > 0 and t["n"] >= MIN_TEST_TRADES else "paper"
    # Fewer than 5 kept working: fill the rest with the next best, as paper calls.
    rest = sorted([r for r in results if r not in chosen and r["train"]["net"] > 0],
                  key=lambda r: -r["train"]["net"])
    for r in rest[: PICK - len(chosen)]:
        r["status"] = "paper"
        chosen.append(r)
    paper = not any(r["status"] == "tested" for r in chosen)

    cfg = dict(trained_at=now.isoformat(timespec="minutes"), capital=CAPITAL,
               stocks=[dict(sym=r["sym"], params=r["params"], label=r["label"],
                            status=r["status"], train=r["train"], val=r["val"],
                            test=r["test"]) for r in chosen])
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

    # ---------------------------- report ----------------------------
    lines = [f"🧠 Weekly study: {len(POOL) - len(failed)} "
             f"stocks x {len(E.combos())} rule sets, ~{len(tdays)} trading days, "
             f"Rs {CAPITAL:,.0f} per trade, after charges.", ""]
    if not chosen:
        lines.append("❌ No stock had a rule set that made money even in the training "
                     "period. The live bot will stay quiet this week.")
    else:
        lines.append("Watching next week:" if not paper else
                     "⚠️ Nothing passed every test. Watching the best ones anyway, "
                     "as PAPER calls only (don't use real money):")
        calls_day = 0.0
        for r in chosen:
            tr, va, te = r["train"], r["val"], r["test"]
            icon = "✅" if r["status"] == "tested" else "📝"
            calls_day += te["n"] / max(r["test_days"], 1)
            lines.append(f"{icon} {r['sym']}: {r['label']}\n"
                         f"   train {tr['n']} trades {tr['acc']:.0f}% {tr['net']:+,.0f} | "
                         f"validate {va['n']} {va['acc']:.0f}% {va['net']:+,.0f} | "
                         f"TEST {te['n']} {te['acc']:.0f}% {te['net']:+,.0f}")
        lines += ["", f"About {calls_day:.1f} calls per day in the test period.",
                  "✅ = made money in train, validate AND the unseen test period. "
                  "📝 = paper only: watch, don't trade with real money.",
                  "% = trades that made money after charges."]
    avg = {k: idea_net[k] / idea_n[k] for k in idea_net if idea_n[k]}
    ranked = sorted(avg.items(), key=lambda x: -x[1])
    lines += ["", "Average result per trade in training, all stocks (best 5 ideas):"]
    for (kind, tf), v in ranked[:5]:
        lines.append(f"  {E.NAMES[kind]}{E.TREND_TEXT.get(tf, '')}: Rs {v:+,.0f}/trade")
    if failed:
        lines.append("No data for: " + ", ".join(failed))
    send("\n".join(lines))


if __name__ == "__main__":
    main()
