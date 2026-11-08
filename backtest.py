"""
Multi-stock backtest for the intraday bot.

Replays the EXACT same rules as the live bot (it imports them) over past
5-minute candles from Groww for a pool of liquid stocks, one day at a time,
checking every 5 minutes like the live schedule does.

What it does with real-world friction:
  - Groww intraday charges (brokerage, STT, stamp duty, exchange, SEBI, GST).
  - A small slippage on every entry and exit (SLIP_PCT).
  - Fixed money per trade (CAPITAL), so every stock is compared fairly.

How stocks are chosen (to avoid just picking last season's lucky winners):
  - OLD period = everything except the newest TEST_DAYS days: used to rank.
  - NEW period = the newest TEST_DAYS days: must ALSO be profitable.

Past results do not predict future results. Not financial advice.
"""

import os
import time
from datetime import datetime, time as dtime, timedelta

import pandas as pd

import sail_intraday_bot as bot

DAYS = int(os.getenv("BACKTEST_DAYS", "365") or "365")
CAPITAL = float(os.getenv("BACKTEST_CAPITAL", "20000") or "20000")  # Rs per trade
SLIP_PCT = 0.03       # adverse slippage per entry/exit, % of price
TEST_DAYS = 90        # newest calendar days kept aside as the "new" check
MIN_CALLS = 12
KEEP = 10

POOL = ["SAIL", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK",
        "INFY", "ITC", "NTPC", "ONGC", "COALINDIA", "POWERGRID",
        "TATASTEEL", "VEDL", "BANKBARODA", "BHARTIARTL"]


# ----------------------------- data -----------------------------------
def login():
    from growwapi import GrowwAPI
    import pyotp
    totp = pyotp.TOTP(bot.GROWW_TOTP_SECRET).now()
    token = GrowwAPI.get_access_token(api_key=bot.GROWW_TOTP_TOKEN, totp=totp)
    return GrowwAPI(token)


def fetch_symbol(g, symbol: str, days: int) -> pd.DataFrame:
    end = datetime.now(bot.IST).replace(hour=15, minute=30, second=0, microsecond=0)
    cur = (end - timedelta(days=days)).replace(hour=9, minute=15)
    frames = []
    while cur < end:
        nxt = min(cur + timedelta(days=29), end)   # API limit: 30 days of 5-min candles
        for attempt in range(3):
            try:
                resp = g.get_historical_candles(
                    exchange=g.EXCHANGE_NSE,
                    segment=g.SEGMENT_CASH,
                    groww_symbol=f"NSE-{symbol}",
                    start_time=cur.strftime("%Y-%m-%d %H:%M:%S"),
                    end_time=nxt.strftime("%Y-%m-%d %H:%M:%S"),
                    candle_interval=g.CANDLE_INTERVAL_MIN_5,
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        frames.append(bot.candles_to_df(resp.get("candles")))
        cur = nxt
        time.sleep(0.4)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


# --------------------------- simulation --------------------------------
def simulate_day(day, day_df):
    """Replays one day. Returns a trade dict or None."""
    state, trade = {}, None
    start = datetime.combine(day, dtime(9, 30), tzinfo=bot.IST) + timedelta(seconds=30)
    for k in range(80):
        now = start + timedelta(minutes=5 * k)
        if now.time() > bot.STOP_T:
            break
        if trade is None and (state.get("status") == "skipped"
                              or now.time() > bot.NO_NEW_AFTER):
            break  # nothing can happen any more today
        prev = state.get("status")
        state, msgs = bot.process(day_df, state, now)
        status = state.get("status")

        if status == "open" and trade is None:
            trade = dict(day=day, side=state["side"], entry=state["entry"],
                         sl=state["sl"], t1=state["t1"], t2=state["t2"])
        if trade and prev == "open" and status == "closed":
            head = msgs[0] if msgs else ""
            if head.startswith("❌"):
                trade["why"] = "SL"
            elif head.startswith("⚪"):
                trade["why"] = "T1 then cost"
            elif head.startswith("🎯🎯"):
                trade["why"] = "T2"
            else:
                trade["why"] = "square-off"
                done = day_df[day_df.index + timedelta(minutes=bot.BAR_MIN) <= now]
                trade["last"] = float(done["Close"].iloc[-1])
            trade["t1_hit"] = bool(state.get("t1_hit"))
            return trade
    return None


def leg_cost(value: float, side: str) -> float:
    """Groww intraday charges for one executed order."""
    brokerage = min(20.0, 0.001 * value)
    exch = value * 0.0000297          # NSE transaction charge
    sebi = value * 0.000001
    ipft = value * 0.000001
    gst = 0.18 * (brokerage + exch)
    stt = 0.00025 * value if side == "sell" else 0.0
    stamp = 0.00003 * value if side == "buy" else 0.0
    return brokerage + exch + sebi + ipft + gst + stt + stamp


def evaluate(t: dict) -> float:
    """Net profit in Rs for one trade after charges and slippage."""
    entry, side = t["entry"], t["side"]
    qty = max(1, int(CAPITAL / entry))
    half = qty // 2
    why, last = t["why"], t.get("last")
    if why == "SL":
        legs = [(qty, t["sl"])]
    elif why == "T1 then cost":
        legs = [(half, t["t1"]), (qty - half, entry)]
    elif why == "T2":
        legs = [(half, t["t1"]), (qty - half, t["t2"])]
    elif t.get("t1_hit"):
        legs = [(half, t["t1"]), (qty - half, last)]
    else:
        legs = [(qty, last)]
    legs = [(q, p) for q, p in legs if q > 0]

    sign = 1 if side == "BUY" else -1
    s = SLIP_PCT / 100
    e_fill = entry * (1 + sign * s)
    entry_side, exit_side = ("buy", "sell") if side == "BUY" else ("sell", "buy")
    gross, cost = 0.0, leg_cost(qty * e_fill, entry_side)
    for q, p in legs:
        x_fill = p * (1 - sign * s)
        gross += sign * q * (x_fill - e_fill)
        cost += leg_cost(q * x_fill, exit_side)
    return gross - cost


def run_symbol(df: pd.DataFrame):
    trades, days = [], 0
    for day, g in df.groupby(df.index.date):
        if len(g[(g.index.time >= bot.OPEN_T) & (g.index.time < bot.RANGE_END)]) < 3:
            continue
        days += 1
        t = simulate_day(day, g)
        if t:
            t["net"] = evaluate(t)
            trades.append(t)
    return days, trades


# ---------------------------- reporting --------------------------------
def analyse(results: dict):
    """results: symbol -> (days, trades). Returns rows and the chosen list."""
    all_days = [t["day"] for _, tr in results.values() for t in tr]
    if not all_days:
        return [], []
    cutoff = max(all_days) - timedelta(days=TEST_DAYS)
    rows = []
    for sym, (days, trades) in results.items():
        old = sum(t["net"] for t in trades if t["day"] < cutoff)
        new = sum(t["net"] for t in trades if t["day"] >= cutoff)
        wins = sum(1 for t in trades if t["net"] > 0)
        rows.append(dict(sym=sym, calls=len(trades), wins=wins, old=old,
                         new=new, total=old + new))
    rows.sort(key=lambda r: r["old"], reverse=True)
    chosen = [r for r in rows if r["calls"] >= MIN_CALLS
              and r["old"] > 0 and r["new"] > 0][:KEEP]
    return rows, chosen


def report(days_tested: int, rows: list, chosen: list, failed: list) -> list:
    out = []
    head = [f"📊 Backtest: {len(rows)} stocks, about {days_tested} trading days",
            f"Rs {CAPITAL:,.0f} per trade, after Groww charges + slippage.",
            f"'old' = ranking period, 'new' = newest {TEST_DAYS} days (unseen).", ""]
    lines = [f"{r['sym']}: {r['calls']} calls, win {r['wins']}, "
             f"old {r['old']:+,.0f}, new {r['new']:+,.0f}" for r in rows]
    out.append("\n".join(head + lines))
    if failed:
        out.append("No data for: " + ", ".join(failed))
    if chosen:
        tot = sum(r["total"] for r in chosen)
        names = ", ".join(r["sym"] for r in chosen)
        out.append(f"✅ Stocks profitable in BOTH periods (top {KEEP}):\n{names}\n\n"
                   f"Together they made Rs {tot:+,.0f} over the whole test "
                   f"(about Rs {tot / max(days_tested, 1):+,.0f} per trading day).\n\n"
                   "Past results do not predict the future. Real fills are "
                   "worse than this test. A filter, not a promise.")
    else:
        out.append("❌ No stock was profitable in both periods after costs. "
                   "These rules are not good enough to trade yet.")
    return out


def main() -> None:
    g = login()
    results, failed = {}, []
    for sym in POOL:
        try:
            df = fetch_symbol(g, sym, DAYS)
            if df.empty:
                failed.append(sym)
                continue
            results[sym] = run_symbol(df)
            print(sym, "done:", results[sym][0], "days,", len(results[sym][1]), "calls")
        except Exception as e:
            print(sym, "failed:", type(e).__name__, e)
            failed.append(sym)
    if not results:
        bot.send("📊 Backtest: Groww returned no candles. Check the Groww "
                 "secrets and that the API plan is active.")
        return
    rows, chosen = analyse(results)
    days_tested = max(d for d, _ in results.values())
    for m in report(days_tested, rows, chosen, failed):
        bot.send(m)


if __name__ == "__main__":
    main()
