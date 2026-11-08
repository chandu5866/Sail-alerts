"""
Backtest for the SAIL intraday bot.

Replays the EXACT same rules as the live bot (it imports them) over past
5-minute candles from Groww, one day at a time, checking every 5 minutes
like the live schedule does. Sends a summary to Telegram.

Simple assumptions (real trading will be worse):
  - Entry is the close of the signal candle (live, you enter a bit later).
  - At Target 1 half the position is sold, the rest runs to Target 2,
    back to cost, or the 3:05 PM square-off.
  - A rough cost of COST_PCT of the price is deducted per trade
    (brokerage, taxes, slippage). Real costs depend on quantity.
Past results do not predict future results. Not financial advice.
"""

import os
import time
from datetime import datetime, time as dtime, timedelta

import pandas as pd

import sail_intraday_bot as bot

DAYS = int(os.getenv("BACKTEST_DAYS", "120") or "120")
QTY = 100            # shares, only used to show rupee amounts
COST_PCT = 0.10      # rough cost per trade, % of entry price


def fetch(days: int) -> pd.DataFrame:
    from growwapi import GrowwAPI
    import pyotp
    totp = pyotp.TOTP(bot.GROWW_TOTP_SECRET).now()
    token = GrowwAPI.get_access_token(api_key=bot.GROWW_TOTP_TOKEN, totp=totp)
    g = GrowwAPI(token)

    end = datetime.now(bot.IST).replace(hour=15, minute=30, second=0, microsecond=0)
    cur = (end - timedelta(days=days)).replace(hour=9, minute=15)
    frames = []
    while cur < end:
        nxt = min(cur + timedelta(days=29), end)   # API limit: 30 days of 5-min candles
        resp = g.get_historical_candles(
            exchange=g.EXCHANGE_NSE,
            segment=g.SEGMENT_CASH,
            groww_symbol=bot.GROWW_SYMBOL,
            start_time=cur.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=nxt.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval=g.CANDLE_INTERVAL_MIN_5,
        )
        frames.append(bot.candles_to_df(resp.get("candles")))
        cur = nxt
        time.sleep(1)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def simulate_day(day, day_df):
    """Returns a trade dict or None, plus a reason string if no trade."""
    state, trade = {}, None
    start = datetime.combine(day, dtime(9, 30), tzinfo=bot.IST) + timedelta(seconds=30)
    for k in range(0, 80):
        now = start + timedelta(minutes=5 * k)
        if now.time() > bot.STOP_T:
            break
        prev = state.get("status")
        state, msgs = bot.process(day_df, state, now)
        status = state.get("status")

        if status == "open" and trade is None:
            trade = dict(day=day, side=state["side"], entry=state["entry"],
                         sl=state["sl"], t1=state["t1"], t2=state["t2"],
                         time=now.strftime("%H:%M"))
        if trade and prev == "open" and status == "closed":
            sign = 1 if trade["side"] == "BUY" else -1
            risk = abs(trade["entry"] - trade["sl"])
            head = msgs[0] if msgs else ""
            if head.startswith("❌"):
                pnl, why = -risk, "SL"
            elif head.startswith("⚪"):
                pnl, why = 0.5 * risk, "T1 then cost"
            elif head.startswith("🎯🎯"):
                pnl, why = 1.5 * risk, "T2"
            else:  # square-off
                done = day_df[day_df.index + timedelta(minutes=bot.BAR_MIN) <= now]
                last = float(done["Close"].iloc[-1])
                move = sign * (last - trade["entry"])
                pnl = (0.5 * risk + 0.5 * move) if state.get("t1_hit") else move
                why = "square-off"
            trade.update(pnl=pnl, why=why, risk=risk)
            return trade
    if trade:  # never closed (should not happen)
        trade.update(pnl=0.0, why="open", risk=abs(trade["entry"] - trade["sl"]))
    return trade


def run_backtest(df: pd.DataFrame):
    trades, days = [], 0
    for day, g in df.groupby(df.index.date):
        if len(g[(g.index.time >= bot.OPEN_T) & (g.index.time < bot.RANGE_END)]) < 3:
            continue
        days += 1
        t = simulate_day(day, g)
        if t:
            trades.append(t)
    return days, trades


def summarize(days: int, trades: list) -> str:
    if not trades:
        return (f"📊 {bot.NAME} backtest: {days} trading days tested, "
                f"the rules made no calls.")
    for t in trades:
        t["net"] = t["pnl"] - t["entry"] * COST_PCT / 100
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    gross = sum(t["pnl"] for t in trades) * QTY
    net = sum(t["net"] for t in trades) * QTY
    streak = cur = 0
    for t in trades:
        cur = cur + 1 if t["pnl"] < 0 else 0
        streak = max(streak, cur)
    buys = sum(1 for t in trades if t["side"] == "BUY")
    lines = [
        f"📊 {bot.NAME} backtest ({days} trading days)",
        f"Calls: {len(trades)} (buy {buys}, sell {len(trades) - buys})",
        f"Winners {len(wins)} | Losers {len(losses)} | "
        f"Win rate {len(wins) / len(trades) * 100:.0f}%",
        f"Result for {QTY} shares, before costs: Rs {gross:,.0f}",
        f"After rough costs ({COST_PCT}% per trade): Rs {net:,.0f}",
        f"Longest losing streak: {streak}",
        "",
        "Last calls:",
    ]
    for t in trades[-8:]:
        lines.append(f"{t['day']:%d %b} {t['side']} {t['entry']:.2f} "
                     f"-> {t['why']} ({t['pnl'] * QTY:+,.0f})")
    lines += ["", "Past results do not predict the future. Real fills are "
              "worse than this test. Treat it as a filter, not a promise."]
    return "\n".join(lines)


def main() -> None:
    df = fetch(DAYS)
    if df.empty:
        bot.send("📊 Backtest: Groww returned no candles. Check the Groww "
                 "secrets and that the API plan is active.")
        return
    days, trades = run_backtest(df)
    msg = summarize(days, trades)
    bot.send(msg)


if __name__ == "__main__":
    main()
