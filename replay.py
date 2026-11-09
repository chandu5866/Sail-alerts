"""
Dry run: replays recent trading days through the live bot's exact rules and
sends Telegram what it WOULD have sent - every call with its time, entry,
stop-loss, target, how it ended and the Rs result after charges.

Uses the 5 shares and rules in live_config.json (from the weekly study) and
real 5-minute candles from Groww. No orders, nothing written to the trade log.

Run from the Actions tab: "Replay (dry run)" -> days = how many recent
trading days to replay (1 = the latest day, e.g. today after the close).

Past results do not predict future results. Not financial advice.
"""

import json
import os
from datetime import datetime, timedelta

import engine as E
from sail_intraday_bot import IST, fmt, send

DAYS = max(1, int(os.getenv("REPLAY_DAYS", "1") or "1"))
CONFIG_FILE = "live_config.json"
WHY = {"TARGET": "🎯 target", "STOP": "❌ stop-loss", "CLOSE": "⏰ squared off"}


def send_chunks(parts, limit=3500):
    buf = ""
    for p in parts:
        if buf and len(buf) + len(p) + 1 > limit:
            send(buf)
            buf = ""
        buf = f"{buf}\n{p}" if buf else p
    if buf:
        send(buf)


def main() -> None:
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        send("🔁 Replay: no picks yet. Run the 'Weekly study' workflow first.")
        return
    if not cfg.get("stocks"):
        send("🔁 Replay: the last study found no usable setup, so there is nothing to replay.")
        return
    capital = cfg.get("capital", 20000)
    g = E.login()
    now = datetime.now(IST)
    start = now.date() - timedelta(days=E.HISTORY_DAYS_NEEDED + DAYS * 2 + 7)
    step = timedelta(minutes=E.BAR_MIN)

    per_day, all_days, failed = {}, set(), []
    for stock in cfg["stocks"]:
        sym, p = stock["sym"], stock["params"]
        try:
            df = E.fetch_5min(g, sym, start, now)
        except Exception as e:
            failed.append(f"{sym} ({type(e).__name__})")
            continue
        df = df[df.index + step <= now]                       # completed candles only
        if df.empty:
            failed.append(sym)
            continue
        ind = E.indicators(df)
        s = E.signals(ind, p)
        for d, i0, i1 in ind["days"][-DAYS:]:
            all_days.add(d)
            for t in E.run_day(ind, s, p, i0, i1):
                net = (E.net_pnl(t["side"], t["entry"], t["exit"], capital)
                       if t["exit"] is not None else None)
                per_day.setdefault(d, []).append(dict(
                    sym=sym, status=stock["status"], t=t, net=net,
                    t_in=ind["index"][t["entry_i"]] + step,
                    t_out=(ind["index"][t["exit_i"]] + step) if t["exit_i"] is not None else None))

    days = sorted(all_days)[-DAYS:]
    lines = [f"🔁 DRY RUN - what the bot would have sent ({len(days)} day(s)), "
             f"Rs {capital:,.0f} per call, after charges. No orders were placed.",
             "Shares: " + ", ".join(f"{s['sym']}{'' if s['status'] == 'tested' else ' 📝'}"
                                    for s in cfg["stocks"])]
    tot_n = tot_win = 0
    tot_net = 0.0
    for d in days:
        rows = sorted(per_day.get(d, []), key=lambda r: r["t_in"])
        lines += ["", f"📅 {d:%a %d %b}"]
        if not rows:
            lines.append("  No calls.")
            continue
        day_net, day_win, day_n = 0.0, 0, 0
        for r in rows:
            t = r["t"]
            icon = "🟢 BUY" if t["side"] == "BUY" else "🔴 SELL"
            tag = "" if r["status"] == "tested" else " 📝"
            lines.append(f"  {r['t_in']:%H:%M} {icon} {r['sym']}{tag} at {fmt(t['entry'])} "
                         f"(SL {fmt(t['sl'])}, Tg {fmt(t['tgt'])})")
            if r["net"] is None:
                lines.append("        still open when the data ended")
                continue
            lines.append(f"        {r['t_out']:%H:%M} {WHY[t['why']]} at {fmt(t['exit'])} "
                         f"-> Rs {r['net']:+,.0f}")
            day_n += 1
            day_win += r["net"] > 0
            day_net += r["net"]
        lines.append(f"  Day: {day_win}/{day_n} in profit, net Rs {day_net:+,.0f}")
        tot_n += day_n
        tot_win += day_win
        tot_net += day_net
    if len(days) > 1 and tot_n:
        lines += ["", f"TOTAL: {tot_n} calls, {tot_win / tot_n * 100:.0f}% in profit, "
                      f"net Rs {tot_net:+,.0f} after charges."]
    lines += ["", "Assumes you got in at the signal candle's close. Real fills come a "
                  "little later. One trade at a time: you would skip calls that came "
                  "while you were already in a trade."]
    if failed:
        lines.append("No data for: " + ", ".join(failed))
    send_chunks(lines)


if __name__ == "__main__":
    main()
