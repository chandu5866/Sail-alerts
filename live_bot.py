"""
Live bot: watches the 5 shares picked by the weekly study (live_config.json)
on every 5-minute candle and sends BUY / SELL calls to Telegram.

Runs every trading day 8:45 AM - 3:31 PM IST as two GitHub jobs (a job can
last at most 6 hours, see live.yml): 8:45 -> 12:00, then 11:55 -> 3:31,
handing over through live_state.json. Every check replays the day with
engine.py (the same code the study tested), so a late or restarted job cannot
skip a target or stop-loss.

Telegram messages
  📋 morning plan (8:45)  🟢 BUY / 🔴 SELL call (9:35 - 2:30)  🎯 target
  ❌ stop-loss  ⏰ square off (3:15)  📊 day scorecard
  A call that reaches you more than LATE_MIN minutes late is marked "missed"
  and its exits stay quiet: never chase a stale call.

Manual run from the Actions tab with mode "test": checks Groww + Telegram and
shows today's plan.

Educational tool, not financial advice. Check the live price before ordering.
"""

import csv
import json
import os
import time
from datetime import datetime, time as dtime, timedelta

import engine as E
from sail_intraday_bot import IST, fmt, send

CONFIG_FILE = "live_config.json"
STATE_FILE = "live_state.json"
LOG_FILE = "trades_log.csv"
HANDOFF = dtime(12, 0, 30)     # the morning job stops here
DAY_END = dtime(15, 31)
SUMMARY_AT = dtime(15, 15, 20)  # right after the 3:10-3:15 candle
LATE_MIN = 8                   # calls older than this are "missed"
GRACE_S = 25                   # wait after a candle closes so Groww has it
HISTORY_DAYS = 10              # calendar days of 5-min candles to warm up indicators
ICON = {"BUY": "🟢", "SELL": "🔴"}


def load_config():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_state(today) -> dict:
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        if s.get("date") == str(today):
            return s
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return dict(date=str(today), sent=[], missed=[], plan_sent=False, done=False)


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ------------------------------ messages ---------------------------------
def plan_msg(cfg, now, late):
    cap = cfg.get("capital", 20000)
    lines = [f"📋 Today's watch ({now:%a %d %b}), picked by the study of "
             f"{cfg['trained_at'][:10]}:", ""]
    for s in cfg["stocks"]:
        te = s["test"]
        icon = "✅" if s["status"] == "tested" else "📝 paper only"
        lines.append(f"{icon} {s['sym']}: {s['label']}\n"
                     f"   unseen test: {te['n']} trades, {te['acc']:.0f}% in profit, "
                     f"Rs {te['net']:+,.0f}")
    lines += ["", f"Calls come between 9:35 and 2:30, Rs {cap:,.0f} per call, intraday (MIS).",
              "📝 = didn't pass every test: watch it, don't use real money."]
    if late:
        lines.append(f"⚠️ Bot started late ({now:%H:%M}). Calls more than {LATE_MIN} min "
                     f"old are marked missed.")
    return "\n".join(lines)


def entry_msg(stock, t, ts, capital):
    qty = max(1, int(capital // t["entry"]))
    verb = "BUY" if t["side"] == "BUY" else "SELL (short)"
    tag = "" if stock["status"] == "tested" else "\n📝 Paper call: this setup didn't pass every test."
    return (f"{ICON[t['side']]} {verb} {stock['sym']} at ~{fmt(t['entry'])}  ({ts:%H:%M})\n"
            f"SL {fmt(t['sl'])} | Target {fmt(t['tgt'])}\n"
            f"Qty ~{qty} for Rs {capital:,.0f}. Intraday (MIS).\n"
            f"Why: {stock['label']}.\n"
            f"Skip if the price already moved 0.3% past entry.{tag}")


def exit_msg(sym, t):
    if t["why"] == "TARGET":
        return f"🎯 {sym}: target {fmt(t['exit'])} reached. Exit the {t['side']} trade."
    if t["why"] == "STOP":
        return f"❌ {sym}: SL {fmt(t['exit'])} hit. Exit the {t['side']} trade."
    return f"⏰ {sym}: square off your {t['side']} trade now (last {fmt(t['exit'])})."


# ------------------------------ scorecard ---------------------------------
def scorecard(state, all_trades, capital):
    rows = []
    for sym, n, t, status in all_trades:
        if t["exit"] is None:
            continue
        net = E.net_pnl(t["side"], t["entry"], t["exit"], capital)
        rows.append(dict(date=state["date"], symbol=sym, side=t["side"], entry=t["entry"],
                         exit=t["exit"], why=t["why"], net=round(net, 2),
                         alerted=int(f"{sym}:{n}" not in state["missed"]),
                         status=status))
    if rows:
        new_file = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new_file:
                w.writeheader()
            w.writerows(rows)
    lines = [f"📊 Day scorecard {state['date']}"]
    if not rows:
        lines.append("No calls today.")
    for x in rows:
        mark = "✅" if x["net"] > 0 else "❌"
        extra = " (missed alert)" if not x["alerted"] else ""
        extra += " (paper)" if x["status"] != "tested" else ""
        lines.append(f"{mark} {x['side']} {x['symbol']} {fmt(x['entry'])} -> "
                     f"{fmt(x['exit'])} {x['why'].lower()}, net Rs {x['net']:+,.0f}{extra}")
    if rows:
        lines.append(f"Today: {sum(x['net'] > 0 for x in rows)}/{len(rows)} in profit, "
                     f"net Rs {sum(x['net'] for x in rows):+,.0f} after charges")
    try:
        with open(LOG_FILE) as f:
            hist = list(csv.DictReader(f))
    except FileNotFoundError:
        hist = []
    tested = [x for x in hist if x.get("status") == "tested"]
    for name, part in (("Last 30 calls", hist[-30:]), ("All calls so far", hist),
                       ("✅ tested calls only", tested)):
        if part:
            n = len(part)
            wins = sum(1 for x in part if float(x["net"]) > 0)
            tot = sum(float(x["net"]) for x in part)
            lines.append(f"{name}: {n}, accuracy {wins / n * 100:.0f}%, net Rs {tot:+,.0f}")
    send("\n".join(lines))


# ------------------------------ main loop ---------------------------------
def check(g, cfg, state, now):
    """Replay today for every watched stock and send anything new."""
    capital = cfg.get("capital", 20000)
    all_trades, have_data = [], False
    for stock in cfg["stocks"]:
        sym = stock["sym"]
        try:
            df = E.fetch_5min(g, sym, now.date() - timedelta(days=HISTORY_DAYS), now, pause=0.1)
        except Exception as e:
            print(sym, "data failed:", type(e).__name__, e)
            continue
        df = df[df.index + timedelta(minutes=E.BAR_MIN) <= now]      # completed candles only
        if df.empty or df.index[-1].date() != now.date():
            continue
        have_data = True
        ind = E.indicators(df)
        d, i0, i1 = ind["days"][-1]
        trades = E.run_day(ind, E.signals(ind, stock["params"]), stock["params"], i0, i1)
        step = timedelta(minutes=E.BAR_MIN)
        for n, t in enumerate(trades):
            all_trades.append((sym, n, t, stock["status"]))
            tkey = f"{sym}:{n}"
            ev = [("ENTRY", ind["index"][t["entry_i"]] + step)]
            if t["status"] == "closed":
                ev.append(("EXIT", ind["index"][t["exit_i"]] + step))
            for kind, ts in ev:
                key = f"{tkey}:{kind}"
                if key in state["sent"]:
                    continue
                state["sent"].append(key)
                if kind == "ENTRY" and (now - ts).total_seconds() / 60 > LATE_MIN:
                    state["missed"].append(tkey)
                    send(f"⏱ Missed: {t['side']} {sym} was due at {ts:%H:%M} "
                         f"(~{fmt(t['entry'])}). Too late now. Don't chase it.")
                    continue
                if tkey in state["missed"]:
                    continue
                send(entry_msg(stock, t, ts, capital) if kind == "ENTRY" else exit_msg(sym, t))
    if not have_data and now.time() >= dtime(9, 40):
        send("⚠️ No candles from Groww today by 9:40. Market holiday or data problem: "
             "no calls today.")
        state["done"] = True
    return all_trades


def next_wake(now, end_dt):
    base = now.replace(hour=9, minute=15, second=0, microsecond=0)
    step = timedelta(minutes=5)
    k = max(1, int((now - base - timedelta(seconds=GRACE_S)) // step) + 1)
    return min(base + k * step + timedelta(seconds=GRACE_S), end_dt)


def session(g, cfg, state, end_t, clock=lambda: datetime.now(IST), sleep=time.sleep):
    today = clock().date()
    end_dt = datetime.combine(today, end_t, tzinfo=IST)
    first = datetime.combine(today, dtime(9, 35, GRACE_S), tzinfo=IST)
    catch_up = clock() >= first
    while True:
        now = clock()
        if now >= end_dt:
            break
        if not catch_up:
            wake = max(next_wake(now, end_dt), min(first, end_dt))
            if wake > now:
                sleep((wake - now).total_seconds())
        catch_up = False
        now = clock()
        trades = check(g, cfg, state, now)
        if not state["done"] and now.time() >= SUMMARY_AT:
            scorecard(state, trades, cfg.get("capital", 20000))
            state["done"] = True
        save_state(state)
        if state["done"]:
            break


def main() -> None:
    now = datetime.now(IST)
    cfg = load_config()
    test = (os.getenv("RUN_MODE") or "run").strip().lower() == "test"
    if not cfg or not cfg.get("stocks"):
        if test or now.time() < dtime(11, 30):           # say it once a day, not twice
            send("⚠️ Live bot: no picks yet. Run the 'Weekly study' workflow first."
                 if not cfg else "ℹ️ Live bot: the last study found no usable setup, "
                 "so there are no calls this week.")
        return
    if test:
        g = E.login()
        df = E.fetch_5min(g, cfg["stocks"][0]["sym"], now.date() - timedelta(days=7), now)
        last = (f"Latest {cfg['stocks'][0]['sym']} candle: {fmt(float(df['Close'].iloc[-1]))} "
                f"at {df.index[-1]:%d %b %H:%M}" if not df.empty else "No candles received.")
        send(f"✅ Live bot test: Groww OK, Telegram OK.\n{last}\n\n" + plan_msg(cfg, now, False))
        return
    if now.weekday() >= 5:
        print("Weekend.")
        return
    state = load_state(now.date())
    if state["done"]:
        print("Today is already finished.")
        return
    if not state["plan_sent"]:
        send(plan_msg(cfg, now, now.time() > dtime(9, 35)))
        state["plan_sent"] = True
        save_state(state)
    g = E.login()
    session(g, cfg, state, HANDOFF if now.time() < dtime(11, 30) else DAY_END)
    save_state(state)


if __name__ == "__main__":
    main()
