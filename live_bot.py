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
