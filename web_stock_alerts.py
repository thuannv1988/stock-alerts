"""
web_stock_alerts.py
--------------------
A local web-based dashboard for the same MACD/EMA alert scanner - runs in
your browser instead of a Tkinter window, so it sidesteps Tcl/Tk issues
entirely.

Shows a spreadsheet-style table:
    Ticker | Price | EMA10 | Delta% | EMA50 | Delta% | EMA100 | Delta% |
    EMA150 | Delta% | EMA200 | Delta% | MACD | Signal | Hist | Trend | Comments

Features
--------
- Timeframe selector: 1 Month / 1 Week / 1 Day / 4 Hour (as buttons)
- "Run" button to fetch fresh data and refill the table
- Add / remove tickers from the watchlist directly in the page
- Rows that are "near" an EMA or "near" a MACD crossover are highlighted
- Click any ticker symbol to open a chart window: candlesticks with EMA
  overlays on top, MACD line/signal/histogram on a linked panel below

HOW TO RUN
----------
Same venv/folder as your other scripts:
    python web_stock_alerts.py
Then open your browser to:
    http://127.0.0.1:5000
Leave the terminal running while you use the page. Press Ctrl+C in the
terminal to stop the server when you're done.

Dependencies: flask, pandas, yfinance
    pip install flask pandas yfinance
(or add "flask" to requirements.txt and run pip install -r requirements.txt)
"""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from functools import wraps
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from flask import (
    Flask, jsonify, redirect, render_template_string, request, session, url_for,
)
from werkzeug.security import check_password_hash

# --------------------------------------------------------------------------
# LOGIN CONFIG
# --------------------------------------------------------------------------
# Credentials are read from environment variables - NEVER hardcode a real
# password in this file. Generate a hash locally with gen_password_hash.py,
# then set these two env vars (locally, or in your hosting platform's
# dashboard):
#   ADMIN_EMAIL           e.g. you@example.com
#   ADMIN_PASSWORD_HASH   the hash printed by gen_password_hash.py
#   SECRET_KEY            any long random string (keeps login sessions
#                          secure and consistent across server restarts)
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "")

# --------------------------------------------------------------------------
# CONFIG (same values/logic as the desktop version)
# --------------------------------------------------------------------------

DEFAULT_WATCHLIST = ["NOW", "CRM", "ORCL", "NFLX", "NVO", "NKE", "ADBE", "BULL", "HOOD", "BBAI"]

EMA_PERIODS = [10, 50, 100, 150, 200]

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

EMA_PROXIMITY_PCT = 1.5
MACD_LOOKBACK_BARS = 5
MACD_NEAR_PCT_OF_AVG = 25.0

TIMEFRAME_OPTIONS: Dict[str, Dict[str, Optional[str]]] = {
    "1 Month": {"period": "10y", "interval": "1mo", "resample": None},
    "1 Week":  {"period": "5y",  "interval": "1wk", "resample": None},
    "1 Day":   {"period": "2y",  "interval": "1d",  "resample": None},
    "4 Hour":  {"period": "60d", "interval": "1h",  "resample": "4h"},
}

COLUMNS = [
    ("ticker", "Ticker"), ("price", "Price"),
    ("ema10", "EMA10"), ("d10", "Delta %"),
    ("ema50", "EMA50"), ("d50", "Delta %"),
    ("ema100", "EMA100"), ("d100", "Delta %"),
    ("ema150", "EMA150"), ("d150", "Delta %"),
    ("ema200", "EMA200"), ("d200", "Delta %"),
    ("macd", "MACD"), ("signal", "Signal"), ("hist", "Hist"),
    ("trend", "Bullish/Bearish"), ("action", "Action"), ("comments", "Comments"),
]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.permanent_session_lifetime = timedelta(minutes=15)


# --------------------------------------------------------------------------
# ANALYSIS (identical math to the desktop GUI)
# --------------------------------------------------------------------------

def fetch_history(ticker: str, timeframe_cfg: dict) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(
        period=timeframe_cfg["period"],
        interval=timeframe_cfg["interval"],
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data returned for '{ticker}'.")

    resample_rule = timeframe_cfg.get("resample")
    if resample_rule:
        df = df.resample(resample_rule).agg(
            {"Open": "first", "High": "max", "Low": "min",
             "Close": "last", "Volume": "sum"}
        ).dropna()

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for period in EMA_PERIODS:
        df[f"EMA{period}"] = df["Close"].ewm(span=period, adjust=False).mean()

    ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    return df


def build_alert_comment(df: pd.DataFrame, price: float):
    """
    Shared logic used by both the table scan and the chart page.
    Returns (comment_text, overall_flagged, trend, near_cols) where
    near_cols is the list of specific column ids (e.g. "ema10", "d10",
    "macd") that triggered an alert - used to highlight just those cells.
    """
    flagged = False
    comments: List[str] = []
    near_cols: List[str] = []

    for period in EMA_PERIODS:
        ema_val = df[f"EMA{period}"].iloc[-1]
        pct = (price - ema_val) / ema_val * 100.0
        near = abs(pct) <= EMA_PROXIMITY_PCT
        if near:
            flagged = True
            near_cols.append(f"ema{period}")
            near_cols.append(f"d{period}")
            side = "above" if pct >= 0 else "below"
            comments.append(f"Near EMA{period} ({side})")

    hist = df["MACD_HIST"]
    latest_hist = hist.iloc[-1]
    prev_hist = hist.iloc[-2] if len(hist) > 1 else latest_hist
    macd_val = df["MACD"].iloc[-1]
    signal_val = df["MACD_SIGNAL"].iloc[-1]
    trend = "BULLISH" if macd_val > signal_val else "BEARISH"

    narrowing = abs(latest_hist) < abs(prev_hist)
    recent_avg_abs = hist.tail(MACD_LOOKBACK_BARS).abs().mean()
    near_cross = False
    if recent_avg_abs and not pd.isna(recent_avg_abs) and recent_avg_abs != 0:
        near_cross = narrowing and (
            abs(latest_hist) <= recent_avg_abs * (MACD_NEAR_PCT_OF_AVG / 100.0)
        )

    if near_cross:
        flagged = True
        near_cols.extend(["macd", "signal", "hist"])
        comments.append("MACD near crossover")
    elif narrowing:
        near_cols.extend(["macd", "signal", "hist"])
        comments.append("MACD gap narrowing")

    comment_text = "; ".join(comments) if comments else "No alerts - all clear"
    return comment_text, flagged, trend, near_cols, macd_val, signal_val, latest_hist


def determine_action(df: pd.DataFrame) -> str:
    """Looks at the two most recent MACD histogram bars to see if a crossover
    JUST happened on the latest completed bar. Returns 'BUY' (crossed up),
    'SELL' (crossed down), or '-' (no fresh cross - trend just continuing)."""
    hist = df["MACD_HIST"]
    if len(hist) < 2:
        return "-"
    prev, latest = hist.iloc[-2], hist.iloc[-1]
    if pd.isna(prev) or pd.isna(latest):
        return "-"
    if prev <= 0 < latest:
        return "BUY"
    if prev >= 0 > latest:
        return "SELL"
    return "-"


def get_action_signal(ticker: str, timeframe_label: str, timeframe_cfg: dict, already_fetched_df: pd.DataFrame) -> str:
    """Action is always based on the 4-Hour MACD specifically, regardless of
    which timeframe is currently selected for the rest of the table. Reuses
    the already-fetched dataframe when the main view IS 4-Hour (avoids a
    redundant network call); otherwise fetches 4h data separately."""
    if timeframe_label == "4 Hour":
        return determine_action(already_fetched_df)
    try:
        df4h = fetch_history(ticker, TIMEFRAME_OPTIONS["4 Hour"])
        df4h = compute_indicators(df4h)
        return determine_action(df4h)
    except Exception:  # noqa: BLE001 - a failed 4h fetch shouldn't break the row
        return "-"


def analyze_ticker(ticker: str, timeframe_cfg: dict, timeframe_label: str = "1 Day") -> dict:
    try:
        df = fetch_history(ticker, timeframe_cfg)
        df = compute_indicators(df)
    except Exception as exc:  # noqa: BLE001
        return {"ticker": ticker, "error": str(exc)}

    price = df["Close"].iloc[-1]
    values: Dict[str, str] = {"ticker": ticker, "price": f"{price:,.2f}"}

    for period in EMA_PERIODS:
        ema_val = df[f"EMA{period}"].iloc[-1]
        pct = (price - ema_val) / ema_val * 100.0
        values[f"ema{period}"] = f"{ema_val:,.2f}"
        values[f"d{period}"] = f"{pct:+.2f}%"

    comment_text, flagged, trend, near_cols, macd_val, signal_val, latest_hist = (
        build_alert_comment(df, price)
    )

    values["macd"] = f"{macd_val:+.4f}"
    values["signal"] = f"{signal_val:+.4f}"
    values["hist"] = f"{latest_hist:+.4f}"
    values["trend"] = trend
    values["action"] = get_action_signal(ticker, timeframe_label, timeframe_cfg, df)
    values["comments"] = comment_text
    values["flagged"] = flagged
    values["near_cols"] = near_cols
    values["error"] = None
    return values


# --------------------------------------------------------------------------
# LOGIN / AUTH
# --------------------------------------------------------------------------

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sign in - Admin Dashboard</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; background: #f3f4f6; margin: 0;
         display: flex; align-items: center; justify-content: center; height: 100vh; }
  .box { background: white; padding: 32px 36px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
         width: 320px; }
  h1 { font-size: 18px; margin: 0 0 18px 0; text-align: center; }
  label { font-size: 13px; color: #444; display: block; margin-bottom: 4px; margin-top: 12px; }
  input[type=email], input[type=password] { width: 100%; padding: 8px; font-size: 14px;
         box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
  button { width: 100%; margin-top: 18px; padding: 10px; font-size: 14px; font-weight: 600;
           background: #1565c0; color: white; border: none; border-radius: 4px; cursor: pointer; }
  button:hover { background: #1257a8; }
  .error { color: #c62828; font-size: 13px; margin-top: 12px; text-align: center; }
</style>
</head>
<body>
  <form class="box" method="POST">
    <h1>Admin Dashboard</h1>
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required autofocus>
    <label for="password">Password</label>
    <input type="password" id="password" name="password" required>
    <button type="submit">Sign in</button>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
  </form>
</body>
</html>
"""


@app.route("/healthz")
def healthz():
    # Deliberately NOT behind login_required - this is only a lightweight
    # "are you awake" endpoint for an external keep-alive pinger (e.g.
    # UptimeRobot) to hit every few minutes, so the free-tier host doesn't
    # spin the app down from inactivity. It reveals nothing about your
    # watchlist or data.
    return "OK", 200


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not ADMIN_EMAIL or not ADMIN_PASSWORD_HASH:
            error = "Server is not configured with login credentials yet."
        elif email == ADMIN_EMAIL.strip().lower() and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session.clear()
            session["logged_in"] = True
            session.permanent = True
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        else:
            error = "Invalid email or password."

    return render_template_string(LOGIN_TEMPLATE, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------

@app.route("/api/scan", methods=["POST"])
@login_required
def api_scan():
    payload = request.get_json(force=True) or {}
    tickers: List[str] = payload.get("tickers", [])
    timeframe: str = payload.get("timeframe", "1 Day")
    timeframe_cfg = TIMEFRAME_OPTIONS.get(timeframe, TIMEFRAME_OPTIONS["1 Day"])

    results = [analyze_ticker(t, timeframe_cfg, timeframe) for t in tickers]
    return jsonify({"results": results})


@app.route("/api/chart-data/<ticker>")
@login_required
def api_chart_data(ticker):
    timeframe = request.args.get("timeframe", "1 Day")
    timeframe_cfg = TIMEFRAME_OPTIONS.get(timeframe, TIMEFRAME_OPTIONS["1 Day"])

    try:
        df = fetch_history(ticker, timeframe_cfg)
        df = compute_indicators(df)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    # Short, readable date labels per timeframe (long ISO strings clutter
    # the x-axis when the category axis has hundreds of ticks).
    interval = timeframe_cfg["interval"]
    if interval == "1mo":
        date_fmt = "%b %Y"          # e.g. "Feb 2024"
    elif interval == "1wk":
        date_fmt = "%b %d, %y"      # e.g. "Feb 04, 24"
    elif interval == "1d":
        date_fmt = "%m/%d/%y"       # e.g. "02/04/24"
    else:  # 4-hour (resampled hourly)
        date_fmt = "%m/%d %H:%M"    # e.g. "02/04 12:00"

    # EMAs as an ORDERED LIST of [name, values] pairs, not a dict - Flask's
    # jsonify sorts dict keys alphabetically by default, which would scramble
    # EMA10/EMA50/EMA100/EMA150/EMA200 into alphabetical order (EMA10,
    # EMA100, EMA150, EMA200, EMA50). A list preserves the order we want.
    emas_ordered = [
        [f"EMA{p}", [round(v, 2) if pd.notna(v) else None for v in df[f"EMA{p}"].tolist()]]
        for p in EMA_PERIODS
    ]

    price = df["Close"].iloc[-1]
    comment_text, flagged, trend, _near_cols, _m, _s, _h = build_alert_comment(df, price)

    data = {
        "ticker": ticker.upper(),
        "timeframe": timeframe,
        "comment": comment_text,
        "flagged": flagged,
        "trend": trend,
        "dates": [d.strftime(date_fmt) for d in df.index],
        "open": [round(v, 2) for v in df["Open"].tolist()],
        "high": [round(v, 2) for v in df["High"].tolist()],
        "low": [round(v, 2) for v in df["Low"].tolist()],
        "close": [round(v, 2) for v in df["Close"].tolist()],
        "emas": emas_ordered,
        "macd": [round(v, 4) if pd.notna(v) else None for v in df["MACD"].tolist()],
        "signal": [round(v, 4) if pd.notna(v) else None for v in df["MACD_SIGNAL"].tolist()],
        "hist": [round(v, 4) if pd.notna(v) else None for v in df["MACD_HIST"].tolist()],
    }
    return jsonify(data)


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stock Alert Dashboard</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #fafafa; color: #222; }
  h1 { font-size: 20px; margin-bottom: 12px; }
  .controls { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
  select, input[type=text] { padding: 6px 8px; font-size: 14px; }
  button { padding: 7px 14px; font-size: 14px; cursor: pointer; border: 1px solid #888; background: #eee; border-radius: 4px; }
  button:hover { background: #e0e0e0; }
  button:disabled { opacity: 0.5; cursor: default; }
  #runBtn { background: #2e7d32; color: white; border-color: #2e7d32; font-weight: bold; }
  #runBtn:hover { background: #276b2c; }
  .tf-group { display: flex; gap: 6px; }
  .tf-btn { background: #fff; border: 1px solid #999; min-width: 52px; font-weight: 600; }
  .tf-btn.active { background: #1565c0; color: white; border-color: #1565c0; }
  .tf-btn.active:hover { background: #1257a8; }
  .tab-btn { background: #eee; border: 1px solid #999; font-weight: 600; padding: 8px 16px;
             border-radius: 6px 6px 0 0; }
  .tab-btn.active { background: #0d47a1; color: white; border-color: #0d47a1; }
  .tab-add-btn { background: #fff; color: #1565c0; border-style: dashed; }
  .tab-icon { cursor: pointer; font-size: 13px; margin-left: 2px; color: #444; }
  .tab-icon:hover { color: #000; }
  .tab-icon-danger { color: #c62828; }
  .tab-icon-danger:hover { color: #900; }
  table { border-collapse: collapse; width: 100%; background: white; font-size: 13px; }
  th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: center; white-space: nowrap; }
  th { background: #1565c0; color: white; }
  th.group-header { background: #0d47a1; font-size: 14px; }
  tr.flagged { background: #d9f2d9; }
  tr.error { background: #f7d6d6; }
  td.cell-flag { color: #c62828; font-weight: 700; }
  td.comments, th.comments { white-space: normal; text-align: left; max-width: 320px; }
  .remove-btn { color: #a33; cursor: pointer; font-weight: bold; margin-left: 6px; }
  #status { margin-top: 10px; color: #555; font-size: 13px; }
  .table-wrap { overflow-x: auto; }
</style>
</head>
<body>

<h1>Stock Alert Dashboard - MACD &amp; EMA
  <a href="/education" style="font-size:13px; font-weight:normal; color:#1565c0; margin-left:16px; text-decoration:underline;">📚 Education</a>
  <a href="/logout" style="font-size:13px; font-weight:normal; color:#c62828; margin-left:16px; text-decoration:underline;">Log out</a>
</h1>

<div id="tabsBar" style="display:flex; align-items:center; flex-wrap:wrap; gap:6px; margin-bottom:14px;"></div>

<div class="controls">
  <span>Timeframe:</span>
  <div class="tf-group" id="tfGroup">
    <button class="tf-btn" data-tf="1 Month" onclick="selectTimeframe('1 Month')">1M</button>
    <button class="tf-btn" data-tf="1 Week" onclick="selectTimeframe('1 Week')">1W</button>
    <button class="tf-btn active" data-tf="1 Day" onclick="selectTimeframe('1 Day')">1D</button>
    <button class="tf-btn" data-tf="4 Hour" onclick="selectTimeframe('4 Hour')">4h</button>
  </div>

  <button id="runBtn" onclick="runScan()" style="margin-left:16px;">Run</button>

  <label for="tickerInput" style="margin-left:16px;">Add ticker:</label>
  <input type="text" id="tickerInput" placeholder="e.g. AAPL" onkeydown="if(event.key==='Enter') addTicker();">
  <button onclick="addTicker()">Add</button>
</div>

<div id="watchlistBar" style="margin-bottom:10px; font-size:13px;"></div>

<div class="table-wrap">
<table id="resultsTable">
  <thead>
    <tr>
      <th rowspan="2">Ticker</th>
      <th rowspan="2">Price</th>
      <th class="group-header" colspan="10">EMA Indicator</th>
      <th class="group-header" colspan="6">MACD Indicator</th>
    </tr>
    <tr>
      <th>EMA10</th><th>Delta %</th>
      <th>EMA50</th><th>Delta %</th>
      <th>EMA100</th><th>Delta %</th>
      <th>EMA150</th><th>Delta %</th>
      <th>EMA200</th><th>Delta %</th>
      <th>MACD</th><th>Signal</th><th>Hist</th>
      <th>Bullish/Bearish</th><th title="Always based on the 4-Hour MACD crossover, regardless of the timeframe selected above. BUY = MACD just crossed above signal on the latest 4h bar. SELL = MACD just crossed below. &#8722; = no fresh crossover right now." style="cursor:help; border-bottom:1px dotted #fff;">Action &#9432;</th><th class="comments">Comments</th>
    </tr>
  </thead>
  <tbody id="resultsBody"></tbody>
</table>
</div>

<div id="status">Ready.</div>

<script>
const COLUMNS = {{ columns_json | safe }};
const DEFAULT_WATCHLIST = {{ watchlist_json | safe }};
const WORKSPACES_STORAGE_KEY = 'stockAlerts_workspaces';

// A "workspace" is one tab: {id, name, watchlist: [tickers]}. Each tab has
// its own independently saved watchlist, all stored together in this
// browser's local storage.
function loadWorkspaces() {
  try {
    const saved = localStorage.getItem(WORKSPACES_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (err) {
    console.warn('Could not read saved tabs, starting fresh.', err);
  }
  return [{id: 'ws-default', name: 'My Stocks', watchlist: [...DEFAULT_WATCHLIST]}];
}

function saveWorkspaces() {
  // keep the active workspace's stored watchlist in sync with the live array
  const active = getActiveWorkspace();
  if (active) active.watchlist = watchlist;
  try {
    localStorage.setItem(WORKSPACES_STORAGE_KEY, JSON.stringify(workspaces));
  } catch (err) {
    console.warn('Could not save tabs to this browser.', err);
  }
}

function getActiveWorkspace() {
  return workspaces.find(w => w.id === activeWorkspaceId) || workspaces[0];
}

const RESULTS_CACHE_STORAGE_KEY = 'stockAlerts_resultsCache';

// Loads the last-scan cache (per tab) from this browser's local storage,
// so a page refresh still shows your most recent numbers instead of a
// blank table - you'll still see a timestamp so it's clear how fresh it is.
function loadResultsCache() {
  try {
    const saved = localStorage.getItem(RESULTS_CACHE_STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed && typeof parsed === 'object') return parsed;
    }
  } catch (err) {
    console.warn('Could not read saved scan results, starting fresh.', err);
  }
  return {};
}

function saveResultsCache() {
  try {
    localStorage.setItem(RESULTS_CACHE_STORAGE_KEY, JSON.stringify(resultsCache));
  } catch (err) {
    console.warn('Could not save scan results to this browser.', err);
  }
}

let workspaces = loadWorkspaces();
let activeWorkspaceId = workspaces[0].id;
let watchlist = getActiveWorkspace().watchlist;
let currentTimeframe = '1 Day';
let resultsCache = loadResultsCache();  // workspaceId -> {results, statusText} from its last scan

function selectTimeframe(tf) {
  currentTimeframe = tf;
  document.querySelectorAll('.tf-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tf === tf);
  });
  runScan();
}

function renderTabsBar() {
  const bar = document.getElementById('tabsBar');
  bar.innerHTML = '';
  workspaces.forEach(ws => {
    const wrap = document.createElement('span');
    wrap.style.display = 'inline-flex';
    wrap.style.alignItems = 'center';
    wrap.style.marginRight = '6px';

    const btn = document.createElement('button');
    btn.textContent = ws.name;
    btn.className = 'tab-btn' + (ws.id === activeWorkspaceId ? ' active' : '');
    btn.onclick = () => selectWorkspace(ws.id);
    wrap.appendChild(btn);

    if (ws.id === activeWorkspaceId) {
      const renameIcon = document.createElement('span');
      renameIcon.textContent = ' \u270E';
      renameIcon.title = 'Rename this tab';
      renameIcon.className = 'tab-icon';
      renameIcon.onclick = () => renameWorkspace(ws.id);
      wrap.appendChild(renameIcon);

      if (workspaces.length > 1) {
        const delIcon = document.createElement('span');
        delIcon.textContent = ' \u2715';
        delIcon.title = 'Delete this tab';
        delIcon.className = 'tab-icon tab-icon-danger';
        delIcon.onclick = () => deleteWorkspace(ws.id);
        wrap.appendChild(delIcon);
      }
    }
    bar.appendChild(wrap);
  });

  const addBtn = document.createElement('button');
  addBtn.textContent = '+ Add Tab';
  addBtn.className = 'tab-btn tab-add-btn';
  addBtn.onclick = addWorkspace;
  bar.appendChild(addBtn);
}

function selectWorkspace(id) {
  if (id === activeWorkspaceId) return;
  saveWorkspaces();  // persist any pending changes on the tab we're leaving
  activeWorkspaceId = id;
  watchlist = getActiveWorkspace().watchlist;
  renderTabsBar();
  renderWatchlistBar();

  const cached = resultsCache[id];
  if (cached) {
    renderResults(cached.results);
    document.getElementById('status').textContent = cached.statusText + '  (showing last scan for "' + getActiveWorkspace().name + '" - click Run to refresh)';
  } else {
    document.getElementById('resultsBody').innerHTML = '';
    document.getElementById('status').textContent = 'Switched to "' + getActiveWorkspace().name + '" - click Run to scan.';
  }
}

function addWorkspace() {
  const name = prompt('Name for the new tab (e.g. "Spouse Stocks"):', 'New Watchlist');
  if (!name || !name.trim()) return;
  const newWs = {id: 'ws-' + Date.now(), name: name.trim(), watchlist: []};
  workspaces.push(newWs);
  activeWorkspaceId = newWs.id;
  watchlist = newWs.watchlist;
  saveWorkspaces();
  renderTabsBar();
  renderWatchlistBar();
  document.getElementById('resultsBody').innerHTML = '';
  document.getElementById('status').textContent = 'Created tab "' + newWs.name + '" - add some tickers, then click Run.';
}

function renameWorkspace(id) {
  const ws = workspaces.find(w => w.id === id);
  if (!ws) return;
  const newName = prompt('Rename this tab:', ws.name);
  if (!newName || !newName.trim()) return;
  ws.name = newName.trim();
  saveWorkspaces();
  renderTabsBar();
}

function deleteWorkspace(id) {
  if (workspaces.length <= 1) { alert('You need at least one tab.'); return; }
  const ws = workspaces.find(w => w.id === id);
  if (!ws) return;
  const message = 'This will permanently delete the tab "' + ws.name + '" and everything on it. Type the tab name exactly to confirm deletion:';
  const typed = prompt(message);
  if (typed === null) return;  // cancelled
  if (typed.trim() !== ws.name) {
    alert('That did not match "' + ws.name + '" - nothing was deleted.');
    return;
  }
  workspaces = workspaces.filter(w => w.id !== id);
  delete resultsCache[id];
  saveResultsCache();
  if (activeWorkspaceId === id) {
    activeWorkspaceId = workspaces[0].id;
    watchlist = workspaces[0].watchlist;
  }
  saveWorkspaces();
  renderTabsBar();
  renderWatchlistBar();
  const cached = resultsCache[activeWorkspaceId];
  if (cached) {
    renderResults(cached.results);
    document.getElementById('status').textContent = cached.statusText + '  (showing last scan for "' + getActiveWorkspace().name + '" - click Run to refresh)';
  } else {
    document.getElementById('resultsBody').innerHTML = '';
  }
}

function renderWatchlistBar() {
  const bar = document.getElementById('watchlistBar');
  bar.innerHTML = 'Watchlist: ' + watchlist.map(t =>
    `<span>${t}<span class="remove-btn" onclick="removeTicker('${t}')"> &times;</span></span>`
  ).join('  |  ')
    + (watchlist.length ? '' : '<span style="color:#888;">(empty - add a ticker below)</span>');
}

function addTicker() {
  const input = document.getElementById('tickerInput');
  const symbol = input.value.trim().toUpperCase();
  if (!symbol) return;
  if (watchlist.includes(symbol)) { alert(symbol + ' is already on this tab watchlist.'); return; }
  watchlist.push(symbol);
  input.value = '';
  renderWatchlistBar();
  saveWorkspaces();
}

function removeTicker(symbol) {
  watchlist = watchlist.filter(t => t !== symbol);
  renderWatchlistBar();
  saveWorkspaces();
  const row = document.getElementById('row-' + symbol);
  if (row) row.remove();
}

function cell(text, isFlagged) {
  const td = document.createElement('td');
  td.textContent = text;
  if (isFlagged) td.classList.add('cell-flag');
  return td;
}

function actionCell(text) {
  const td = document.createElement('td');
  td.textContent = text;
  td.style.fontWeight = '700';
  td.style.cursor = 'help';
  if (text === 'BUY') {
    td.style.color = '#2e7d32';
    td.title = 'Based on the 4-Hour MACD: it just crossed ABOVE the signal line on the latest 4h bar.';
  } else if (text === 'SELL') {
    td.style.color = '#c62828';
    td.title = 'Based on the 4-Hour MACD: it just crossed BELOW the signal line on the latest 4h bar.';
  } else {
    td.style.color = '#999';
    td.title = 'Based on the 4-Hour MACD: no fresh crossover right now (trend may be continuing, but did not just cross).';
  }
  return td;
}

function tickerCell(ticker) {
  const td = document.createElement('td');
  const a = document.createElement('a');
  a.href = '#';
  a.textContent = ticker;
  a.style.color = '#1565c0';
  a.style.fontWeight = '600';
  a.style.textDecoration = 'underline';
  a.style.cursor = 'pointer';
  a.title = 'Click to open chart';
  a.onclick = (e) => { e.preventDefault(); openChart(ticker); };
  td.appendChild(a);
  return td;
}

function openChart(ticker) {
  const url = '/chart/' + encodeURIComponent(ticker) + '?timeframe=' + encodeURIComponent(currentTimeframe);
  // Fixed to your measured window size at 100% display scaling.
  const w = 1220;
  const h = 735;
  window.open(url, '_blank', `width=${w},height=${h},noopener`);
}

async function runScan() {
  if (watchlist.length === 0) { alert('Add at least one ticker first.'); return; }
  const runBtn = document.getElementById('runBtn');
  const status = document.getElementById('status');
  const timeframe = currentTimeframe;
  const scanForWorkspaceId = activeWorkspaceId;  // remember which tab this scan is for

  runBtn.disabled = true;
  status.textContent = 'Running scan (' + timeframe + ')...';

  try {
    const resp = await fetch('/api/scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({tickers: watchlist, timeframe: timeframe})
    });
    const data = await resp.json();
    const statusText = 'Scan complete - ' + new Date().toLocaleString();

    // Cache these results against the tab they belong to, so switching
    // tabs later - or reloading the page entirely - can restore them
    // instantly without re-scanning.
    resultsCache[scanForWorkspaceId] = {results: data.results, statusText: statusText};
    saveResultsCache();

    // Only update the visible table if the user is still on that same tab
    // (they might have switched away while the scan was in flight).
    if (activeWorkspaceId === scanForWorkspaceId) {
      renderResults(data.results);
      status.textContent = statusText;
    }
  } catch (err) {
    if (activeWorkspaceId === scanForWorkspaceId) {
      status.textContent = 'Error running scan: ' + err;
    }
  } finally {
    runBtn.disabled = false;
  }
}

function renderResults(results) {
  const body = document.getElementById('resultsBody');
  body.innerHTML = '';
  results.forEach(r => {
    const tr = document.createElement('tr');
    tr.id = 'row-' + r.ticker;
    if (r.error) {
      tr.classList.add('error');
      tr.appendChild(tickerCell(r.ticker));
      for (let i = 1; i < COLUMNS.length - 1; i++) tr.appendChild(cell('ERR'));
      tr.appendChild(cell(r.error));
    } else {
      if (r.flagged) tr.classList.add('flagged');
      const nearCols = r.near_cols || [];
      COLUMNS.forEach(([colId, _heading]) => {
        if (colId === 'ticker') {
          tr.appendChild(tickerCell(r.ticker));
        } else if (colId === 'action') {
          tr.appendChild(actionCell(r[colId] !== undefined ? r[colId] : '-'));
        } else {
          tr.appendChild(cell(r[colId] !== undefined ? r[colId] : '-', nearCols.includes(colId)));
        }
      });
    }
    body.appendChild(tr);
  });
}

renderTabsBar();
renderWatchlistBar();

const initialCached = resultsCache[activeWorkspaceId];
if (initialCached) {
  renderResults(initialCached.results);
  document.getElementById('status').textContent = initialCached.statusText + '  (showing last scan for "' + getActiveWorkspace().name + '" - click Run to refresh)';
}
</script>

</body>
</html>
"""


EDUCATION_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Education - MACD &amp; EMA Basics</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #fafafa; color: #222; }
  .wrap { max-width: 820px; margin: 0 auto; padding: 24px 28px 60px; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  h2 { font-size: 18px; margin-top: 34px; color: #0d47a1; border-bottom: 2px solid #e3ecf7; padding-bottom: 6px; }
  h3 { font-size: 15px; margin-top: 20px; color: #1565c0; }
  p, li { line-height: 1.6; font-size: 14.5px; }
  .top-nav { margin-bottom: 18px; }
  .top-nav a { color: #1565c0; text-decoration: underline; font-size: 13px; }
  .disclaimer { background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px; padding: 14px 16px;
                font-size: 13px; color: #6b5400; margin: 18px 0; }
  .example-box { background: #eef4fc; border-left: 4px solid #1565c0; padding: 10px 16px; margin: 14px 0;
                 border-radius: 0 6px 6px 0; }
  .caution-box { background: #fdecea; border-left: 4px solid #c62828; padding: 10px 16px; margin: 14px 0;
                 border-radius: 0 6px 6px 0; }
  .diagram-box { text-align: center; margin: 18px 0; }
  .diagram-caption { font-size: 12px; color: #777; margin-top: 6px; font-style: italic; }
  code { background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 13px; }
  ul { padding-left: 22px; }
  .formula { background: #f4f6f8; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 14px;
             font-family: Consolas, monospace; font-size: 13.5px; margin: 8px 0; }
</style>
</head>
<body>
<div class="wrap">

<div class="top-nav"><a href="/">← Back to Dashboard</a></div>

<h1>Understanding MACD &amp; EMA Signals</h1>
<p style="color:#666; margin-top:0;">A plain-language reference for reading the indicators on your dashboard.</p>

<div class="disclaimer">
<strong>This is general, educational information only — not financial advice.</strong>
It explains what these indicators traditionally mean and how traders commonly use them.
It is not a recommendation to buy or sell any specific stock, and it can't account for your
personal financial situation, risk tolerance, or goals. Markets can move against any signal,
including the ones described below.
</div>

<h2>Definitions</h2>
<div class="formula">
MACD&nbsp;&nbsp;=&nbsp;&nbsp;EMA(12) &minus; EMA(26)<br>
Signal Line&nbsp;&nbsp;=&nbsp;&nbsp;EMA(9) of the MACD line<br>
Histogram&nbsp;&nbsp;=&nbsp;&nbsp;MACD &minus; Signal Line
</div>
<p>
These three numbers are exactly what your dashboard's <strong>MACD</strong>, <strong>Signal</strong>, and
<strong>Hist</strong> columns show for each ticker.
</p>

<h2>Reading a MACD Crossover</h2>
<h3>Bullish crossover — often watched as a potential ENTRY signal</h3>
<p>
This happens when the MACD line crosses <em>above</em> the signal line — upward momentum gaining strength.
On your dashboard, this shows up as the histogram flipping from negative to positive.
</p>

<h3>Bearish crossover — often watched as a potential EXIT signal</h3>
<p>
This happens when the MACD line crosses <em>below</em> the signal line — upward momentum weakening or
reversing. Histogram bars shrinking while still positive is what your dashboard flags as
"MACD gap narrowing" — a heads-up that a bearish crossover may be approaching, even before it's confirmed.
</p>

<div class="diagram-box">
<svg viewBox="0 0 720 380" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:720px; background:#fff; border:1px solid #ddd; border-radius:8px;">
<rect x="50" y="20" width="650" height="190" fill="#fbfcfe" stroke="#e3e3e3"/>
<rect x="50" y="224" width="650" height="130" fill="#fbfcfe" stroke="#e3e3e3"/>
<polyline points="50.0,187.5 61.0,172.1 72.0,156.8 83.1,141.9 94.1,127.4 105.1,113.6 116.1,100.6 127.1,88.4 138.1,77.3 149.2,67.3 160.2,58.5 171.2,51.0 182.2,44.9 193.2,40.3 204.2,37.0 215.3,35.3 226.3,35.0 237.3,36.1 248.3,38.6 259.3,42.4 270.3,47.4 281.4,53.7 292.4,61.0 303.4,69.2 314.4,78.2 325.4,87.9 336.4,98.1 347.5,108.7 358.5,119.5 369.5,130.3 380.5,141.0 391.5,151.5 402.5,161.5 413.6,170.9 424.6,179.6 435.6,187.4 446.6,194.2 457.6,199.9 468.6,204.4 479.7,207.7 490.7,209.5 501.7,210.0 512.7,209.0 523.7,206.6 534.7,202.7 545.8,197.4 556.8,190.7 567.8,182.6 578.8,173.3 589.8,162.8 600.8,151.2 611.9,138.6 622.9,125.2 633.9,111.0 644.9,96.3 655.9,81.3 666.9,65.9 678.0,50.5 689.0,35.1 700.0,20.0" fill="none" stroke="#333" stroke-width="2" />
<text x="54" y="36" font-size="12" fill="#333" font-weight="600">Price (illustrative)</text>
<line x1="50" y1="289.0" x2="700" y2="289.0" stroke="#bbb" stroke-dasharray="2,2"/>
<rect x="46.8" y="272.2" width="6.5" height="16.8" fill="#26a69a" opacity="0.55" />
<rect x="57.8" y="269.1" width="6.5" height="19.9" fill="#26a69a" opacity="0.55" />
<rect x="68.8" y="266.3" width="6.5" height="22.7" fill="#26a69a" opacity="0.55" />
<rect x="79.8" y="263.7" width="6.5" height="25.3" fill="#26a69a" opacity="0.55" />
<rect x="90.8" y="261.4" width="6.5" height="27.6" fill="#26a69a" opacity="0.55" />
<rect x="101.8" y="259.5" width="6.5" height="29.5" fill="#26a69a" opacity="0.55" />
<rect x="112.9" y="257.9" width="6.5" height="31.1" fill="#26a69a" opacity="0.55" />
<rect x="123.9" y="256.8" width="6.5" height="32.2" fill="#26a69a" opacity="0.55" />
<rect x="134.9" y="256.0" width="6.5" height="33.0" fill="#26a69a" opacity="0.55" />
<rect x="145.9" y="255.6" width="6.5" height="33.4" fill="#26a69a" opacity="0.55" />
<rect x="156.9" y="255.7" width="6.5" height="33.3" fill="#26a69a" opacity="0.55" />
<rect x="167.9" y="256.1" width="6.5" height="32.9" fill="#26a69a" opacity="0.55" />
<rect x="179.0" y="257.0" width="6.5" height="32.0" fill="#26a69a" opacity="0.55" />
<rect x="190.0" y="258.2" width="6.5" height="30.8" fill="#26a69a" opacity="0.55" />
<rect x="201.0" y="259.9" width="6.5" height="29.1" fill="#26a69a" opacity="0.55" />
<rect x="212.0" y="261.9" width="6.5" height="27.1" fill="#26a69a" opacity="0.55" />
<rect x="223.0" y="264.2" width="6.5" height="24.8" fill="#26a69a" opacity="0.55" />
<rect x="234.0" y="266.8" width="6.5" height="22.2" fill="#26a69a" opacity="0.55" />
<rect x="245.1" y="269.7" width="6.5" height="19.3" fill="#26a69a" opacity="0.55" />
<rect x="256.1" y="272.9" width="6.5" height="16.1" fill="#26a69a" opacity="0.55" />
<rect x="267.1" y="276.2" width="6.5" height="12.8" fill="#26a69a" opacity="0.55" />
<rect x="278.1" y="279.7" width="6.5" height="9.3" fill="#26a69a" opacity="0.55" />
<rect x="289.1" y="283.3" width="6.5" height="5.7" fill="#26a69a" opacity="0.55" />
<rect x="300.1" y="287.0" width="6.5" height="2.0" fill="#26a69a" opacity="0.55" />
<rect x="311.2" y="289.0" width="6.5" height="1.7" fill="#ef5350" opacity="0.55" />
<rect x="322.2" y="289.0" width="6.5" height="5.4" fill="#ef5350" opacity="0.55" />
<rect x="333.2" y="289.0" width="6.5" height="9.0" fill="#ef5350" opacity="0.55" />
<rect x="344.2" y="289.0" width="6.5" height="12.5" fill="#ef5350" opacity="0.55" />
<rect x="355.2" y="289.0" width="6.5" height="15.9" fill="#ef5350" opacity="0.55" />
<rect x="366.2" y="289.0" width="6.5" height="19.1" fill="#ef5350" opacity="0.55" />
<rect x="377.3" y="289.0" width="6.5" height="22.0" fill="#ef5350" opacity="0.55" />
<rect x="388.3" y="289.0" width="6.5" height="24.6" fill="#ef5350" opacity="0.55" />
<rect x="399.3" y="289.0" width="6.5" height="27.0" fill="#ef5350" opacity="0.55" />
<rect x="410.3" y="289.0" width="6.5" height="29.0" fill="#ef5350" opacity="0.55" />
<rect x="421.3" y="289.0" width="6.5" height="30.7" fill="#ef5350" opacity="0.55" />
<rect x="432.3" y="289.0" width="6.5" height="31.9" fill="#ef5350" opacity="0.55" />
<rect x="443.4" y="289.0" width="6.5" height="32.8" fill="#ef5350" opacity="0.55" />
<rect x="454.4" y="289.0" width="6.5" height="33.3" fill="#ef5350" opacity="0.55" />
<rect x="465.4" y="289.0" width="6.5" height="33.4" fill="#ef5350" opacity="0.55" />
<rect x="476.4" y="289.0" width="6.5" height="33.0" fill="#ef5350" opacity="0.55" />
<rect x="487.4" y="289.0" width="6.5" height="32.3" fill="#ef5350" opacity="0.55" />
<rect x="498.4" y="289.0" width="6.5" height="31.2" fill="#ef5350" opacity="0.55" />
<rect x="509.5" y="289.0" width="6.5" height="29.6" fill="#ef5350" opacity="0.55" />
<rect x="520.5" y="289.0" width="6.5" height="27.7" fill="#ef5350" opacity="0.55" />
<rect x="531.5" y="289.0" width="6.5" height="25.5" fill="#ef5350" opacity="0.55" />
<rect x="542.5" y="289.0" width="6.5" height="22.9" fill="#ef5350" opacity="0.55" />
<rect x="553.5" y="289.0" width="6.5" height="20.1" fill="#ef5350" opacity="0.55" />
<rect x="564.5" y="289.0" width="6.5" height="17.0" fill="#ef5350" opacity="0.55" />
<rect x="575.6" y="289.0" width="6.5" height="13.7" fill="#ef5350" opacity="0.55" />
<rect x="586.6" y="289.0" width="6.5" height="10.3" fill="#ef5350" opacity="0.55" />
<rect x="597.6" y="289.0" width="6.5" height="6.7" fill="#ef5350" opacity="0.55" />
<rect x="608.6" y="289.0" width="6.5" height="3.0" fill="#ef5350" opacity="0.55" />
<rect x="619.6" y="288.3" width="6.5" height="0.7" fill="#26a69a" opacity="0.55" />
<rect x="630.6" y="284.6" width="6.5" height="4.4" fill="#26a69a" opacity="0.55" />
<rect x="641.7" y="281.0" width="6.5" height="8.0" fill="#26a69a" opacity="0.55" />
<rect x="652.7" y="277.4" width="6.5" height="11.6" fill="#26a69a" opacity="0.55" />
<rect x="663.7" y="274.0" width="6.5" height="15.0" fill="#26a69a" opacity="0.55" />
<rect x="674.7" y="270.8" width="6.5" height="18.2" fill="#26a69a" opacity="0.55" />
<rect x="685.7" y="267.8" width="6.5" height="21.2" fill="#26a69a" opacity="0.55" />
<rect x="696.8" y="265.1" width="6.5" height="23.9" fill="#26a69a" opacity="0.55" />
<polyline points="50.0,327.3 61.0,322.5 72.0,317.2 83.1,311.6 94.1,305.7 105.1,299.6 116.1,293.4 127.1,287.1 138.1,280.9 149.2,274.7 160.2,268.7 171.2,263.0 182.2,257.6 193.2,252.6 204.2,248.0 215.3,244.0 226.3,240.4 237.3,237.5 248.3,235.3 259.3,233.6 270.3,232.7 281.4,232.5 292.4,232.9 303.4,234.1 314.4,235.9 325.4,238.4 336.4,241.5 347.5,245.2 358.5,249.5 369.5,254.2 380.5,259.3 391.5,264.8 402.5,270.7 413.6,276.7 424.6,282.9 435.6,289.2 446.6,295.4 457.6,301.6 468.6,307.6 479.7,313.4 490.7,318.9 501.7,324.1 512.7,328.8 523.7,333.0 534.7,336.7 545.8,339.7 556.8,342.2 567.8,344.0 578.8,345.1 589.8,345.5 600.8,345.3 611.9,344.3 622.9,342.6 633.9,340.3 644.9,337.4 655.9,333.9 666.9,329.8 678.0,325.2 689.0,320.1 700.0,314.7" fill="none" stroke="#1565c0" stroke-width="2" />
<polyline points="50.0,344.1 61.0,342.3 72.0,339.9 83.1,336.9 94.1,333.3 105.1,329.1 116.1,324.4 127.1,319.3 138.1,313.9 149.2,308.1 160.2,302.1 171.2,295.9 182.2,289.6 193.2,283.4 204.2,277.2 215.3,271.1 226.3,265.3 237.3,259.7 248.3,254.5 259.3,249.8 270.3,245.5 281.4,241.8 292.4,238.6 303.4,236.1 314.4,234.2 325.4,233.0 336.4,232.5 347.5,232.7 358.5,233.6 369.5,235.1 380.5,237.3 391.5,240.2 402.5,243.7 413.6,247.7 424.6,252.2 435.6,257.2 446.6,262.6 457.6,268.3 468.6,274.3 479.7,280.4 490.7,286.6 501.7,292.9 512.7,299.2 523.7,305.3 534.7,311.2 545.8,316.8 556.8,322.1 567.8,327.0 578.8,331.4 589.8,335.3 600.8,338.6 611.9,341.3 622.9,343.3 633.9,344.7 644.9,345.4 655.9,345.4 666.9,344.8 678.0,343.4 689.0,341.3 700.0,338.6" fill="none" stroke="#ff9800" stroke-width="2" />
<text x="54" y="240" font-size="12" fill="#1565c0" font-weight="600">MACD</text>
<text x="120" y="240" font-size="12" fill="#ff9800" font-weight="600">Signal</text>
<line x1="314.4" y1="212.0" x2="314.4" y2="222.0" stroke="#c62828" stroke-width="1" stroke-dasharray="3,3" opacity="0.6" />
<circle cx="314.4" cy="235.9" r="4" fill="#c62828" />
<text x="314.4" y="224.0" font-size="11" font-weight="700" fill="#c62828" text-anchor="middle">SELL</text>
<line x1="622.9" y1="212.0" x2="622.9" y2="222.0" stroke="#2e7d32" stroke-width="1" stroke-dasharray="3,3" opacity="0.6" />
<circle cx="622.9" cy="342.6" r="4" fill="#2e7d32" />
<text x="622.9" y="224.0" font-size="11" font-weight="700" fill="#2e7d32" text-anchor="middle">BUY</text>
</svg>
<div class="diagram-caption">Illustrative example with synthetic data — not a real stock or real prices. Shows a bearish crossover (SELL) followed later by a bullish crossover (BUY).</div>
</div>

<h2>Confirmation: Waiting for the Move to Prove Itself</h2>
<p>
A crossover on its own is often just the first hint — many traders wait for "confirmation" before acting:
the MACD line clearly separating from the signal line, or the histogram growing for a few bars in a row,
rather than immediately reversing again. Acting only on the very first tick of a crossover is more prone to
false signals than waiting a bar or two for the move to hold.
</p>

<h2>Divergence: A Different Kind of Signal</h2>
<p>
Divergence is when <strong>price</strong> and <strong>MACD</strong> disagree with each other — which can be
an early warning that the current trend is losing steam, even before a crossover happens:
</p>
<ul>
  <li><strong>Bearish divergence:</strong> price makes a new high, but MACD makes a <em>lower</em> high than
      its previous peak. Suggests the rally's momentum is fading even though price is still climbing.</li>
  <li><strong>Bullish divergence:</strong> price makes a new low, but MACD makes a <em>higher</em> low than
      its previous trough. Suggests selling pressure is fading even though price is still falling.</li>
</ul>
<p>
Divergence isn't shown directly as a column on your dashboard today — it takes comparing MACD's shape across
multiple recent peaks/troughs by eye on the chart window for each ticker.
</p>

<h2>Using EMAs to Confirm the Signal</h2>
<p>
MACD crossovers happen often, and not every one leads to a sustained move. A common approach is to only
trust a MACD signal when it agrees with the broader trend, which is where EMA10/50/100/150/200 come in:
</p>
<ul>
  <li><strong>Price above the longer EMAs (150/200)</strong> generally suggests a broader uptrend — a
      bullish MACD crossover here is often considered a higher-quality entry signal than the same crossover
      happening while price is below those EMAs.</li>
  <li><strong>Price below the longer EMAs</strong> generally suggests a broader downtrend — a bearish MACD
      crossover here may carry more weight as an exit/avoid-entry signal.</li>
  <li><strong>Shorter EMAs (10/50) crossing the longer ones</strong> is itself a separate trend-change signal
      some traders use alongside MACD for extra confirmation.</li>
</ul>

<h2>Important Caveats</h2>
<div class="caution-box">
<ul style="margin:0;">
  <li><strong>MACD is a lagging indicator</strong> — built from moving averages of past prices, so it
      confirms a move after it's already begun, not before.</li>
  <li><strong>False signals ("whipsaws") are common in sideways/choppy markets</strong> — MACD can cross
      back and forth repeatedly without any real trend developing.</li>
  <li><strong>No indicator works in isolation</strong> — volume, overall market conditions, news, and
      risk management (position sizing, stop-losses) all matter alongside any technical signal.</li>
  <li><strong>Past patterns don't guarantee future results.</strong></li>
</ul>
</div>

<h2>How This Maps to Your Dashboard</h2>
<p>
Each ticker's <strong>Bullish/Bearish</strong> column reflects the current MACD vs. signal line position, and
the <strong>Comments</strong> column flags when a crossover looks close or price is near a specific EMA. Cells
turn red when they're part of what triggered a flagged comment.
</p>

<h2>Further Reading</h2>
<p>
For a deeper technical reference: <a href="https://www.investopedia.com/terms/m/macd.asp" target="_blank" rel="noopener">Investopedia — Moving Average Convergence/Divergence (MACD)</a>.
</p>

</div>
</body>
</html>
"""


@app.route("/education")
@login_required
def education():
    return render_template_string(EDUCATION_TEMPLATE)


@app.route("/")
@login_required
def index():
    return render_template_string(
        PAGE_TEMPLATE,
        timeframes=list(TIMEFRAME_OPTIONS.keys()),
        columns=COLUMNS,
        columns_json=[[c[0], c[1]] for c in COLUMNS],
        watchlist_json=DEFAULT_WATCHLIST,
    )


CHART_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ ticker }} chart</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; margin: 14px; background: #fafafa; color: #222; }
  h1 { font-size: 18px; margin: 0 0 10px 0; }
  .controls { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
  .tf-group { display: flex; gap: 6px; }
  button.tf-btn { padding: 6px 12px; font-size: 13px; cursor: pointer; border: 1px solid #999;
                  background: #fff; border-radius: 4px; min-width: 48px; font-weight: 600; }
  button.tf-btn.active { background: #1565c0; color: white; border-color: #1565c0; }
  #commentBadge { padding: 6px 12px; border-radius: 4px; font-size: 13px; font-weight: 600; }
  #commentBadge.flagged { background: #fdecea; color: #c62828; border: 1px solid #ef9a9a; }
  #commentBadge.clear { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
  #status { font-size: 12px; color: #666; margin-bottom: 8px; }
  #chart { width: 100%; }
  .legend-note { font-size: 12px; color: #666; margin-top: 4px; }
</style>
</head>
<body>

<h1>{{ ticker }} - Price (Candlestick) + EMA, with MACD below</h1>

<div class="controls">
  <span>Timeframe:</span>
  <div class="tf-group" id="tfGroup">
    <button class="tf-btn" data-tf="1 Month" onclick="loadChart('1 Month')">1M</button>
    <button class="tf-btn" data-tf="1 Week" onclick="loadChart('1 Week')">1W</button>
    <button class="tf-btn" data-tf="1 Day" onclick="loadChart('1 Day')">1D</button>
    <button class="tf-btn" data-tf="4 Hour" onclick="loadChart('4 Hour')">4h</button>
  </div>
  <span id="commentBadge">-</span>
</div>

<div id="status">Loading...</div>
<div id="chart"></div>
<div class="legend-note">EMA10 pink, EMA50 orange, EMA100 green, EMA150 blue, EMA200 purple. MACD line blue, Signal line orange, histogram bars green/red. Click a point to drop a vertical marker line across both charts, then drag it to move it.</div>

<script>
const TICKER = {{ ticker_json | safe }};
const EMA_COLORS = {
  'EMA10': '#e91e63',
  'EMA50': '#ff9800',
  'EMA100': '#43a047',
  'EMA150': '#1e88e5',
  'EMA200': '#8e24aa'
};

async function loadChart(timeframe) {
  document.querySelectorAll('.tf-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tf === timeframe);
  });
  const status = document.getElementById('status');
  status.textContent = 'Loading ' + TICKER + ' (' + timeframe + ')...';

  try {
    const resp = await fetch('/api/chart-data/' + encodeURIComponent(TICKER) + '?timeframe=' + encodeURIComponent(timeframe));
    const data = await resp.json();
    if (data.error) {
      status.textContent = 'Error: ' + data.error;
      return;
    }
    renderChart(data);
    const badge = document.getElementById('commentBadge');
    badge.textContent = data.comment;
    badge.classList.toggle('flagged', !!data.flagged);
    badge.classList.toggle('clear', !data.flagged);
    status.textContent = TICKER + ' - ' + timeframe + ' - last updated ' + new Date().toLocaleTimeString();
  } catch (err) {
    status.textContent = 'Error loading chart: ' + err;
  }
}


function renderChart(data) {
  const traces = [];

  traces.push({
    type: 'candlestick',
    x: data.dates,
    open: data.open,
    high: data.high,
    low: data.low,
    close: data.close,
    name: TICKER,
    yaxis: 'y',
    increasing: {line: {color: '#26a69a'}},
    decreasing: {line: {color: '#ef5350'}}
  });

  // data.emas is now an ordered list of [name, values] pairs (not a dict),
  // so it plots in the same EMA10 -> EMA50 -> EMA100 -> EMA150 -> EMA200
  // order the backend sends, instead of being re-sorted alphabetically.
  data.emas.forEach(([emaKey, emaValues]) => {
    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: data.dates,
      y: emaValues,
      name: emaKey,
      yaxis: 'y',
      line: {color: EMA_COLORS[emaKey] || '#999', width: 1.5}
    });
  });

  traces.push({
    type: 'scatter', mode: 'lines', x: data.dates, y: data.macd,
    name: 'MACD', yaxis: 'y2', line: {color: '#1565c0', width: 1.5}
  });
  traces.push({
    type: 'scatter', mode: 'lines', x: data.dates, y: data.signal,
    name: 'Signal', yaxis: 'y2', line: {color: '#ff9800', width: 1.5}
  });
  const histColors = data.hist.map(v => (v >= 0 ? '#26a69a' : '#ef5350'));
  traces.push({
    type: 'bar', x: data.dates, y: data.hist,
    name: 'Histogram', yaxis: 'y2', marker: {color: histColors}
  });

  const layout = {
    margin: {t: 20, b: 60, l: 55, r: 20},
    showlegend: true,
    legend: {orientation: 'h', y: 1.05},
    xaxis: {
      rangeslider: {visible: false},
      type: 'category',
      anchor: 'y2',
      nticks: 14,          // cap how many date labels are drawn, avoids clutter
      tickangle: -40,
      tickfont: {size: 10}
    },
    yaxis: {
      domain: [0.35, 1],
      title: 'Price'
    },
    yaxis2: {
      domain: [0, 0.28],
      title: 'MACD'
    },
    height: 750,
    shapes: []   // crosshair line(s) get added here on click, see below
  };

  Plotly.newPlot('chart', traces, layout, {
    responsive: true,
    displaylogo: false,
    edits: {shapePosition: true}   // lets the user drag shapes (our crosshair line) with the mouse
  }).then(gd => {
      // Click a point on either subplot -> draw one vertical dotted line
      // that spans both the price chart and the MACD panel underneath.
      // editable:true on the shape + edits.shapePosition above is what
      // makes it possible to then grab that same line and drag it left/
      // right afterward, instead of only being able to re-click to move it.
      gd.on('plotly_click', function (evt) {
        if (!evt.points || evt.points.length === 0) return;
        const xVal = evt.points[0].x;
        Plotly.relayout(gd, {
          shapes: [{
            type: 'line',
            xref: 'x',
            yref: 'paper',   // paper coords (0-1) span the WHOLE figure,
            y0: 0, y1: 1,     // so this line crosses both subplots at once
            x0: xVal, x1: xVal,
            line: {color: '#555555', width: 2, dash: 'dot'},
            editable: true
          }]
        });
      });
    });
}

loadChart('{{ timeframe }}');
</script>

</body>
</html>
"""


@app.route("/chart/<ticker>")
@login_required
def chart_page(ticker):
    timeframe = request.args.get("timeframe", "1 Day")
    if timeframe not in TIMEFRAME_OPTIONS:
        timeframe = "1 Day"
    return render_template_string(
        CHART_PAGE_TEMPLATE,
        ticker=ticker.upper(),
        ticker_json=f'"{ticker.upper()}"',
        timeframe=timeframe,
    )


if __name__ == "__main__":
    # host="0.0.0.0" + reading $PORT lets this run unmodified on free hosting
    # platforms (Render, Railway, Fly.io, etc). Locally, PORT is unset so it
    # just falls back to 127.0.0.1:5000 as before.
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    print(f"Starting server... open http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port} in your browser.")
    app.run(host=host, port=port, debug=False)
