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
    ("trend", "Bullish/Bearish"), ("comments", "Comments"),
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


def analyze_ticker(ticker: str, timeframe_cfg: dict) -> dict:
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

    results = [analyze_ticker(t, timeframe_cfg) for t in tickers]
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
      <th class="group-header" colspan="5">MACD Indicator</th>
    </tr>
    <tr>
      <th>EMA10</th><th>Delta %</th>
      <th>EMA50</th><th>Delta %</th>
      <th>EMA100</th><th>Delta %</th>
      <th>EMA150</th><th>Delta %</th>
      <th>EMA200</th><th>Delta %</th>
      <th>MACD</th><th>Signal</th><th>Hist</th>
      <th>Bullish/Bearish</th><th class="comments">Comments</th>
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
