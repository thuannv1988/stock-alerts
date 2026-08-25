# Stock Alert Dashboard (MACD + EMA)

A Python tool that checks your watchlist for:

- **MACD signal** (bullish/bearish) and whether it's **close to a crossover**
- **How close price is to EMA10, EMA50, EMA100, EMA150, EMA200** (as a %, and above/below)
- Per-stock candlestick charts with EMA overlays and a linked MACD panel

No paid data feed required — it pulls free price history from Yahoo
Finance via the `yfinance` library.

There are two ways to use it:
- **`web_stock_alerts.py`** — a browser-based dashboard (recommended, this is what's deployed)
- **`stock_alerts.py`** — a simple command-line version

---

## 1. Setup in VS Code (local use)

1. **Install Python** (3.9+): https://www.python.org/downloads/
2. **Open this folder** in VS Code, open a terminal (`` Ctrl+` ``).
3. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - Mac/Linux: `source venv/bin/activate`
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. Run it:
   ```bash
   python web_stock_alerts.py
   ```
   Open **http://127.0.0.1:5000**.

---

## 2. Login protection

The dashboard requires an email + password before anyone can see it.

1. Generate a password hash locally (your plaintext password is never
   saved anywhere):
   ```bash
   python gen_password_hash.py
   ```
2. Set three environment variables — `ADMIN_EMAIL`, `ADMIN_PASSWORD_HASH`
   (the hash from step 1, not your plaintext password), and `SECRET_KEY`
   (any long random string, e.g. `python -c "import secrets; print(secrets.token_hex(32))"`).

Locally (PowerShell):
```powershell
$env:ADMIN_EMAIL="you@example.com"
$env:ADMIN_PASSWORD_HASH="scrypt:32768:8:1$..."
$env:SECRET_KEY="paste the random string here"
python web_stock_alerts.py
```

---

## 3. Multiple tabs (e.g. separate watchlists for you and your spouse)

You can create separate named tabs, each with its own independent
watchlist — handy for keeping your stocks and someone else's stocks apart
on the same dashboard.

- Click **+ Add Tab** to create a new one; you'll be asked to name it
  (e.g. "Wife's Stocks").
- Click a tab to switch to it — its own watchlist and results show up.
- While a tab is active, click the pencil (✎) next to its name to rename
  it, or the ✕ to delete it (you always need at least one tab).
- Each tab's watchlist is saved in this browser's local storage, tied to
  that tab's name — so switching tabs, logging out, or closing the
  browser doesn't lose anything.

Note: like the watchlist itself, tabs are saved per-browser/per-device,
not synced across devices — if you check the site from a different
computer or phone, you'll see fresh default tabs there, not the ones from
this browser.

---

## 4. Deployed on Render (free)

This app is set up to deploy on Render's free web service tier:

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn web_stock_alerts:app` (matches `Procfile`)
- **Environment variables:** `ADMIN_EMAIL`, `ADMIN_PASSWORD_HASH`, `SECRET_KEY`
- **Your subdomain:** whatever you named the service, e.g.
  `https://your-chosen-name.onrender.com`

**Free tier behavior:** the service spins down after ~15 minutes of no
traffic, and the next request after that takes 30-60 seconds to wake back
up. A `/healthz` endpoint (no login required) is included specifically so
a free keep-alive pinger (e.g. UptimeRobot) can ping it every 5-10
minutes and prevent that sleep entirely.

To point a domain you own at this instead of the `.onrender.com`
subdomain, add it under the service's Settings → Custom Domains in the
Render dashboard, then update your DNS records with your domain
registrar as Render instructs. Connecting a custom domain is free;
owning the domain name itself is not (that's paid via whatever registrar
you buy it from, e.g. Namecheap, GoDaddy).

---

## 5. Command-line version: `stock_alerts.py`

A simpler, no-browser alternative that prints a text report:
```bash
python stock_alerts.py AAPL NVDA
```

---

## 6. Desktop GUI alternative: `gui_stock_alerts.py`

A Tkinter desktop-window version with the same layout as the web
dashboard. Requires a working Tcl/Tk installation bundled with Python —
if that gives you trouble, use the web dashboard instead.

```bash
python gui_stock_alerts.py
```

---

## Notes & caveats

- Data comes from Yahoo Finance via `yfinance`. It's free but
  **unofficial and rate-limited** — if you scan a very large watchlist
  very frequently you may get temporary errors; just re-run.
- This is a **technical-analysis helper, not investment advice**.
- All thresholds (EMA proximity %, MACD "near" sensitivity, lookback
  period, history length) are configurable constants near the top of
  each script.
