# Daily Stock Picks Bot + Dashboard

A Python bot that screens stocks daily for MA20 crossovers with momentum, volume, and sentiment confirmation. GitHub Actions runs it on a cron, commits the output JSON (with 60-day OHLC history), and Cloudflare Pages / Vercel serves the dashboard with interactive candlestick charts.

## Architecture

```
GitHub Actions (cron)
    ↓ runs daily_picks_bot.py
    ↓ writes public/picks.json
    ↓ commits to repo
    ↓
Cloudflare Pages / Vercel
    ↓ serves public/ as static site
    ↓ index.html fetches picks.json
```

## File layout

```
.
├── .github/workflows/daily-picks.yml   ← cron job
├── daily_picks_bot.py                  ← the screener
├── public/
│   ├── index.html                      ← the dashboard
│   └── picks.json                      ← updated daily by cron
└── README.md
```

## Setup

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin git@github.com:YOURNAME/daily-picks.git
git push -u origin main
```

### 2. Enable Actions write permission

In GitHub: **Settings → Actions → General → Workflow permissions → Read and write permissions**. This lets the cron commit `picks.json` back to the repo.

### 3. Trigger first run

**Actions** tab → **Daily Stock Picks** → **Run workflow**. Verify `public/picks.json` updates.

### 4. Deploy the dashboard

**Cloudflare Pages:**
1. Connect your GitHub repo at https://dash.cloudflare.com/?to=/:account/pages
2. Build settings:
   - Build command: *(leave blank)*
   - Build output directory: `public`
3. Deploy. You'll get a `*.pages.dev` URL.

**Vercel:**
1. Import the GitHub repo at https://vercel.com/new
2. Framework preset: **Other**
3. Output directory: `public`
4. Deploy. You'll get a `*.vercel.app` URL.

Both auto-redeploy whenever the cron commits a new `picks.json`.

### 5. Schedule

Default cron is `0 9 * * 1-5` (5pm SGT weekdays, after SGX close). Edit `.github/workflows/daily-picks.yml` for a different time. Cron is in **UTC**.

| Time you want | Cron |
|---|---|
| 5pm SGT (after SGX close) | `0 9 * * 1-5` |
| 5pm ET (after US close) | `0 21 * * 1-5` |
| 6am SGT (premarket prep) | `0 22 * * 0-4` |

### 6. Customize the watchlist

Edit `US_DEFAULT` and `SGX_DEFAULT` in `daily_picks_bot.py`, or pass `--watchlist tickers.txt` to use a file (one ticker per line, `#` for comments).

## Local testing

```bash
pip install yfinance pandas numpy
python daily_picks_bot.py --market both --output public/picks.json
# Open public/index.html in a browser (use a local server for fetch to work):
python -m http.server -d public 8000
# Visit http://localhost:8000
```

## Upgrades to consider

- **Sentiment**: replace the keyword-based `score_headline` with FinBERT (HuggingFace) or a StockTwits/Reddit feed via PRAW for real social signal
- **Notifications**: add a Telegram or Discord webhook step in the workflow to ping you when picks land
- **Backtesting**: validate the signal combination on 1–2 years of history before trusting it with capital
- **Position sizing**: the bot currently ranks but doesn't size; add stop-loss levels (2× ATR or recent swing low) and target sizes

## Disclaimer

Not financial advice. This is engineering scaffolding for screening, not a recommendation system. Verify against your own risk rules before trading. Past patterns don't guarantee future returns.
