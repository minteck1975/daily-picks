# The Tape — Daily Stock Picks

A trend-following screener for US large-caps that runs nightly on GitHub Actions, scoring stocks across 12+ signals (technicals, sentiment, insider activity, market context) and publishing the top 3 picks to a public dashboard with full trade setups and a running track record.

**Live dashboard:** edit this URL to point at your deployment → `https://YOUR-USER.github.io/daily-picks/`

---

## What it does

Each weekday after market close, the bot:

1. **Scans 100 large-cap US stocks** — full sector mix, heavy on tech/AI
2. **Filters for established uptrends** — price > MA20 > MA50 with MA20 rising
3. **Enriches the top ~12 candidates** with sentiment, insider data, weekly trend, earnings calendar, and sector classification
4. **Selects up to 3 picks** — at most one per sector, none with earnings in the next 3 days, none with weekly-down trends
5. **Computes a full trade setup per pick** — ATR-based stop, nearest-resistance target, risk-reward ratio
6. **Snapshots the broader regime** — sector breadth, VIX, junk-credit (HYG), risk-on/off label
7. **Logs picks to a persistent track record** that auto-computes returns at +1 / +5 / +30 days
8. **Commits everything back to the repo**, triggering a Pages redeploy

---

## The signal stack

| Signal | What it catches | Weight |
|---|---|---|
| Trend (price > MA20 > MA50, MA20 rising) | Established uptrends | Required |
| RSI(14) 50–75 rising | Healthy momentum, not overbought | +1.0 |
| MACD histogram positive + expanding | Accelerating momentum | +1.0 |
| Volume ≥ 1.5× 20d avg | Institutional participation | +1.5 |
| Relative strength vs SPY +30d > 3% | Trend leaders, not followers | +2.0 |
| Weekly trend up (10w MA + rising) | Multi-timeframe alignment | +1.5 |
| Within 15% of 52-week high | Long-term uptrend intact | +1.0 |
| ATR 1–6% of price | Reasonable volatility | filter |
| Consolidation breakout (≤5d ago) | High-quality entry pattern | +2.0 |
| StockTwits ≥70% bullish (5+ tagged) | Strong crowd sentiment | +2.0 |
| Reddit chatter (4 major subs, 7d) | Retail discussion volume | +1.5 |
| News sentiment via FinBERT | Real NLP on headlines + summaries | ±2.0 |
| Insider buys ≥$1M (30d via OpenInsider) | Smart money conviction | +2.0 |

Final filters: skip earnings within 3 days, exclude weekly downtrends, max 1 pick per GICS sector.

---

## Architecture

```
┌─────────────────────────────────────┐
│  GitHub Actions cron (5pm SGT)      │
│  ─ runs daily_picks_bot.py          │
│  ─ reads + updates history.json     │
│  ─ writes picks.json                │
│  ─ commits both back to main        │
└──────────────┬──────────────────────┘
               │ commit triggers
               ▼
┌─────────────────────────────────────┐
│  Deploy Pages workflow              │
│  ─ uploads public/ to GH Pages      │
└──────────────┬──────────────────────┘
               │ ~60s
               ▼
┌─────────────────────────────────────┐
│  GitHub Pages (CDN-cached)          │
│  https://USER.github.io/REPO/       │
│  ─ index.html fetches picks.json    │
│  ─ Track tab fetches history.json   │
└─────────────────────────────────────┘
```

### Data flow inside `daily_picks_bot.py`

```
1. Load yfinance benchmarks (SPY, ^VIX, HYG, 11 sector ETFs)
2. Initialize FinBERT model (one-time per run)
3. Pass 1 — technical scan of all 100 tickers
   ├─ daily OHLCV + MA20/50, RSI, MACD, ATR
   ├─ relative strength vs SPY
   └─ consolidation/breakout detection
4. Pass 2 — enrich top ~12 candidates
   ├─ sector classification (Yahoo)
   ├─ news sentiment (FinBERT)
   ├─ StockTwits + Reddit chatter
   ├─ insider buying (OpenInsider scrape)
   ├─ earnings calendar
   └─ weekly trend confirmation
5. Compute trade setup (stop/target/R:R) for each
6. Apply selection filters (sector cap, earnings, weekly)
7. Update history.json: settle returns for unsettled picks, append today's
8. Compute regime: sector spread, breadth %, VIX level, HYG trend
9. Write picks.json
```

---

## File layout

```
.
├── .github/workflows/
│   ├── daily-picks.yml          # cron job (5pm SGT weekdays)
│   └── pages.yml                # auto-deploy public/ to Pages
├── daily_picks_bot.py           # the screener
├── requirements.txt             # yfinance, pandas, numpy, requests, transformers, torch
├── public/
│   ├── index.html               # editorial dashboard (4 tabs)
│   ├── picks.json               # today's picks + sector data + regime
│   ├── history.json             # full pick history with computed returns
│   ├── favicon.svg              # SVG icon for browser tabs
│   ├── favicon-32.png           # PNG fallback
│   ├── apple-touch-icon.png     # iOS home screen
│   ├── icon-192.png             # Android PWA
│   ├── icon-512.png             # Android PWA splash
│   └── manifest.json            # PWA metadata
└── README.md
```

---

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

Repo **Settings → Actions → General → Workflow permissions → Read and write permissions**.
Required so the cron can commit `picks.json` and `history.json` back.

### 3. Enable GitHub Pages

Repo **Settings → Pages → Source → GitHub Actions**.
The `pages.yml` workflow handles deployment automatically.

### 4. Trigger first run

**Actions** tab → **Daily Stock Picks** → **Run workflow**.
First run takes ~10–12 minutes (downloads PyTorch CPU + FinBERT model).
Subsequent runs ~3–5 minutes thanks to the HuggingFace cache.

### 5. Visit the dashboard

`https://YOUR-USER.github.io/daily-picks/`

Add to phone home screen for a native-app experience.

---

## Configuration

### Universe selection

In `daily-picks.yml`, change the `--market` flag:

| Flag | Tickers | Use case |
|---|---|---|
| `--market top100` | 100 large-caps, sector-balanced | **Default** |
| `--market ai` | 33 AI-focused names | Concentrated thematic |
| `--market us` | 47 liquid US names | Lighter scan |
| `--market sgx` | 15 SGX blue chips | Singapore market |

Or use a custom file: `--watchlist tickers.txt` (one ticker per line).

### Schedule

Cron line in `daily-picks.yml`. Times are UTC.

| Time you want | Cron |
|---|---|
| 5pm SGT (after SGX close) | `0 9 * * 1-5` |
| 5pm ET (after US close)   | `0 21 * * 1-5` |
| 6am SGT (premarket prep)  | `0 22 * * 0-4` |

### Tunable parameters

In `daily_picks_bot.py`, adjustable constants:

```python
HISTORY_MAX_AGE_DAYS = 365   # auto-prune old history entries
# Selection rules (in select_top):
#   max_per_sector = 1       # at most 1 pick per GICS sector
#   earnings_window = 3      # skip tickers with earnings in next N days
```

Signal weights are inside `evaluate()` — search for `score +=` to find every contribution and tune.

---

## Dashboard guide

The dashboard has four tabs:

**Picks** — today's top 3 with:
- Composite score and reason list
- Live stats (price, MA20, RSI, volume ratio)
- Trade setup: stop loss (2× ATR), nearest-resistance target, risk-reward ratio
- 60-day candlestick chart with MA20/MA50 overlays and volume histogram
- Chips showing key signal hits (RS leader, weekly trend, breakout, insider, etc.)

**Market** — broader context:
- Risk-On / Risk-Off / Neutral regime label with one-sentence interpretation
- Health gauges: VIX (volatility), Breadth (% of universe above 50DMA), HYG (junk credit)
- Heatmap of all 11 SPDR sector ETFs sorted by today's return

**Track Record** — accountability:
- Win rate at 1d / 5d / 30d horizons with average return and alpha vs SPY
- Recent picks table showing actual returns at each horizon

**About** — methodology explanation, signal-by-signal.

---

## Local testing

```bash
pip install -r requirements.txt
python daily_picks_bot.py --market top100 --output public/picks.json

# Serve dashboard locally:
python -m http.server -d public 8000
# Visit http://localhost:8000
```

To skip FinBERT during local dev (faster):

```bash
DISABLE_FINBERT=1 python daily_picks_bot.py --market top100 --output public/picks.json
```

---

## Limitations & honesty

This is a **screener**, not a portfolio system. It tells you what to look at, not how much to buy or when to sell beyond the suggested stop.

- **Yahoo Finance data** has occasional gaps and dividend-adjustment quirks
- **OpenInsider scraping** is brittle — if their HTML changes, insider data silently drops to zero
- **StockTwits/Reddit** skew retail and can be gamed by coordinated posting
- **FinBERT** is accurate on financial text but not magical — it can misread sarcasm or context
- **No backtest** has been run on this exact configuration. The signal stack is well-established in trend-following literature; *that is not the same as proven on this implementation*. Watch the Track Record tab and let the data tell you whether the screener works for your timeframe.

---

## Upgrades worth considering

- **Backtester** — run the full signal stack through 1–2 years of historical data, report hit rates and alpha distribution. Until this exists, the system is design-validated but not statistically validated.
- **Notifications** — add a Telegram or Discord webhook step in the workflow so picks come to your phone instead of you opening the dashboard.
- **Watchlist tier** — display picks ranked 4–10 (passed technical filter, didn't quite top score) on no-pick days.
- **Correlation-aware sector cap** — two stocks in different sectors can still be 0.9 correlated when one theme dominates. Detect and de-duplicate.
- **Pre-market gap detection** — separate 7am ET workflow that flags watchlist tickers gapping >3%.
- **Options flow** — paid feed (UnusualWhales etc.) gives high-quality smart-money signal that's hard to replicate from free sources.

---

## Disclaimer

Not financial advice. This is engineering scaffolding for screening, not a recommendation system. Past patterns don't predict future returns. Verify against your own risk rules, paper-trade for a meaningful period before risking capital, and remember that any system surfaced "looks good" until it doesn't.
