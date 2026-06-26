# The Tape — Daily Stock Picks

A trend-following screener for US large-caps that runs nightly on GitHub Actions, scoring stocks across 15+ signals (technicals, fundamentals, freshness, sentiment, insider activity, market context). Publishes up to 3 picks to a public dashboard with stop/target setups, realistic entry tracking, simulated-execution stats, and a running track record.

**Live dashboard:** `https://minteck1975.github.io/daily-picks/`

---

## What it does

Each weekday at 5:21 AM Singapore time (just after the US close), the bot:

1. **Scans 100 large-cap US stocks** — sector-balanced, heavy on tech/AI
2. **Filters for established uptrends** — price > MA20 > MA50, MA20 rising
3. **Enriches the top ~12 candidates** with fundamentals, sentiment, insider data, weekly trend, earnings calendar, and trend-stage classification
4. **Applies a 5-day ticker cooldown** — won't re-pick names selected in the last 5 days, forcing fresh setups instead of doubling down
5. **Selects up to 3 picks** — at most one per sector, none with earnings in the next 3 days, none with weekly-down trends
6. **Computes a full trade setup per pick** — ATR-based stop, nearest-resistance target, risk-reward ratio
7. **Snapshots the broader regime** — sector breadth, VIX, junk-credit (HYG), Fear & Greed Index
8. **Maintains a persistent track record** that auto-computes returns at 1/3/5/30 days, backfills realistic entries (next-day open), and simulates stop-loss/target execution
9. **Commits everything back to the repo**, triggering an automatic Pages redeploy

---

## The signal stack

### Required (hard filter)

Stocks must satisfy all of: price > MA20 > MA50, MA20 rising over 5 sessions. Without this, no further scoring happens.

### Freshness & acceleration (favored)

| Signal | Weight | Catches |
|---|---|---|
| Fresh MA50 reclaim (≤15d ago) | **+3.0** | Just-crossed-above-MA50, early-stage uptrend |
| Fresh MA50 reclaim (16–30d) | +1.5 | Early but not "first day" |
| Golden cross (MA20 > MA50, ≤20d ago) | **+2.5** | Structural shift from base to uptrend |
| ROC accelerating (5d > 10d > 20d rate) | **+2.5** | Trend picking up pace |
| Volume expanding (5d avg > 1.3× 20d avg) | +1.5 | Recent institutional accumulation |
| Volume warming (>1.1×) | +0.5 | Mild expansion |

### Established trend (rewarded, but less heavily than before)

| Signal | Weight | Catches |
|---|---|---|
| RS leader (vs SPY +30d > 3%) | +1.0 | Outperforming the index |
| Weekly trend up (10w MA + rising) | +1.5 | Multi-timeframe alignment |
| Near 52-week high (within 15%) | +0.5 | Long-term uptrend intact |
| Sweet spot (10–30% below 52w high) | +0.5 | Runway intact, not yet extended |

### Maturity penalties (discount late-stage)

| Signal | Weight | Reasoning |
|---|---|---|
| Trend > 60 days old | −0.5 | Statistical mean reversion looming |
| RSI > 72 AND price > 8% above MA20 | −1.5 | Clearly overbought + extended |
| Near 52w high AND RSI > 68 AND trend > 40d | −1.0 | Classic topping pattern |

### Sentiment layer

| Signal | Weight | Source |
|---|---|---|
| News sentiment (title + summary) | ±2.0 | FinBERT NLP on yfinance news |
| StockTwits ≥70% bullish (5+ tagged) | +2.0 | Crowd sentiment |
| Reddit chatter (4 major subs, 7d) | +1.5 | Discussion volume + upvotes |

### Insider conviction

| Signal | Weight | Source |
|---|---|---|
| Insider buys ≥$1M (last 30d) | +2.0 | OpenInsider scrape (Form 4 filings) |

### Selection filters (applied after scoring)

1. Skip any ticker with earnings in the next 3 trading days
2. Exclude any explicit weekly downtrends
3. **5-day ticker cooldown** — exclude names picked in the last 5 days
4. Maximum 1 pick per GICS sector

The top 3 by composite score that pass all filters become the day's picks.

### Trend stage classification

Each pick is tagged with a stage chip on the dashboard:

- **● Fresh trend** — under 20 days into the move, or fresh MA50 cross within 15 days
- **● Accelerating** — established trend, ROC accelerating
- **● Developing** — normal trend in mid-progression
- **● Mature** — late-stage, near 52w high, elevated RSI

---

## Architecture

```
┌────────────────────────────────────────┐
│  GitHub Actions cron (5:21 AM SGT)     │
│  ─ runs daily_picks_bot.py             │
│  ─ updates history.json (realistic     │
│    entries, returns, simulated exits)  │
│  ─ writes picks.json                   │
│  ─ commits both back to main           │
└──────────────┬─────────────────────────┘
               │ workflow_run trigger
               ▼
┌────────────────────────────────────────┐
│  Deploy Pages workflow                 │
│  ─ uploads public/ to GH Pages         │
└──────────────┬─────────────────────────┘
               │ ~60s
               ▼
┌────────────────────────────────────────┐
│  GitHub Pages (CDN-cached)             │
│  ─ index.html fetches picks.json       │
│  ─ Track tab fetches history.json      │
└────────────────────────────────────────┘
```

### Critical: how the workflows are chained

Commits made by the cron use the default `GITHUB_TOKEN`, which **does not trigger normal `push` workflows**. Without explicit chaining, the dashboard would never auto-redeploy after a cron run.

Fix: `pages.yml` uses a `workflow_run` trigger that fires the moment `daily-picks.yml` completes. This makes the full chain automatic.

### Data flow inside `daily_picks_bot.py`

```
1. Load benchmarks (SPY, ^VIX, HYG, 11 sector ETFs)
2. Initialize FinBERT (one-time per run)
3. Pass 1 — technical scan of all 100 tickers
   ├─ daily OHLCV + MA20/50, RSI, MACD, ATR
   ├─ relative strength vs SPY
   ├─ freshness metrics (days above MA50, golden cross days, ROC)
   └─ consolidation/breakout detection
4. Load history.json
5. Apply ticker cooldown filter to candidates
6. Pass 2 — enrich top ~12 candidates
   ├─ sector + fundamentals (P/E, growth, market cap)
   ├─ news sentiment (FinBERT)
   ├─ StockTwits + Reddit chatter
   ├─ insider buying (OpenInsider)
   ├─ earnings calendar
   └─ weekly trend confirmation
7. Compute trade setup (stop/target/RR) for each
8. Apply selection filters (sector cap, earnings, weekly)
9. History pipeline:
   ├─ Backfill realistic entries (signal close → next-day open)
   ├─ Recompute returns at 1d/3d/5d/30d horizons
   ├─ Simulate stop/target exits on all unsettled picks
   └─ Append today's new picks
10. Fetch regime: VIX, HYG, breadth %, Fear & Greed
11. Fetch top 10 market headlines + FinBERT mood
12. Write picks.json
```

---

## File layout

```
.
├── .github/workflows/
│   ├── daily-picks.yml          # cron job (5:21 AM SGT, post-US-close)
│   └── pages.yml                # auto-deploy via workflow_run chaining
├── daily_picks_bot.py           # the screener (~2100 lines)
├── requirements.txt             # yfinance, pandas, numpy, requests, transformers, torch
├── public/
│   ├── index.html               # editorial dashboard (5 tabs)
│   ├── picks.json               # today's picks + sector data + regime + news
│   ├── history.json             # full pick history with computed returns + exits
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

Repo **Settings → Actions → General → Workflow permissions → Read and write permissions**. Required so the cron can commit `picks.json` and `history.json` back.

### 3. Enable GitHub Pages

Repo **Settings → Pages → Source → GitHub Actions**. The `pages.yml` workflow handles deployment automatically (both on push and after `daily-picks.yml` finishes).

### 4. Trigger first run

**Actions** tab → **Daily Stock Picks** → **Run workflow**. First run takes ~10–12 minutes (downloads PyTorch CPU + FinBERT model). Subsequent runs ~3–5 minutes thanks to the HuggingFace cache.

### 5. Visit the dashboard

`https://YOUR-USER.github.io/daily-picks/`

Add to phone home screen for a native-app PWA experience.

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

Cron line in `daily-picks.yml`. **Times are always UTC.** Singapore is UTC+8 (no DST).

| Time you want | Cron | Notes |
|---|---|---|
| **5:21 AM SGT (after US close)** | `21 21 * * 1-5` | **Current.** Off-minute dodges GH load spike |
| 5:00 AM SGT exactly | `0 21 * * 1-5` | Top-of-hour — may be delayed |
| 5:00 PM SGT (after SGX close) | `0 9 * * 1-5` | Original setting |
| 6:00 AM SGT (extra buffer) | `0 22 * * 1-5` | Full hour cushion year-round |

US market closes 21:00 UTC (winter) or 20:00 UTC (summer). A 21:21 UTC run lands right at or just after the close with fresh end-of-day bars.

### Tunable parameters

In `daily_picks_bot.py`:

```python
HISTORY_MAX_AGE_DAYS = 365   # auto-prune old history
COOLDOWN_DAYS = 5             # don't re-pick a ticker within N days
HORIZON_DAYS = 30             # max holding period for exit simulation
DEFAULT_STOP_PCT = 0.08       # 8% fallback stop for legacy picks
DEFAULT_TARGET_PCT = 0.15     # 15% fallback target for legacy picks
```

Signal weights live inside `evaluate()` — search for `score +=` and `score -=` to find every contribution and tune.

In `select_top()`:

```python
max_per_sector = 1     # at most 1 pick per GICS sector
earnings_window = 3    # skip tickers with earnings in next N days
```

---

## Dashboard guide

Five tabs:

### Picks
Today's top selections with:
- Composite score + reason list
- Trend-stage chip (fresh / accelerating / developing / mature)
- Acceleration chips (ROC, MA50 reclaim, volume expanding)
- Fundamental stats: P/E (TTM), Forward P/E, Revenue Growth, EPS Growth
- 60-day candlestick chart with MA20/MA50 overlays and volume histogram
- Signal chips (RS leader, weekly trend, insider, breakout, etc.)

### Market
Broader context:
- Risk-On / Risk-Off / Neutral regime label
- CNN Fear & Greed Index analog gauge (semicircle with needle, FEAR–GREED zones)
- Three health gauges: VIX, Breadth (% above 50DMA), HYG (junk credit)
- Heatmap of all 11 SPDR sector ETFs

### News
- Top 10 market headlines (aggregated from major tickers + ETFs)
- FinBERT-computed overall headline mood (strongly positive → strongly negative)
- Source, time-ago, headline, summary preview for each item

### Track Record
- **Buy & hold performance** at 1d / 3d / 5d horizons (raw returns + alpha vs SPY)
- **Simulated execution panel** — what disciplined stop/target execution would have returned (win rate, avg return, avg days held, breakdown of stop / target / horizon exits)
- Recent picks table with entry, returns at each horizon, and Exit column showing how each pick would have closed under the bot's original stop/target setup

### About
Methodology explanation, signal-by-signal.

---

## Performance tracking system

The track record uses a multi-stage pipeline that runs every cron:

### Stage 1: Realistic entry backfill
On day +1, the bot fetches the next session's OPEN for yesterday's picks and replaces the signal close with the realistic entry price. The original signal close is preserved as `signal_close` for reference. Picks marked with `*` in the dashboard table haven't been backfilled yet.

### Stage 2: Raw return computation
Returns at 1d / 3d / 5d / 30d are computed once each and stored permanently. Sanity bounds clip anything outside −80% to +300% (yfinance occasionally returns split-adjusted glitches that would otherwise contaminate stats).

### Stage 3: Simulated stop/target execution
For each pick, the bot walks daily OHLC after entry:
- If next-day **open** ≤ stop → exit at the open (`stop_gap`)
- Else if intraday **low** ≤ stop → exit at stop (`stop`)
- Else if next-day **open** ≥ target → exit at the open (`target_gap`)
- Else if intraday **high** ≥ target → exit at target (`target`)
- If 30 days elapse with no exit → close at day-30 price (`horizon`)

This converts buy-and-hold theoretical returns into what disciplined execution would actually have realized.

---

## Local testing

```bash
pip install -r requirements.txt
python daily_picks_bot.py --market top100 --output public/picks.json

# Serve dashboard locally:
python -m http.server -d public 8000
# Visit http://localhost:8000
```

Faster local runs (skip FinBERT):
```bash
DISABLE_FINBERT=1 python daily_picks_bot.py --market top100 --output public/picks.json
```

---

## Limitations & honesty

This is a **screener**, not a portfolio system. It tells you what to look at, what stop/target to use, and what disciplined execution would have returned. It does not place orders or size positions.

- **Yahoo Finance data** has occasional gaps and dividend-adjustment quirks. Returns > +300% or < −80% are clipped as data artifacts.
- **OpenInsider scraping** is brittle. If their HTML changes, insider data silently drops to zero.
- **StockTwits/Reddit** skew retail and can be gamed by coordinated posting.
- **FinBERT** is accurate on financial text but not magical — it can misread sarcasm or context.
- **CNN Fear & Greed endpoint** is unofficial. Stable for years but could change without notice.
- **Simulated execution does not model slippage.** Real stop-market orders fill below the stop on gap-downs; the simulation assumes you fill at the stop level. Real-world returns would be slightly worse — typically 10–20% lower win rate, 0.1–0.2% lower avg return.
- **No backtest** has been run on this exact configuration. The signal stack is well-established in trend-following literature; *that is not the same as proven on this implementation*. The Track Record tab is your real-time validation.

---

## Upgrades worth considering

- **Backtester** — run the full signal stack through 1–2 years of historical data, report hit rates and alpha distribution at each horizon. Until this exists, the system is design-validated but not statistically validated.
- **Notifications** — Telegram/Discord webhook step so picks come to your phone instead of you opening the dashboard.
- **Slippage modeling** — apply a realistic fill assumption (e.g., +0.1% on entries, −0.2% on stop-gaps) to the exit simulation.
- **Wider targets / better RR** — current setups have RR ~0.15 (small targets, high win rate). Adjusting `compute_trade_setup` to use wider resistance levels would give better RR but lower win rate. Trade-off worth experimenting with.
- **Pre-market gap detection** — separate 7am ET workflow that flags watchlist tickers gapping >3%.
- **Watchlist tier** — display picks ranked 4–10 on no-pick days.

---

## Disclaimer

Not financial advice. This is engineering scaffolding for screening and educational use, not a recommendation system. Past patterns don't predict future returns. Verify against your own risk rules, paper-trade for a meaningful period before risking capital, and remember that any system surfaced "looks good" until it doesn't.
