"""
Daily 3-Pick Trading Bot
========================
Screens a watchlist for stocks with all four signals aligned:
  1. Price crossing above the 20-day moving average
  2. Momentum confirmation (RSI rising, MACD positive)
  3. Volume confirmation (today >= 1.5x 20-day avg volume)
  4. Sentiment proxy (recent news flow + headline scoring)

Outputs the top 3 ranked picks with a written reason for each.

Usage:
  python daily_picks_bot.py
  python daily_picks_bot.py --watchlist my_tickers.txt
  python daily_picks_bot.py --market sgx     # use SGX defaults
  python daily_picks_bot.py --market us      # use US defaults

Schedule with cron / Task Scheduler / launchd to run after market close.
"""

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------- Default watchlists ----------
US_DEFAULT = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    # Semis (broad)
    "AMD", "AVGO", "INTC", "QCOM", "MU", "MRVL", "ARM", "TSM",
    # AI infra hardware
    "SMCI", "ALAB", "VRT", "ANET", "DELL", "COHR",
    # Software/SaaS w/ AI exposure
    "ORCL", "CRM", "ADBE", "NOW", "INTU", "SNOW", "DDOG", "NET",
    "PLTR", "AI", "PATH", "SOUN", "IBM",
    # Consumer / other liquid names
    "NFLX", "SHOP", "UBER", "ABNB", "COIN",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Defensive
    "BRK-B", "WMT", "COST",
]

# Focused AI / AI-infrastructure universe (use with --market ai)
AI_DEFAULT = [
    # Chips / accelerators
    "NVDA", "AMD", "AVGO", "MRVL", "TSM", "ARM", "MU", "QCOM", "INTC",
    # AI server / infra hardware
    "SMCI", "ALAB", "VRT", "ANET", "DELL", "COHR",
    # Hyperscalers / cloud AI
    "MSFT", "GOOGL", "META", "AMZN", "ORCL", "IBM",
    # AI software & data platforms
    "PLTR", "SNOW", "DDOG", "CRM", "NOW", "AI", "PATH", "SOUN", "ADBE", "INTU",
    # Networking / edge for AI
    "NET",
    # Application / autonomy
    "TSLA",
]

# Top-100 US large caps by market cap, sector-balanced (use with --market top100)
TOP100_DEFAULT = [
    # Tech / AI / Internet (35)
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
    "ORCL", "NFLX", "ADBE", "CRM", "AMD", "INTC", "INTU", "QCOM",
    "AMAT", "IBM", "NOW", "MU", "TXN", "ARM", "MRVL", "ANET",
    "PLTR", "SMCI", "CSCO", "LRCX", "KLAC", "PANW", "ABNB", "UBER",
    "SHOP", "SNOW", "DDOG",
    # Financials (15)
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "AXP",
    "BLK", "SPGI", "PGR", "SCHW", "CB", "C", "CME",
    # Healthcare (15)
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "AMGN",
    "PFE", "ISRG", "BMY", "GILD", "DHR", "BSX", "VRTX",
    # Consumer (15)
    "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD", "DIS",
    "NKE", "SBUX", "TJX", "LOW", "BKNG", "CMCSA", "MDLZ",
    # Industrial (10)
    "BRK-B", "HON", "GE", "CAT", "RTX", "BA", "UNP", "DE",
    "ETN", "LMT",
    # Energy (5)
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Utilities / Telecom / Materials (5)
    "T", "VZ", "TMUS", "NEE", "LIN",
]

SGX_DEFAULT = [
    "D05.SI",  # DBS
    "O39.SI",  # OCBC
    "U11.SI",  # UOB
    "Z74.SI",  # SingTel
    "C6L.SI",  # SIA
    "F34.SI",  # Wilmar
    "S68.SI",  # SGX
    "G13.SI",  # Genting Sing
    "Y92.SI",  # Thai Bev
    "C38U.SI", # CapitaLand Mall Trust
    "A17U.SI", # CapitaLand Ascendas REIT
    "BN4.SI",  # Keppel
    "S63.SI",  # ST Engineering
    "U96.SI",  # Sembcorp
    "V03.SI",  # Venture
]


# ---------- Indicator math (no ta-lib dependency) ----------
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


# ---------- Signal evaluation ----------
@dataclass
class Signal:
    ticker: str
    name: str
    sector: str
    price: float
    ma20: float
    ma50: float
    cross_today: bool
    cross_recent: bool       # cross within last 10 sessions (informational)
    pct_above_ma20: float
    pct_above_ma50: float
    ma20_slope_5d: float     # % change in MA20 over last 5 sessions
    in_uptrend: bool         # price > MA20 > MA50
    ma20_rising: bool        # MA20 slope positive over 5 sessions
    rsi: float
    rsi_rising: bool
    macd_hist: float
    macd_positive: bool
    volume_ratio: float      # today vs 20d avg
    volume_strong: bool
    news_count_7d: int
    news_score: float        # -1 .. +1 (very rough)
    st_message_count: int    # StockTwits messages in recent stream
    st_bullish_ratio: float  # 0..1 bullish share of tagged messages (0.5 = neutral)
    st_tagged: int           # number of bullish/bearish-tagged messages
    days_to_earnings: Optional[int]  # None if unknown
    composite_score: float
    reasons: list
    chart_data: list         # last 60 sessions: [{time, open, high, low, close, ma20, ma50}, ...]


# Lightweight headline sentiment (placeholder).
# For production: swap this for FinBERT or a paid sentiment API.
POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "soar", "rally", "upgrade",
    "raises", "raised", "record", "growth", "strong", "wins", "approval",
    "approved", "outperform", "buy", "bullish", "breakout", "expansion",
    "deal", "acquisition", "partnership", "launches", "boost", "exceeds",
}
NEGATIVE_WORDS = {
    "miss", "misses", "plunge", "tumble", "downgrade", "cuts", "lawsuit",
    "investigation", "probe", "sec", "fraud", "warning", "weak", "loss",
    "losses", "bearish", "sell", "decline", "falls", "drops", "concerns",
    "delays", "recall", "fired", "resigns",
}

def score_headline(text: str) -> int:
    if not text:
        return 0
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return pos - neg

def fetch_news_sentiment(tk: yf.Ticker) -> tuple[int, float]:
    try:
        news = tk.news or []
    except Exception:
        return 0, 0.0
    if not news:
        return 0, 0.0
    scores = []
    for item in news[:15]:
        # yfinance schema varies; handle both old + new shapes
        content = item.get("content", item)
        title = content.get("title") or item.get("title", "")
        summary = content.get("summary") or item.get("summary", "")
        scores.append(score_headline(f"{title} {summary}"))
    if not scores:
        return 0, 0.0
    raw = np.mean(scores)
    # squash to roughly [-1, 1]
    return len(scores), float(np.tanh(raw))


def fetch_stocktwits(ticker: str) -> tuple[int, float, int]:
    """
    Fetch recent StockTwits messages for a ticker.
    Returns (message_count, bullish_ratio, tagged_count).
    bullish_ratio is 0..1 where 0.5 means equal bull/bear; 1.0 = all bullish.
    Returns (0, 0.5, 0) on failure or no data.
    """
    # StockTwits uses raw symbols; strip common exchange suffixes
    symbol = ticker.split('.')[0].replace('-', '.')
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; daily-picks-bot/1.0)"},
        )
        if resp.status_code != 200:
            return 0, 0.5, 0
        data = resp.json()
    except Exception:
        return 0, 0.5, 0
    messages = data.get("messages", []) or []
    if not messages:
        return 0, 0.5, 0
    bullish = 0
    bearish = 0
    for m in messages:
        ent = m.get("entities") or {}
        sent = (ent.get("sentiment") or {}).get("basic")
        if sent == "Bullish":
            bullish += 1
        elif sent == "Bearish":
            bearish += 1
    tagged = bullish + bearish
    if tagged == 0:
        return len(messages), 0.5, 0
    return len(messages), bullish / tagged, tagged


def fetch_days_to_earnings(tk: yf.Ticker) -> Optional[int]:
    """
    Returns days until next earnings (>=0) or None if unknown.
    Past earnings dates return None.
    """
    try:
        cal = tk.calendar
    except Exception:
        return None
    if cal is None:
        return None
    earnings_date = None
    # New yfinance: dict with "Earnings Date" -> [date, ...]
    if isinstance(cal, dict):
        ed = cal.get("Earnings Date") or cal.get("earningsDate")
        if isinstance(ed, list) and ed:
            earnings_date = ed[0]
        elif ed is not None:
            earnings_date = ed
    # Old yfinance: DataFrame
    elif hasattr(cal, "loc") and hasattr(cal, "empty") and not cal.empty:
        try:
            if "Earnings Date" in cal.index:
                earnings_date = cal.loc["Earnings Date"][0]
        except Exception:
            pass
    if earnings_date is None:
        return None
    try:
        if hasattr(earnings_date, "date"):
            earnings_date = earnings_date.date()
        delta = (earnings_date - datetime.now().date()).days
        return delta if delta >= 0 else None
    except Exception:
        return None


def fetch_sector(tk: yf.Ticker) -> str:
    try:
        info = tk.info
        return info.get("sector") or info.get("industry") or "Unknown"
    except Exception:
        return "Unknown"


def evaluate(ticker: str, enrich: bool = True) -> Optional[Signal]:
    """
    Evaluate one ticker. Computes technicals always.
    If enrich=False, skips expensive lookups (sector, StockTwits, earnings).
    These are filled in later by enrich_signal() for finalists only.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="6mo", interval="1d", auto_adjust=False)
        if len(df) < 30:
            return None
    except Exception as e:
        print(f"  ! {ticker}: data fetch failed ({e})", file=sys.stderr)
        return None

    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["RSI"] = rsi(df["Close"], 14)
    df["VolAvg20"] = sma(df["Volume"], 20)
    macd_line, sig_line, hist = macd(df["Close"])
    df["MACD_hist"] = hist

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if any(pd.isna(x) for x in [last["MA20"], last["MA50"], last["RSI"], last["VolAvg20"]]):
        return None

    # Trend metrics
    pct_above_ma20 = float((last["Close"] / last["MA20"] - 1) * 100)
    pct_above_ma50 = float((last["Close"] / last["MA50"] - 1) * 100)
    ma_separation = float((last["MA20"] / last["MA50"] - 1) * 100)
    if len(df) > 6 and not pd.isna(df["MA20"].iloc[-6]):
        ma20_slope_5d = float((last["MA20"] / df["MA20"].iloc[-6] - 1) * 100)
    else:
        ma20_slope_5d = 0.0

    in_uptrend = bool(last["Close"] > last["MA20"] > last["MA50"])
    ma20_rising = bool(ma20_slope_5d > 0.3)

    # Cross detection (informational)
    cross_today = bool(last["Close"] > last["MA20"] and prev["Close"] <= prev["MA20"])
    cross_recent = False
    for i in range(1, 11):
        if i + 1 > len(df):
            break
        a = df.iloc[-i]
        b = df.iloc[-i - 1]
        if a["Close"] > a["MA20"] and b["Close"] <= b["MA20"]:
            cross_recent = True
            break

    rsi_now = float(last["RSI"])
    rsi_rising = bool(last["RSI"] > prev["RSI"])
    macd_h = float(last["MACD_hist"])
    macd_positive = bool(macd_h > 0 and macd_h > prev["MACD_hist"])
    vol_ratio = float(last["Volume"] / last["VolAvg20"]) if last["VolAvg20"] else 0
    vol_strong = vol_ratio >= 1.5

    # Defer expensive calls until we know the ticker is a candidate
    if enrich:
        sector = fetch_sector(tk)
        news_count, news_score = fetch_news_sentiment(tk)
        st_messages, st_bull_ratio, st_tagged = fetch_stocktwits(ticker)
        days_to_earn = fetch_days_to_earnings(tk)
        time.sleep(0.3)  # gentle rate limit
    else:
        sector = "Unknown"
        news_count, news_score = 0, 0.0
        st_messages, st_bull_ratio, st_tagged = 0, 0.5, 0
        days_to_earn = None

    # Trend-focused scoring
    score = 0.0
    reasons = []

    if in_uptrend and ma20_rising:
        score += 3.0
        reasons.append(f"In uptrend: price > MA20 > MA50, MA20 rising {ma20_slope_5d:+.2f}% over 5 sessions")

    if 0 < pct_above_ma20 <= 3:
        score += 2.0
        reasons.append(f"Healthy pullback: only {pct_above_ma20:+.2f}% above MA20 — good entry near support")
    elif 3 < pct_above_ma20 <= 8:
        score += 1.0
        reasons.append(f"{pct_above_ma20:+.2f}% above MA20 — moderate extension, room to run")
    elif 8 < pct_above_ma20 <= 15:
        reasons.append(f"{pct_above_ma20:+.2f}% above MA20 — extended, limited entry margin")
    elif pct_above_ma20 > 15:
        score -= 1.5
        reasons.append(f"{pct_above_ma20:+.2f}% above MA20 — overextended, mean-reversion risk")

    if ma_separation >= 4:
        score += 1.5
        reasons.append(f"Established trend: MA20 sits {ma_separation:+.2f}% above MA50")
    elif ma_separation >= 1.5:
        score += 0.5
        reasons.append(f"Trend forming: MA20 {ma_separation:+.2f}% above MA50")

    if 50 <= rsi_now <= 75 and rsi_rising:
        score += 1.0
        reasons.append(f"RSI {rsi_now:.1f} and rising — momentum healthy")
    elif rsi_now > 80:
        score -= 0.5
        reasons.append(f"RSI {rsi_now:.1f} extreme — wait for cooldown")

    if macd_positive:
        score += 1.0
        reasons.append(f"MACD histogram positive and expanding ({macd_h:+.3f})")

    if vol_strong:
        score += 1.5
        reasons.append(f"Volume {vol_ratio:.2f}x the 20-day average — institutional interest")
    elif vol_ratio >= 1.0:
        score += 0.3

    if cross_recent and pct_above_ma20 < 6:
        score += 0.5
        reasons.append("Recent MA20 cross within last 10 sessions — potentially fresh trend")

    # Social sentiment (StockTwits) — primary social signal
    if enrich and st_tagged >= 5:
        if st_bull_ratio >= 0.70:
            score += 2.0
            reasons.append(f"StockTwits crowd is {st_bull_ratio*100:.0f}% bullish across {st_tagged} tagged messages — strong positive sentiment")
        elif st_bull_ratio >= 0.55:
            score += 1.0
            reasons.append(f"StockTwits crowd leans bullish ({st_bull_ratio*100:.0f}% of {st_tagged} tagged messages)")
        elif st_bull_ratio <= 0.30:
            score -= 1.5
            reasons.append(f"StockTwits crowd is {(1-st_bull_ratio)*100:.0f}% bearish across {st_tagged} tagged messages — negative sentiment")
    if enrich and st_messages >= 25:
        score += 0.5
        reasons.append(f"High discussion volume on StockTwits ({st_messages} recent messages)")

    # News headlines (secondary)
    if news_score > 0.15:
        score += 1.0
        reasons.append(f"{news_count} recent headlines, net positive tone (score {news_score:+.2f})")
    elif news_score < -0.15:
        score -= 1.5
        reasons.append(f"{news_count} recent headlines, net negative tone (score {news_score:+.2f})")
    elif news_count >= 5:
        score += 0.3

    # Earnings warning (informational; hard filter happens in select_top)
    if enrich and days_to_earn is not None and 0 <= days_to_earn <= 3:
        reasons.append(f"⚠ Earnings in {days_to_earn} day(s) — excluded from final picks")

    name = ticker
    try:
        info_name = tk.info.get("shortName") or tk.info.get("longName")
        if info_name:
            name = info_name
    except Exception:
        pass

    # Build chart data: last 60 sessions of OHLC + Volume + MA20 + MA50 for the dashboard
    chart_df = df.tail(60)
    chart_data = []
    for ts, row in chart_df.iterrows():
        ma20_val = row["MA20"]
        ma50_val = row["MA50"]
        chart_data.append({
            "time": ts.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),  4),
            "high":   round(float(row["High"]),  4),
            "low":    round(float(row["Low"]),   4),
            "close":  round(float(row["Close"]), 4),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
            "ma20":   None if pd.isna(ma20_val) else round(float(ma20_val), 4),
            "ma50":   None if pd.isna(ma50_val) else round(float(ma50_val), 4),
        })

    return Signal(
        ticker=ticker,
        name=name,
        sector=sector,
        price=float(last["Close"]),
        ma20=float(last["MA20"]),
        ma50=float(last["MA50"]),
        cross_today=cross_today,
        cross_recent=cross_recent,
        pct_above_ma20=pct_above_ma20,
        pct_above_ma50=pct_above_ma50,
        ma20_slope_5d=ma20_slope_5d,
        in_uptrend=in_uptrend,
        ma20_rising=ma20_rising,
        rsi=rsi_now,
        rsi_rising=rsi_rising,
        macd_hist=macd_h,
        macd_positive=macd_positive,
        volume_ratio=vol_ratio,
        volume_strong=vol_strong,
        news_count_7d=news_count,
        news_score=news_score,
        st_message_count=st_messages,
        st_bullish_ratio=st_bull_ratio,
        st_tagged=st_tagged,
        days_to_earnings=days_to_earn,
        composite_score=round(score, 2),
        reasons=reasons,
        chart_data=chart_data,
    )


def select_top(signals: list, n: int = 3, max_per_sector: int = 1, earnings_window: int = 3) -> list:
    """
    Pick top N with two diversification rules:
      1. Skip tickers with earnings in the next `earnings_window` days
      2. Allow at most `max_per_sector` picks per GICS sector
    """
    eligible = [s for s in signals if s.in_uptrend and s.ma20_rising]
    # Earnings filter
    eligible = [
        s for s in eligible
        if s.days_to_earnings is None or s.days_to_earnings > earnings_window
    ]
    eligible.sort(key=lambda s: s.composite_score, reverse=True)

    chosen = []
    sector_count: dict = {}
    for s in eligible:
        if sector_count.get(s.sector, 0) >= max_per_sector:
            continue
        chosen.append(s)
        sector_count[s.sector] = sector_count.get(s.sector, 0) + 1
        if len(chosen) >= n:
            break
    return chosen


def render_report(picks: list, scanned: int) -> str:
    today = datetime.now().strftime("%A, %d %b %Y")
    lines = []
    lines.append(f"DAILY PICKS — {today}")
    lines.append(f"Scanned {scanned} tickers")
    lines.append("=" * 60)
    if not picks:
        lines.append("No tickers met the criteria today. Stay in cash.")
        return "\n".join(lines)
    for i, s in enumerate(picks, 1):
        lines.append(f"\n#{i}  {s.ticker} — {s.name}")
        lines.append(f"    Sector: {s.sector}  |  Price: {s.price:.2f}  |  MA20: {s.ma20:.2f}  |  Score: {s.composite_score}")
        if s.days_to_earnings is not None:
            lines.append(f"    Next earnings in {s.days_to_earnings} day(s)")
        lines.append("    Why:")
        for r in s.reasons:
            lines.append(f"      • {r}")
    lines.append("\n" + "-" * 60)
    lines.append("Not financial advice. Verify against your own risk rules before trading.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "sgx", "ai", "top100", "both"], default="both")
    parser.add_argument("--watchlist", help="Path to a text file with one ticker per line")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--output", help="Write JSON output to this file path (for dashboard)")
    args = parser.parse_args()

    if args.watchlist:
        with open(args.watchlist) as f:
            tickers = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif args.market == "us":
        tickers = US_DEFAULT
    elif args.market == "sgx":
        tickers = SGX_DEFAULT
    elif args.market == "ai":
        tickers = AI_DEFAULT
    elif args.market == "top100":
        tickers = TOP100_DEFAULT
    else:
        tickers = US_DEFAULT + SGX_DEFAULT

    print(f"Scanning {len(tickers)} tickers (technicals only)...", file=sys.stderr)
    # Pass 1: cheap technicals only — no API calls beyond price data
    technical_signals = []
    for t in tickers:
        s = evaluate(t, enrich=False)
        if s is not None:
            technical_signals.append(s)

    # Filter to candidates that pass the technical bar (uptrend + rising MA20)
    candidates = [s for s in technical_signals if s.in_uptrend and s.ma20_rising]
    candidates.sort(key=lambda s: s.composite_score, reverse=True)
    # Enrich only the top ~3x what we need, to give the sector filter room
    enrich_count = min(len(candidates), max(args.top * 4, 8))
    top_candidates = candidates[:enrich_count]
    print(f"Enriching {len(top_candidates)} candidates with sentiment/earnings/sector...", file=sys.stderr)

    # Pass 2: re-evaluate with full enrichment for finalists
    signals = []
    for s in top_candidates:
        full = evaluate(s.ticker, enrich=True)
        if full is not None:
            signals.append(full)

    picks = select_top(signals, n=args.top)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "market": args.market,
        "scanned": len(technical_signals),
        "enriched": len(signals),
        "universe_size": len(tickers),
        "picks": [asdict(p) for p in picks],
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {len(picks)} picks to {args.output}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.output:
        print(render_report(picks, scanned=len(signals)))


if __name__ == "__main__":
    main()
