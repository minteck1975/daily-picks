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
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------- Default watchlists ----------
US_DEFAULT = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "AVGO", "NFLX", "CRM", "ORCL", "ADBE", "INTC", "QCOM", "MU",
    "PLTR", "COIN", "SHOP", "UBER", "ABNB", "SNOW", "DDOG", "NET",
    "JPM", "BAC", "GS", "V", "MA", "BRK-B", "WMT", "COST",
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
    price: float
    ma20: float
    cross_today: bool
    cross_recent: bool       # cross within last 3 sessions
    pct_above_ma20: float
    rsi: float
    rsi_rising: bool
    macd_hist: float
    macd_positive: bool
    volume_ratio: float      # today vs 20d avg
    volume_strong: bool
    news_count_7d: int
    news_score: float        # -1 .. +1 (very rough)
    composite_score: float
    reasons: list
    chart_data: list         # last 60 sessions: [{time, open, high, low, close, ma20}, ...]


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


def evaluate(ticker: str) -> Optional[Signal]:
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="6mo", interval="1d", auto_adjust=False)
        if len(df) < 30:
            return None
    except Exception as e:
        print(f"  ! {ticker}: data fetch failed ({e})", file=sys.stderr)
        return None

    df["MA20"] = sma(df["Close"], 20)
    df["RSI"] = rsi(df["Close"], 14)
    df["VolAvg20"] = sma(df["Volume"], 20)
    macd_line, sig_line, hist = macd(df["Close"])
    df["MACD_hist"] = hist

    last = df.iloc[-1]
    prev = df.iloc[-2]

    if any(pd.isna(x) for x in [last["MA20"], last["RSI"], last["VolAvg20"]]):
        return None

    # Cross detection: close above MA20 today, close below or equal MA20 yesterday
    cross_today = bool(last["Close"] > last["MA20"] and prev["Close"] <= prev["MA20"])
    cross_recent = False
    for i in range(1, 4):
        if i + 1 > len(df):
            break
        a = df.iloc[-i]
        b = df.iloc[-i - 1]
        if a["Close"] > a["MA20"] and b["Close"] <= b["MA20"]:
            cross_recent = True
            break

    pct_above_ma20 = float((last["Close"] / last["MA20"] - 1) * 100)
    rsi_now = float(last["RSI"])
    rsi_rising = bool(last["RSI"] > prev["RSI"])
    macd_h = float(last["MACD_hist"])
    macd_positive = bool(macd_h > 0 and macd_h > prev["MACD_hist"])
    vol_ratio = float(last["Volume"] / last["VolAvg20"]) if last["VolAvg20"] else 0
    vol_strong = vol_ratio >= 1.5

    news_count, news_score = fetch_news_sentiment(tk)

    # Composite scoring (tunable weights)
    score = 0.0
    reasons = []

    if cross_today:
        score += 3.0
        reasons.append(f"Price closed above MA20 today ({last['Close']:.2f} vs MA20 {last['MA20']:.2f})")
    elif cross_recent:
        score += 1.5
        reasons.append(f"MA20 cross within last 3 sessions; now {pct_above_ma20:+.2f}% above MA20")

    if 50 <= rsi_now <= 70 and rsi_rising:
        score += 1.5
        reasons.append(f"RSI {rsi_now:.1f} and rising — momentum building, not yet overbought")
    elif rsi_now > 70:
        score -= 0.5
        reasons.append(f"RSI {rsi_now:.1f} is overbought (caution)")

    if macd_positive:
        score += 1.5
        reasons.append(f"MACD histogram positive and expanding ({macd_h:+.3f})")

    if vol_strong:
        score += 2.0
        reasons.append(f"Volume {vol_ratio:.2f}x the 20-day average — institutional interest")
    elif vol_ratio >= 1.0:
        score += 0.5

    # Sentiment
    if news_score > 0.15:
        score += 1.5
        reasons.append(f"{news_count} recent headlines, net positive tone (score {news_score:+.2f})")
    elif news_score < -0.15:
        score -= 1.5
        reasons.append(f"{news_count} recent headlines, net negative tone (score {news_score:+.2f})")
    elif news_count >= 5:
        score += 0.5
        reasons.append(f"{news_count} recent headlines — high attention, neutral tone")

    name = ticker
    try:
        info_name = tk.info.get("shortName") or tk.info.get("longName")
        if info_name:
            name = info_name
    except Exception:
        pass

    # Build chart data: last 60 sessions of OHLC + MA20 for the dashboard
    chart_df = df.tail(60)
    chart_data = []
    for ts, row in chart_df.iterrows():
        ma20_val = row["MA20"]
        chart_data.append({
            "time": ts.strftime("%Y-%m-%d"),
            "open":  round(float(row["Open"]),  4),
            "high":  round(float(row["High"]),  4),
            "low":   round(float(row["Low"]),   4),
            "close": round(float(row["Close"]), 4),
            "ma20":  None if pd.isna(ma20_val) else round(float(ma20_val), 4),
        })

    return Signal(
        ticker=ticker,
        name=name,
        price=float(last["Close"]),
        ma20=float(last["MA20"]),
        cross_today=cross_today,
        cross_recent=cross_recent,
        pct_above_ma20=pct_above_ma20,
        rsi=rsi_now,
        rsi_rising=rsi_rising,
        macd_hist=macd_h,
        macd_positive=macd_positive,
        volume_ratio=vol_ratio,
        volume_strong=vol_strong,
        news_count_7d=news_count,
        news_score=news_score,
        composite_score=round(score, 2),
        reasons=reasons,
        chart_data=chart_data,
    )


def select_top(signals: list, n: int = 3) -> list:
    # Hard requirement: must have crossed (today or recently)
    eligible = [s for s in signals if s.cross_today or s.cross_recent]
    eligible.sort(key=lambda s: s.composite_score, reverse=True)
    return eligible[:n]


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
        lines.append(f"    Price: {s.price:.2f}  |  MA20: {s.ma20:.2f}  |  Score: {s.composite_score}")
        lines.append("    Why:")
        for r in s.reasons:
            lines.append(f"      • {r}")
    lines.append("\n" + "-" * 60)
    lines.append("Not financial advice. Verify against your own risk rules before trading.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["us", "sgx", "both"], default="both")
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
    else:
        tickers = US_DEFAULT + SGX_DEFAULT

    print(f"Scanning {len(tickers)} tickers...", file=sys.stderr)
    signals = []
    for t in tickers:
        s = evaluate(t)
        if s is not None:
            signals.append(s)

    picks = select_top(signals, n=args.top)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "market": args.market,
        "scanned": len(signals),
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
