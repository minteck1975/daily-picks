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
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# FinBERT is optional — bot falls back to keyword scoring if unavailable
try:
    from transformers import pipeline
    _FINBERT_AVAILABLE = True
except ImportError:
    _FINBERT_AVAILABLE = False

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

# SPDR sector ETFs — used to gauge sector breadth + risk-on/off regime
SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLC":  "Communication Services",
    "XLY":  "Consumer Cyclical",
    "XLF":  "Financial Services",
    "XLI":  "Industrials",
    "XLB":  "Basic Materials",
    "XLE":  "Energy",
    "XLV":  "Healthcare",
    "XLP":  "Consumer Defensive",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
}
# Risk-on / risk-off groupings for regime detection
CYCLICAL_SECTORS  = {"XLK", "XLY", "XLF", "XLI", "XLB", "XLC"}
DEFENSIVE_SECTORS = {"XLU", "XLP", "XLV", "XLRE"}


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
    cross_recent: bool
    pct_above_ma20: float
    pct_above_ma50: float
    ma20_slope_5d: float
    in_uptrend: bool
    ma20_rising: bool
    rsi: float
    rsi_rising: bool
    macd_hist: float
    macd_positive: bool
    volume_ratio: float
    volume_strong: bool
    # New: relative strength
    rs_vs_spy_30d: float       # outperformance vs SPY over 30 sessions, %
    rs_leader: bool            # outperforming SPY by > 3%
    # New: multi-timeframe
    weekly_uptrend: Optional[bool]   # weekly close > weekly MA10 AND MA10 rising
    # New: 52-week high context
    pct_below_52w_high: float        # how far below 52w high (negative number)
    near_52w_high: bool              # within 15% of 52w high
    # New: volatility (ATR)
    atr_14: float                    # absolute ATR
    atr_pct: float                   # ATR as % of price
    # New: consolidation/breakout
    breakout_recent: bool            # broke out of tight range in last 10 sessions
    breakout_days_since: Optional[int]
    consolidation_range_pct: Optional[float]
    # Existing sentiment
    news_count_7d: int
    news_score: float
    st_message_count: int
    st_bullish_ratio: float
    st_tagged: int
    # New: Reddit + insider
    reddit_mention_count: int
    reddit_score_total: int
    insider_buy_count_30d: int
    insider_buy_value_usd: int
    # New: X / Twitter sentiment
    x_mention_count: int
    x_sentiment: float           # -1 .. +1 (FinBERT-scored) or 0 if no data
    x_source: str                # 'api' | 'nitter' | 'none'
    days_to_earnings: Optional[int]
    composite_score: float
    reasons: list
    chart_data: list
    # New: trade setup
    stop_loss: float
    stop_distance_pct: float
    target: float
    target_distance_pct: float
    rr_ratio: float
    # New: fundamentals
    pe_ttm: Optional[float]
    pe_forward: Optional[float]
    revenue_growth_yoy: Optional[float]
    earnings_growth_yoy: Optional[float]
    profit_margin: Optional[float]
    market_cap: Optional[int]
    # New: freshness & acceleration
    days_above_ma50: Optional[int]
    days_since_ma50_cross: Optional[int]
    golden_cross_days: Optional[int]
    roc_5d: float
    roc_10d: float
    roc_20d: float
    roc_accelerating: bool
    volume_trend_ratio: float
    stage: str   # 'fresh' | 'developing' | 'accelerating' | 'mature'


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


def fetch_fundamentals(tk: yf.Ticker) -> dict:
    """Pull key fundamental ratios from yfinance .info. Reuses cached info."""
    empty = {
        "pe_ttm": None, "pe_forward": None,
        "revenue_growth_yoy": None, "earnings_growth_yoy": None,
        "profit_margin": None, "market_cap": None,
    }
    try:
        info = tk.info or {}
    except Exception:
        return empty

    def _f(key):
        v = info.get(key)
        if v is None:
            return None
        try:
            f = float(v)
            # Filter out NaN, infinity, and meaningless zeros for ratios
            if f != f or f in (float("inf"), float("-inf")):
                return None
            return f
        except (TypeError, ValueError):
            return None

    pe_ttm = _f("trailingPE")
    pe_forward = _f("forwardPE")
    # P/E often returns garbage for unprofitable companies; cap insanity
    if pe_ttm is not None and (pe_ttm <= 0 or pe_ttm > 1000):
        pe_ttm = None
    if pe_forward is not None and (pe_forward <= 0 or pe_forward > 1000):
        pe_forward = None

    market_cap = info.get("marketCap")
    try:
        market_cap = int(market_cap) if market_cap else None
    except (TypeError, ValueError):
        market_cap = None

    return {
        "pe_ttm": round(pe_ttm, 2) if pe_ttm is not None else None,
        "pe_forward": round(pe_forward, 2) if pe_forward is not None else None,
        "revenue_growth_yoy": _f("revenueGrowth"),
        "earnings_growth_yoy": _f("earningsGrowth"),
        "profit_margin": _f("profitMargins"),
        "market_cap": market_cap,
    }


# ---------- New: technical helpers ----------
def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range — measures average price range per session."""
    high = df["High"]
    low = df["Low"]
    close_prev = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------- New: freshness & acceleration helpers ----------
def compute_days_above_ma50(df: pd.DataFrame) -> Optional[int]:
    """Consecutive days price has been above MA50 (counting backwards from today)."""
    if "MA50" not in df.columns or len(df) < 50:
        return None
    above = df["Close"] > df["MA50"]
    count = 0
    for i in range(len(above) - 1, -1, -1):
        val = above.iloc[i]
        if pd.isna(val) or not val:
            break
        count += 1
    return count


def compute_days_since_ma50_cross_up(df: pd.DataFrame, lookback: int = 60) -> Optional[int]:
    """Days since the most recent upward cross of price above MA50."""
    if "MA50" not in df.columns or len(df) < 51:
        return None
    close = df["Close"]; ma50 = df["MA50"]
    look = min(lookback, len(df) - 1)
    for i in range(look):
        ai = -1 - i; bi = -2 - i
        if bi < -len(df): break
        if pd.isna(ma50.iloc[ai]) or pd.isna(ma50.iloc[bi]): continue
        if close.iloc[ai] > ma50.iloc[ai] and close.iloc[bi] <= ma50.iloc[bi]:
            return i + 1
    return None


def compute_golden_cross_days(df: pd.DataFrame, lookback: int = 60) -> Optional[int]:
    """Days since MA20 crossed above MA50 ('golden cross')."""
    if "MA20" not in df.columns or "MA50" not in df.columns or len(df) < 51:
        return None
    ma20 = df["MA20"]; ma50 = df["MA50"]
    look = min(lookback, len(df) - 1)
    for i in range(look):
        ai = -1 - i; bi = -2 - i
        if bi < -len(df): break
        if pd.isna(ma20.iloc[ai]) or pd.isna(ma50.iloc[ai]): continue
        if pd.isna(ma20.iloc[bi]) or pd.isna(ma50.iloc[bi]): continue
        if ma20.iloc[ai] > ma50.iloc[ai] and ma20.iloc[bi] <= ma50.iloc[bi]:
            return i + 1
    return None


def compute_roc_acceleration(df: pd.DataFrame) -> dict:
    """Rate of change at 5/10/20 days. Acceleration: per-day rate increasing as horizon shortens."""
    if len(df) < 21 or "Close" not in df.columns:
        return {"roc_5d": 0.0, "roc_10d": 0.0, "roc_20d": 0.0, "accelerating": False}
    close = df["Close"]
    last = float(close.iloc[-1])
    if pd.isna(last) or last == 0:
        return {"roc_5d": 0.0, "roc_10d": 0.0, "roc_20d": 0.0, "accelerating": False}

    def roc(p):
        if len(close) <= p: return None
        prev = float(close.iloc[-1 - p])
        if pd.isna(prev) or prev == 0: return None
        return (last / prev - 1) * 100

    r5, r10, r20 = roc(5), roc(10), roc(20)
    if r5 is None or r10 is None or r20 is None:
        return {"roc_5d": r5 or 0.0, "roc_10d": r10 or 0.0,
                "roc_20d": r20 or 0.0, "accelerating": False}

    # Per-day rate of change. Acceleration means recent days outpace older days.
    rate_5  = r5 / 5
    rate_10 = r10 / 10
    rate_20 = r20 / 20
    accelerating = bool(r5 > 0 and rate_5 > rate_10 > rate_20)

    return {"roc_5d": round(r5, 2), "roc_10d": round(r10, 2),
            "roc_20d": round(r20, 2), "accelerating": accelerating}


def compute_volume_trend(df: pd.DataFrame) -> float:
    """Ratio of last 5 days avg volume / last 20 days avg volume. >1.2 = volume expansion."""
    if "Volume" not in df.columns or len(df) < 20:
        return 1.0
    vol = df["Volume"]
    recent = vol.tail(5).mean()
    older = vol.tail(20).mean()
    if pd.isna(older) or older == 0:
        return 1.0
    return round(float(recent / older), 2)


def classify_stage(days_above_ma50: Optional[int],
                   days_since_ma50_cross: Optional[int],
                   golden_cross_days: Optional[int],
                   roc_accelerating: bool,
                   near_52w_high: bool,
                   rsi: float,
                   pct_above_ma20: float) -> str:
    """
    Classify trend stage:
      fresh        — uptrend < 20 days old, or fresh MA50 cross in last 15 days
      accelerating — established but picking up pace (ROC accelerating)
      mature       — well-established, near 52w high, momentum getting tired
      developing   — everything in between
    """
    if days_since_ma50_cross is not None and days_since_ma50_cross <= 15:
        return "fresh"
    if golden_cross_days is not None and golden_cross_days <= 20:
        return "fresh"
    if days_above_ma50 is not None and days_above_ma50 < 20:
        return "fresh"
    if days_above_ma50 is not None and days_above_ma50 > 50 and near_52w_high and rsi > 65:
        return "mature"
    if pct_above_ma20 > 12 and rsi > 70:
        return "mature"
    if roc_accelerating:
        return "accelerating"
    return "developing"


def compute_52w_high_distance(df: pd.DataFrame) -> tuple[float, float]:
    """Returns (high_value, pct_below_high). pct is negative (or 0)."""
    window = df["High"].iloc[-252:] if len(df) >= 252 else df["High"]
    high = float(window.max())
    last = float(df["Close"].iloc[-1])
    pct = (last / high - 1) * 100 if high else 0.0
    return high, pct


def compute_rs_vs_spy(stock_df: pd.DataFrame, spy_df: pd.DataFrame, lookback: int = 30) -> float:
    """% outperformance of stock vs SPY over `lookback` sessions."""
    if len(stock_df) < lookback + 1 or len(spy_df) < lookback + 1:
        return 0.0
    stock_ret = (stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-lookback - 1] - 1) * 100
    spy_ret = (spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[-lookback - 1] - 1) * 100
    return float(stock_ret - spy_ret)


def check_weekly_uptrend(ticker: str) -> Optional[bool]:
    """
    Returns True if weekly trend is also up: close > 10-week MA AND MA rising.
    None if data insufficient or fetch failed.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="1y", interval="1wk", auto_adjust=False)
    except Exception:
        return None
    if len(df) < 12:
        return None
    df["MA10"] = df["Close"].rolling(10).mean()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if pd.isna(last["MA10"]) or pd.isna(prev["MA10"]):
        return None
    return bool(last["Close"] > last["MA10"] and last["MA10"] > prev["MA10"])


def detect_breakout(df: pd.DataFrame,
                    consolidation_window: int = 30,
                    breakout_window: int = 10,
                    max_range_pct: float = 12.0) -> dict:
    """
    Detect: stock spent recent weeks in a tight range, then broke above it.
    Returns {is_breakout, days_since, range_pct}.
    """
    needed = consolidation_window + breakout_window + 2
    if len(df) < needed:
        return {"is_breakout": False, "days_since": None, "range_pct": None}

    cons = df.iloc[-(consolidation_window + breakout_window):-breakout_window]
    cons_high = float(cons["High"].max())
    cons_low = float(cons["Low"].min())
    if cons_low <= 0:
        return {"is_breakout": False, "days_since": None, "range_pct": None}
    range_pct = (cons_high / cons_low - 1) * 100

    if range_pct > max_range_pct:
        return {"is_breakout": False, "days_since": None, "range_pct": round(range_pct, 2)}

    recent = df.iloc[-breakout_window:]
    breakout_idx = None
    for i in range(len(recent)):
        if float(recent["Close"].iloc[i]) > cons_high * 1.01:
            breakout_idx = len(df) - breakout_window + i
            break
    if breakout_idx is None:
        return {"is_breakout": False, "days_since": None, "range_pct": round(range_pct, 2)}

    return {
        "is_breakout": True,
        "days_since": int(len(df) - 1 - breakout_idx),
        "range_pct": round(range_pct, 2),
    }


# ---------- New: insider buying via OpenInsider ----------
def fetch_insider_buying(ticker: str) -> dict:
    """
    Scrape OpenInsider for Form 4 purchase activity in last 30 days.
    Returns {buy_count, total_value_usd}. Failures return zeros.
    """
    base = "http://openinsider.com/screener"
    params = (
        f"?s={ticker}&fd=30&xt=2&grp=0&cnt=20"  # xt=2 → purchase only
    )
    try:
        resp = requests.get(
            base + params,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; daily-picks-bot/1.0)"},
        )
        if resp.status_code != 200:
            return {"buy_count": 0, "total_value_usd": 0}
    except Exception:
        return {"buy_count": 0, "total_value_usd": 0}

    # Parse table without bs4 dependency — use regex on the value column
    # OpenInsider rows look like ... <td>$1,234,567</td> ... in the value cell
    # We also use a simple heuristic: count rows containing the ticker symbol
    # as a row anchor, and pull dollar values.
    text = resp.text
    # Find the screener table block
    table_match = re.search(r'<table[^>]*tinytable[^>]*>(.*?)</table>',
                            text, flags=re.DOTALL | re.IGNORECASE)
    if not table_match:
        return {"buy_count": 0, "total_value_usd": 0}
    table_html = table_match.group(1)
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, flags=re.DOTALL | re.IGNORECASE)

    buy_count = 0
    total_value = 0
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, flags=re.DOTALL | re.IGNORECASE)
        if len(cells) < 13:
            continue
        # Strip HTML tags from each cell
        clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        # Trade type column is usually idx 7-8; "P - Purchase" indicates buy
        joined = " ".join(clean).lower()
        if "purchase" not in joined and "p - " not in joined:
            continue
        # Look for a dollar value in the row
        money_match = re.search(r'\$([\d,]+)', " ".join(clean))
        if money_match:
            try:
                value = int(money_match.group(1).replace(",", ""))
                if value > 0:
                    buy_count += 1
                    total_value += value
            except ValueError:
                continue
    return {"buy_count": buy_count, "total_value_usd": total_value}


# ---------- New: Reddit mentions via public JSON ----------
def fetch_reddit_mentions(ticker: str) -> dict:
    """
    Count Reddit posts mentioning ticker in major investing subs over last week.
    Uses unauthenticated public endpoint — rate limited but free.
    """
    subs = "wallstreetbets+stocks+investing+stockmarket"
    url = (f"https://www.reddit.com/r/{subs}/search.json"
           f"?q=%24{ticker}&restrict_sr=on&t=week&limit=25")
    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "daily-picks-bot/1.0 (research)"},
        )
        if resp.status_code != 200:
            return {"count": 0, "score_total": 0}
        data = resp.json()
    except Exception:
        return {"count": 0, "score_total": 0}
    posts = (data.get("data") or {}).get("children") or []
    count = len(posts)
    score_total = sum((p.get("data") or {}).get("score", 0) for p in posts)
    return {"count": count, "score_total": int(score_total)}


# ---------- New: X / Twitter sentiment ----------
# Tries paid X API (if X_BEARER_TOKEN env var set), falls back to Nitter mirrors.
# Both gracefully degrade to empty result on failure.
NITTER_MIRRORS = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.tiekoetter.com",
    "https://nitter.cz",
    "https://nitter.fdn.fr",
]

def _fetch_x_via_api(ticker: str) -> dict:
    """Use official X API v2. Requires X_BEARER_TOKEN env var (paid tier)."""
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not token:
        return {"count": 0, "sentiment": 0.0, "source": "none"}
    url = "https://api.x.com/2/tweets/search/recent"
    params = {
        "query": f"${ticker} OR #{ticker} -is:retweet lang:en",
        "max_results": "50",
    }
    try:
        resp = requests.get(
            url, params=params, timeout=10,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "daily-picks-bot/1.0",
            },
        )
        if resp.status_code != 200:
            return {"count": 0, "sentiment": 0.0, "source": "none"}
        data = resp.json()
    except Exception as e:
        print(f"  ! X API failed for {ticker}: {e}", file=sys.stderr)
        return {"count": 0, "sentiment": 0.0, "source": "none"}
    tweets = data.get("data") or []
    if not tweets:
        return {"count": 0, "sentiment": 0.0, "source": "api"}
    texts = [t.get("text", "")[:280] for t in tweets if t.get("text")]
    if not texts:
        return {"count": 0, "sentiment": 0.0, "source": "api"}
    sentiment = get_sentiment_scorer().score_texts(texts)
    return {
        "count": len(tweets),
        "sentiment": round(float(sentiment), 3),
        "source": "api",
    }


def _fetch_x_via_nitter(ticker: str) -> dict:
    """Scrape public Nitter mirrors. Best-effort, often unreliable."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    for mirror in NITTER_MIRRORS:
        try:
            url = f"{mirror}/search?f=tweets&q=%24{ticker}"
            resp = requests.get(url, timeout=8, headers=headers)
            if resp.status_code != 200:
                continue
            html = resp.text
            # Extract tweet text from common Nitter HTML structure
            tweets = re.findall(
                r'<div class="tweet-content[^"]*">(.+?)</div>',
                html, flags=re.DOTALL
            )
            if not tweets:
                continue
            texts = []
            for t in tweets[:25]:
                clean = re.sub(r'<[^>]+>', '', t).strip()
                if clean:
                    texts.append(clean[:280])
            if not texts:
                continue
            sentiment = get_sentiment_scorer().score_texts(texts)
            return {
                "count": len(texts),
                "sentiment": round(float(sentiment), 3),
                "source": "nitter",
            }
        except Exception:
            continue
    return {"count": 0, "sentiment": 0.0, "source": "none"}


def fetch_x_mentions(ticker: str) -> dict:
    """
    Returns recent X mentions + FinBERT sentiment.
    Tries paid API first (if X_BEARER_TOKEN set), falls back to Nitter.
    Returns {count, sentiment (-1..+1), source: 'api'|'nitter'|'none'}.
    """
    api_result = _fetch_x_via_api(ticker)
    if api_result["source"] == "api" and api_result["count"] > 0:
        return api_result
    return _fetch_x_via_nitter(ticker)


# ---------- New: FinBERT sentiment scorer ----------
class _SentimentScorer:
    """Tries FinBERT first, falls back to keyword scoring on any failure."""
    def __init__(self):
        self._pipe = None
        if _FINBERT_AVAILABLE and not os.environ.get("DISABLE_FINBERT"):
            try:
                self._pipe = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                    max_length=256,
                )
                print("FinBERT loaded successfully", file=sys.stderr)
            except Exception as e:
                print(f"FinBERT unavailable ({type(e).__name__}: {e}); using keyword fallback",
                      file=sys.stderr)
                self._pipe = None
        else:
            print("FinBERT not installed; using keyword fallback", file=sys.stderr)

    def score_texts(self, texts: list) -> float:
        """Average sentiment in [-1, +1] across `texts`."""
        if not texts:
            return 0.0
        if self._pipe is not None:
            try:
                results = self._pipe(texts)
                scores = []
                for r in results:
                    label = (r.get("label") or "").lower()
                    s = float(r.get("score") or 0)
                    if "positive" in label:
                        scores.append(s)
                    elif "negative" in label:
                        scores.append(-s)
                    else:
                        scores.append(0.0)
                if scores:
                    return float(np.mean(scores))
            except Exception as e:
                print(f"  ! FinBERT inference failed: {e}", file=sys.stderr)
        # Keyword fallback
        raws = [score_headline(t) for t in texts]
        return float(np.tanh(np.mean(raws))) if raws else 0.0


# Singleton scorer (loads model once per run)
_SCORER: Optional[_SentimentScorer] = None
def get_sentiment_scorer() -> _SentimentScorer:
    global _SCORER
    if _SCORER is None:
        _SCORER = _SentimentScorer()
    return _SCORER


def fetch_news_sentiment_finbert(tk: yf.Ticker) -> tuple[int, float]:
    """Same as fetch_news_sentiment but uses FinBERT on title+summary."""
    try:
        news = tk.news or []
    except Exception:
        return 0, 0.0
    if not news:
        return 0, 0.0
    texts = []
    for item in news[:15]:
        content = item.get("content", item)
        title = content.get("title") or item.get("title", "") or ""
        summary = content.get("summary") or item.get("summary", "") or ""
        merged = f"{title}. {summary}".strip()
        if merged:
            texts.append(merged)
    if not texts:
        return 0, 0.0
    avg = get_sentiment_scorer().score_texts(texts)
    return len(texts), float(avg)


# ---------- New: trade setup ----------
def compute_trade_setup(df: pd.DataFrame, entry: float, atr: float,
                        min_rr: float = 2.0, max_target_pct: float = 25.0) -> dict:
    """
    ATR-based stop + target that enforces a minimum risk-reward ratio.

    Old behavior: nearest resistance became the target, giving RR ~0.15 and
    trivial "target hit on day 1 for +0.5%" phantom exits.

    New behavior: walk resistance levels from nearest to farthest, take the
    first one that gives RR >= min_rr. If none qualify within max_target_pct
    of price, synthesize a target at exactly min_rr * stop_distance so RR
    stays honest even without natural overhead resistance.
    """
    stop = entry - 2 * atr
    stop_dist = entry - stop
    if stop_dist <= 0:
        stop_dist = entry * 0.05  # safety: 5% stop minimum
        stop = entry - stop_dist

    # Candidate resistance levels: distinct daily highs from last 60 sessions
    # sitting meaningfully above entry. Sort ascending — try closest first.
    recent_highs = df["High"].iloc[-60:]
    max_target_price = entry * (1 + max_target_pct / 100)
    candidates = sorted({
        round(float(h), 4)
        for h in recent_highs
        if entry * 1.005 < h <= max_target_price
    })

    target = None
    target_source = "synthesized"
    for lvl in candidates:
        rr_here = (lvl - entry) / stop_dist
        if rr_here >= min_rr:
            target = lvl
            target_source = "resistance"
            break

    if target is None:
        # No natural resistance gave enough RR — synthesize exactly min_rr
        target = entry + stop_dist * min_rr

    target_dist = target - entry
    rr = target_dist / stop_dist if stop_dist > 0 else 0.0
    rr_rounded = round(rr, 2)

    return {
        "stop_loss": round(stop, 2),
        "stop_distance_pct": round((stop_dist / entry) * 100, 2),
        "target": round(target, 2),
        "target_distance_pct": round((target_dist / entry) * 100, 2),
        "rr_ratio": rr_rounded,
        "target_source": target_source,               # 'resistance' or 'synthesized'
        "rr_meets_minimum": bool(rr_rounded >= min_rr - 1e-9),  # sanity flag
    }


# ---------- New: market breadth + VIX + HYG ----------
def compute_breadth(technical_signals: list) -> dict:
    """% of universe above their 50-day MA — measures market participation."""
    if not technical_signals:
        return {"pct_above_50dma": None, "count_above": 0, "total": 0}
    above = sum(1 for s in technical_signals if s.price > s.ma50)
    total = len(technical_signals)
    return {
        "pct_above_50dma": round((above / total) * 100, 1),
        "count_above": above,
        "total": total,
    }


def fetch_vix_and_hyg() -> dict:
    """Fetch VIX (volatility) and HYG (junk credit) for regime context."""
    out = {"vix": None, "hyg": None}
    try:
        vix_df = yf.Ticker("^VIX").history(period="1mo", interval="1d")
        if len(vix_df) >= 2:
            level = float(vix_df["Close"].iloc[-1])
            prev = float(vix_df["Close"].iloc[-2])
            out["vix"] = {
                "level": round(level, 2),
                "day_change_pct": round((level / prev - 1) * 100, 2) if prev else 0.0,
                "regime": ("complacent" if level < 15
                           else "stressed" if level > 25 else "normal"),
            }
    except Exception as e:
        print(f"  ! VIX fetch failed: {e}", file=sys.stderr)
    try:
        hyg_df = yf.Ticker("HYG").history(period="3mo", interval="1d")
        if len(hyg_df) >= 50:
            hyg_df["MA50"] = hyg_df["Close"].rolling(50).mean()
            last = hyg_df.iloc[-1]
            ma50_val = float(last["MA50"])
            close = float(last["Close"])
            month_idx = -21 if len(hyg_df) >= 21 else 0
            month_ago = float(hyg_df["Close"].iloc[month_idx])
            out["hyg"] = {
                "price": round(close, 2),
                "above_ma50": bool(close > ma50_val),
                "pct_vs_ma50": round((close / ma50_val - 1) * 100, 2) if ma50_val else 0.0,
                "month_return": round((close / month_ago - 1) * 100, 2) if month_ago else 0.0,
            }
    except Exception as e:
        print(f"  ! HYG fetch failed: {e}", file=sys.stderr)
    return out


def fetch_market_news(top_n: int = 10, pick_tickers: Optional[list] = None) -> dict:
    """
    Aggregate top market news from broad-market ETFs + mega-caps + today's picks.
    Robustly handles multiple yfinance news schemas. Dedupes by URL,
    sorts newest first, runs FinBERT to compute overall mood.
    """
    # Use ETFs + reliable mega-caps + today's picks for broad coverage.
    # Indices like ^GSPC don't carry news in yfinance, so we use ETF proxies.
    proxies = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    if pick_tickers:
        proxies = list(pick_tickers) + proxies
    # Dedupe while preserving order
    seen_proxies = set()
    proxies = [t for t in proxies if not (t in seen_proxies or seen_proxies.add(t))]

    items: list = []
    seen_urls: set = set()
    per_ticker_count = {}

    for sym in proxies:
        try:
            tk = yf.Ticker(sym)
            news = tk.news or []
        except Exception as e:
            print(f"  ! Market news fetch from {sym} failed: {e}", file=sys.stderr)
            continue

        added_this_ticker = 0
        for raw in news:
            content = raw.get("content", raw) if isinstance(raw, dict) else {}
            if not isinstance(content, dict):
                continue

            title = (content.get("title") or raw.get("title") or "").strip()
            summary = (content.get("summary")
                       or content.get("description")
                       or raw.get("summary") or "").strip()

            # URL extraction — try every field Yahoo uses
            url = ""
            for src in (content.get("canonicalUrl"),
                        content.get("clickThroughUrl"),
                        raw.get("canonicalUrl"),
                        raw.get("clickThroughUrl")):
                if isinstance(src, dict):
                    candidate = src.get("url") or ""
                    if candidate:
                        url = candidate
                        break
            if not url:
                url = raw.get("link") or content.get("link") or ""

            # Provider
            provider = ""
            for src in (content.get("provider"), raw.get("provider")):
                if isinstance(src, dict):
                    provider = src.get("displayName") or src.get("name") or ""
                    if provider:
                        break
            if not provider:
                provider = raw.get("publisher") or "Yahoo Finance"

            # Publish time — try ISO string first, then unix timestamp
            pub_time = None
            pd_iso = content.get("pubDate") or content.get("displayTime")
            if pd_iso:
                try:
                    pub_time = int(datetime.fromisoformat(
                        str(pd_iso).replace("Z", "+00:00")).timestamp())
                except Exception:
                    pass
            if pub_time is None:
                pub_time = (raw.get("providerPublishTime")
                            or content.get("providerPublishTime"))

            if not title or not url:
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            items.append({
                "title": title,
                "summary": summary[:280],
                "source": provider,
                "url": url,
                "publish_time": pub_time,
            })
            added_this_ticker += 1

        per_ticker_count[sym] = added_this_ticker

    print(f"  Market news per ticker: {per_ticker_count}", file=sys.stderr)
    print(f"  Total unique headlines collected: {len(items)}", file=sys.stderr)

    # Sort newest first; keep top N
    items.sort(key=lambda x: x.get("publish_time") or 0, reverse=True)
    top_items = items[:top_n]

    # Compute overall headline mood with FinBERT
    overall_mood = 0.0
    if top_items:
        try:
            scorer = get_sentiment_scorer()
            texts = [f"{i['title']}. {i['summary']}" for i in top_items]
            overall_mood = scorer.score_texts(texts)
        except Exception as e:
            print(f"  ! News mood scoring failed: {e}", file=sys.stderr)

    return {
        "items": top_items,
        "overall_mood": round(float(overall_mood), 2),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def fetch_fear_greed() -> Optional[dict]:
    """
    Fetch CNN's Fear & Greed Index (composite of 7 sentiment factors).
    Endpoint is unofficial but widely used. Returns None on failure.
    Score: 0-24 Extreme Fear, 25-44 Fear, 45-55 Neutral, 56-74 Greed, 75-100 Extreme Greed.
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; daily-picks-bot/1.0)",
                "Accept": "application/json",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"  ! Fear & Greed returned {resp.status_code}", file=sys.stderr)
            return None
        data = resp.json()
    except Exception as e:
        print(f"  ! Fear & Greed fetch failed: {e}", file=sys.stderr)
        return None

    fg = data.get("fear_and_greed") or {}
    score = fg.get("score")
    if score is None:
        return None

    def _r(v, n=1):
        try:
            return round(float(v), n)
        except (TypeError, ValueError):
            return None

    return {
        "score": _r(score),
        "rating": fg.get("rating") or "Unknown",
        "previous_close":    _r(fg.get("previous_close")),
        "previous_1_week":   _r(fg.get("previous_1_week")),
        "previous_1_month":  _r(fg.get("previous_1_month")),
        "previous_1_year":   _r(fg.get("previous_1_year")),
    }


# ---------- New: performance tracking ----------
HISTORY_PATH = "public/history.json"
HISTORY_MAX_AGE_DAYS = 365  # prune older entries
COOLDOWN_DAYS = 5             # don't re-pick a ticker within N calendar days
HORIZON_DAYS = 30             # max holding days before forced "horizon" exit
DEFAULT_STOP_PCT = 0.08       # 8% stop for legacy picks missing stop_loss
DEFAULT_TARGET_PCT = 0.15     # 15% target for legacy picks missing target
EXIT_SLIPPAGE_PCT = 0.10      # 0.10% haircut on all exit fills (realistic bid/ask + commission)
EXIT_SIM_VERSION = 2          # bump when sim logic changes so old exits get resimulated

def load_history(path: str = HISTORY_PATH) -> dict:
    if not os.path.exists(path):
        return {"picks": []}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"picks": []}


def save_history(history: dict, path: str = HISTORY_PATH):
    try:
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  ! Failed to save history: {e}", file=sys.stderr)


def update_history_returns(history: dict, spy_df: Optional[pd.DataFrame]) -> dict:
    """For each unsettled pick, compute 1d/3d/5d/30d returns + alpha vs SPY."""
    today = datetime.now(timezone.utc).date()
    picks = history.get("picks") or []
    HORIZONS = [1, 3, 5, 30]

    # Identify picks needing updates
    needs_update = []
    for i, p in enumerate(picks):
        try:
            entry_date = datetime.fromisoformat(p["date"]).date()
        except Exception:
            continue
        days_held = (today - entry_date).days
        if days_held < 1:
            continue
        if all(p.get(f"return_{h}d") is not None for h in HORIZONS):
            continue
        if days_held > 60 and p.get("return_30d") is not None:
            continue
        needs_update.append((i, entry_date, days_held))

    if not needs_update:
        return history

    tickers = sorted({picks[i]["ticker"] for i, _, _ in needs_update})
    print(f"Updating returns for {len(needs_update)} historical picks ({len(tickers)} unique tickers)...",
          file=sys.stderr)

    try:
        if len(tickers) == 1:
            data = yf.Ticker(tickers[0]).history(period="2mo", interval="1d", auto_adjust=False)
            data_dict = {tickers[0]: data}
        else:
            raw = yf.download(tickers, period="2mo", interval="1d",
                              group_by="ticker", auto_adjust=False, progress=False, threads=True)
            data_dict = {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}
    except Exception as e:
        print(f"  ! History update fetch failed: {e}", file=sys.stderr)
        return history

    spy_close = spy_df["Close"] if spy_df is not None else None

    for i, entry_date, days_held in needs_update:
        p = picks[i]
        tk = p["ticker"]
        if tk not in data_dict:
            continue
        df = data_dict[tk]
        if df is None or len(df) == 0:
            continue
        try:
            close_series = df["Close"].dropna()
            after = close_series[close_series.index.date > entry_date]
            entry_price = p.get("entry_price") or 0
            if not entry_price:
                continue
            for h in HORIZONS:
                key = f"return_{h}d"
                if p.get(key) is not None:
                    continue
                if days_held < h:
                    continue
                if len(after) < h:
                    continue
                price_then = float(after.iloc[h - 1])
                ret = (price_then / entry_price - 1) * 100
                # Sanity bound: a single pick > +200% or < -80% is almost certainly
                # a yfinance adjustment glitch (stock split, dividend, delisting).
                if ret < -80 or ret > 300:
                    print(f"  ! {tk} {h}d return {ret:.1f}% looks like a data artifact, skipping",
                          file=sys.stderr)
                    continue
                p[key] = round(ret, 2)
                if spy_close is not None:
                    spy_at = spy_close[spy_close.index.date <= entry_date]
                    spy_after = spy_close[spy_close.index.date > entry_date]
                    if len(spy_at) and len(spy_after) >= h:
                        spy_entry = float(spy_at.iloc[-1])
                        spy_then = float(spy_after.iloc[h - 1])
                        spy_ret = (spy_then / spy_entry - 1) * 100
                        p[f"alpha_{h}d"] = round(ret - spy_ret, 2)
        except Exception as e:
            print(f"  ! {tk}: history calc failed ({e})", file=sys.stderr)

    return history


def update_history_realistic_entries(history: dict) -> dict:
    """
    Realistic entry prices: signal close → next-day open.
    On pick day we only have signal_close. On next workflow run, fetch that
    next session's OPEN and use it as the actual entry_price.
    Cleans up the gap between "what the signal saw" and "what you could trade".
    """
    today = datetime.now(timezone.utc).date()
    picks = history.get("picks") or []

    needs_realistic = []
    for i, p in enumerate(picks):
        # Skip if entry has already been finalized (entry_realistic=True flag)
        if p.get("entry_realistic"):
            continue
        try:
            entry_date = datetime.fromisoformat(p["date"]).date()
        except Exception:
            continue
        days_held = (today - entry_date).days
        if days_held < 1:
            continue
        needs_realistic.append((i, entry_date))

    if not needs_realistic:
        return history

    tickers = sorted({picks[i]["ticker"] for i, _ in needs_realistic})
    print(f"Backfilling realistic entry prices for {len(needs_realistic)} picks...",
          file=sys.stderr)

    try:
        if len(tickers) == 1:
            raw = yf.Ticker(tickers[0]).history(period="2mo", interval="1d", auto_adjust=False)
            data_dict = {tickers[0]: raw}
        else:
            raw = yf.download(tickers, period="2mo", interval="1d",
                              group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
            data_dict = {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}
    except Exception as e:
        print(f"  ! Realistic entry fetch failed: {e}", file=sys.stderr)
        return history

    for i, entry_date in needs_realistic:
        p = picks[i]
        tk = p["ticker"]
        if tk not in data_dict:
            continue
        df = data_dict[tk]
        if df is None or len(df) == 0:
            continue
        try:
            after = df[df.index.date > entry_date]
            if len(after) == 0:
                continue
            next_open = float(after["Open"].iloc[0])
            if pd.isna(next_open) or next_open <= 0:
                continue
            # Preserve the original signal close
            if "signal_close" not in p:
                p["signal_close"] = p.get("entry_price")
            p["entry_price"] = round(next_open, 2)
            p["entry_realistic"] = True
            # Now that entry_price changed, the previously-computed returns are
            # based on signal_close. Clear them so update_history_returns recomputes.
            for k in ("return_1d", "return_3d", "return_5d", "return_30d",
                      "alpha_1d", "alpha_3d", "alpha_5d", "alpha_30d"):
                if k in p:
                    del p[k]
        except Exception as e:
            print(f"  ! {tk}: realistic entry calc failed ({e})", file=sys.stderr)

    return history


def update_history_exits(history: dict) -> dict:
    """
    Walk each historical pick's daily OHLC after entry. Apply the stop_loss
    and target that were set at signal time. Record exit_price, exit_reason,
    exit_return, days_held, exit_slippage_pct.

    v2 changes (August 2026): more honest exit sim.
    - Target hit now requires daily CLOSE > target (not just intraday touch).
      Rationale: an intraday spike touching target for 30 seconds isn't a
      real fill — stop-limit orders often miss, and by close the stock has
      typically retraced. Only counting closes eliminates phantom exits.
    - Stops keep intraday-touch semantics (real stop-market orders DO fill
      on intraday touches — that's their whole point).
    - 0.10% slippage haircut applied to all fills. Real bid/ask + commission.
    - Bumps EXIT_SIM_VERSION so existing sim_version < 2 gets re-simulated.
    """
    today = datetime.now(timezone.utc).date()
    picks = history.get("picks") or []

    needs_exit = []
    for i, p in enumerate(picks):
        # v2 change: pick re-simulation if it was closed under an old sim version
        already_sim_v = p.get("sim_version", 1) if p.get("exit_reason") else None
        if already_sim_v is not None and already_sim_v >= EXIT_SIM_VERSION:
            continue  # already exited under current sim logic
        try:
            entry_date = datetime.fromisoformat(p["date"]).date()
        except Exception:
            continue
        days_held = (today - entry_date).days
        if days_held < 1:
            continue
        entry = p.get("entry_price") or 0
        if not entry:
            continue
        stop = p.get("stop_loss") or entry * (1 - DEFAULT_STOP_PCT)
        target = p.get("target") or entry * (1 + DEFAULT_TARGET_PCT)
        if stop >= entry: stop = entry * (1 - DEFAULT_STOP_PCT)
        if target <= entry: target = entry * (1 + DEFAULT_TARGET_PCT)
        # Clear stale exit fields so re-simulation writes fresh values
        for k in ("exit_date", "exit_price", "exit_return",
                  "exit_reason", "days_held", "exit_slippage_pct"):
            p.pop(k, None)
        needs_exit.append((i, entry_date, entry, stop, target))

    if not needs_exit:
        return history

    tickers = sorted({picks[i]["ticker"] for i, *_ in needs_exit})
    print(f"Simulating stop/target exits for {len(needs_exit)} picks (v{EXIT_SIM_VERSION})...",
          file=sys.stderr)

    try:
        if len(tickers) == 1:
            raw = yf.Ticker(tickers[0]).history(period="3mo", interval="1d", auto_adjust=False)
            data_dict = {tickers[0]: raw}
        else:
            raw = yf.download(tickers, period="3mo", interval="1d",
                              group_by="ticker", auto_adjust=False,
                              progress=False, threads=True)
            data_dict = {t: raw[t] for t in tickers if t in raw.columns.get_level_values(0)}
    except Exception as e:
        print(f"  ! Exit simulation fetch failed: {e}", file=sys.stderr)
        return history

    slip = EXIT_SLIPPAGE_PCT / 100.0

    for i, entry_date, entry, stop, target in needs_exit:
        p = picks[i]
        tk = p["ticker"]
        if tk not in data_dict:
            continue
        df = data_dict[tk]
        if df is None or len(df) == 0:
            continue
        try:
            bars = df[df.index.date > entry_date]
            if len(bars) == 0:
                continue
            for d_idx, (ts, row) in enumerate(bars.iterrows()):
                day_open  = float(row["Open"])  if not pd.isna(row["Open"])  else None
                day_low   = float(row["Low"])   if not pd.isna(row["Low"])   else None
                day_high  = float(row["High"])  if not pd.isna(row["High"])  else None
                day_close = float(row["Close"]) if not pd.isna(row["Close"]) else None
                if None in (day_open, day_low, day_high, day_close):
                    continue

                raw_exit_price = None
                exit_reason = None

                # --- STOP LOGIC (intraday touch — realistic for stop-market orders) ---
                # Gap-down through stop: exit at the open
                if day_open <= stop:
                    raw_exit_price = day_open
                    exit_reason = "stop_gap"
                # Stop hit intraday
                elif day_low <= stop:
                    raw_exit_price = stop
                    exit_reason = "stop"
                # --- TARGET LOGIC (v2: requires CLOSE, not just intraday touch) ---
                # Gap-up open above target AND closed there: honest target hit at open
                elif day_open >= target and day_close >= target:
                    raw_exit_price = day_open
                    exit_reason = "target_gap"
                # Intraday touch of target AND closed above: honest target hit
                elif day_high >= target and day_close >= target:
                    raw_exit_price = target
                    exit_reason = "target"

                if raw_exit_price is not None:
                    # Apply slippage: sells fill worse than the raw level
                    exit_price = raw_exit_price * (1 - slip)
                    p["exit_date"]         = ts.date().isoformat()
                    p["exit_price"]        = round(exit_price, 2)
                    p["exit_return"]       = round((exit_price / entry - 1) * 100, 2)
                    p["exit_reason"]       = exit_reason
                    p["days_held"]         = d_idx + 1
                    p["exit_slippage_pct"] = EXIT_SLIPPAGE_PCT
                    p["sim_version"]       = EXIT_SIM_VERSION
                    break
            else:
                # No exit within available bars. If held longer than horizon,
                # force horizon exit at day-30 close (with slippage).
                if len(bars) >= HORIZON_DAYS:
                    h_close = float(bars["Close"].iloc[HORIZON_DAYS - 1])
                    exit_price = h_close * (1 - slip)
                    p["exit_date"]         = bars.index[HORIZON_DAYS - 1].date().isoformat()
                    p["exit_price"]        = round(exit_price, 2)
                    p["exit_return"]       = round((exit_price / entry - 1) * 100, 2)
                    p["exit_reason"]       = "horizon"
                    p["days_held"]         = HORIZON_DAYS
                    p["exit_slippage_pct"] = EXIT_SLIPPAGE_PCT
                    p["sim_version"]       = EXIT_SIM_VERSION
        except Exception as e:
            print(f"  ! {tk}: exit simulation failed ({e})", file=sys.stderr)

    return history


def get_cooldown_tickers(history: dict, days: int = COOLDOWN_DAYS) -> set:
    """Tickers picked within the last N calendar days — exclude from selection."""
    today = datetime.now(timezone.utc).date()
    out = set()
    for p in history.get("picks") or []:
        try:
            d = datetime.fromisoformat(p["date"]).date()
            if (today - d).days < days:
                out.add(p["ticker"])
        except Exception:
            continue
    return out


def append_picks_to_history(history: dict, new_picks: list, today_iso: str) -> dict:
    if "picks" not in history:
        history["picks"] = []
    # Prune: drop picks older than HISTORY_MAX_AGE_DAYS
    cutoff = datetime.now(timezone.utc).date()
    fresh = []
    for p in history["picks"]:
        try:
            d = datetime.fromisoformat(p["date"]).date()
            if (cutoff - d).days <= HISTORY_MAX_AGE_DAYS:
                fresh.append(p)
        except Exception:
            fresh.append(p)
    history["picks"] = fresh

    # Append new picks (deduplicated by date+ticker so manual re-runs don't double-up)
    existing = {(p.get("date"), p.get("ticker")) for p in history["picks"]}
    for s in new_picks:
        key = (today_iso, s.ticker)
        if key in existing:
            continue
        history["picks"].append({
            "date": today_iso,
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "entry_price": s.price,         # signal close — will be replaced
            "signal_close": s.price,        # immutable record of the signal price
            "entry_realistic": False,       # flag: not yet backfilled to next-day open
            "score": s.composite_score,
            "stop_loss": s.stop_loss,
            "target": s.target,
            "rr_ratio": s.rr_ratio,
            "key_signals": {
                "rs_vs_spy_30d": s.rs_vs_spy_30d,
                "weekly_uptrend": s.weekly_uptrend,
                "near_52w_high": s.near_52w_high,
                "breakout_recent": s.breakout_recent,
                "atr_pct": s.atr_pct,
                "insider_buy": s.insider_buy_count_30d > 0,
                "rs_leader": s.rs_leader,
            },
        })
    return history


def fetch_sector_etfs() -> list:
    """
    Pull recent prices for the 11 SPDR sector ETFs. Returns a list of dicts
    sorted by today's return (descending). Logs failures to stderr.
    """
    rows = []
    failures = []
    for sym, name in SECTOR_ETFS.items():
        try:
            tk = yf.Ticker(sym)
            df = tk.history(period="3mo", interval="1d", auto_adjust=False)
            if len(df) < 25:
                failures.append(f"{sym}: only {len(df)} rows of history")
                continue
            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            month_idx = -21 if len(df) >= 21 else 0
            month = float(df["Close"].iloc[month_idx])
            day_ret = (last / prev - 1) * 100 if prev else 0.0
            month_ret = (last / month - 1) * 100 if month else 0.0
            rows.append({
                "symbol": sym,
                "name": name,
                "price": round(last, 2),
                "day_return": round(day_ret, 2),
                "month_return": round(month_ret, 2),
                "is_cyclical": sym in CYCLICAL_SECTORS,
                "is_defensive": sym in DEFENSIVE_SECTORS,
            })
        except Exception as e:
            failures.append(f"{sym}: {type(e).__name__}: {e}")
            continue
    for f in failures:
        print(f"  ! Sector fetch failure: {f}", file=sys.stderr)
    print(f"Sector ETFs: {len(rows)}/{len(SECTOR_ETFS)} fetched successfully", file=sys.stderr)
    rows.sort(key=lambda r: r["day_return"], reverse=True)
    return rows


def compute_regime(sectors: list) -> dict:
    """
    Determine risk-on / risk-off / neutral by comparing average daily returns
    of cyclical sectors to defensive sectors.
    """
    cyc = [s["day_return"] for s in sectors if s["is_cyclical"]]
    defs = [s["day_return"] for s in sectors if s["is_defensive"]]
    if not cyc or not defs:
        return {"label": "neutral", "score": 0.0, "cyclical_avg": 0.0,
                "defensive_avg": 0.0, "summary": "Insufficient sector data"}
    cyc_avg = sum(cyc) / len(cyc)
    def_avg = sum(defs) / len(defs)
    spread = cyc_avg - def_avg
    if spread > 0.4:
        label = "risk-on"
        summary = (f"Cyclicals are leading defensives by {spread:+.2f}%. "
                   "Money is rotating into growth — favorable backdrop for trend setups.")
    elif spread < -0.4:
        label = "risk-off"
        summary = (f"Defensives are outperforming cyclicals by {-spread:+.2f}%. "
                   "Capital is rotating to safety — be cautious chasing breakouts today.")
    else:
        label = "neutral"
        summary = (f"Cyclicals {cyc_avg:+.2f}% vs defensives {def_avg:+.2f}% — "
                   "no clear regime. Treat picks with normal sizing, no conviction premium.")
    return {
        "label": label,
        "score": round(spread, 2),
        "cyclical_avg": round(cyc_avg, 2),
        "defensive_avg": round(def_avg, 2),
        "summary": summary,
    }


def evaluate(ticker: str, enrich: bool = True, spy_df=None) -> Optional[Signal]:
    """Evaluate one ticker. Heavy lookups deferred unless enrich=True."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="1y", interval="1d", auto_adjust=False)
        if len(df) < 50:
            return None
    except Exception as e:
        print(f"  ! {ticker}: data fetch failed ({e})", file=sys.stderr)
        return None

    df["MA20"] = sma(df["Close"], 20)
    df["MA50"] = sma(df["Close"], 50)
    df["RSI"] = rsi(df["Close"], 14)
    df["VolAvg20"] = sma(df["Volume"], 20)
    df["ATR"] = compute_atr(df, 14)
    macd_line, sig_line, hist = macd(df["Close"])
    df["MACD_hist"] = hist

    last = df.iloc[-1]
    prev = df.iloc[-2]
    if any(pd.isna(x) for x in [last["MA20"], last["MA50"], last["RSI"], last["VolAvg20"], last["ATR"]]):
        return None

    pct_above_ma20 = float((last["Close"] / last["MA20"] - 1) * 100)
    pct_above_ma50 = float((last["Close"] / last["MA50"] - 1) * 100)
    ma_separation = float((last["MA20"] / last["MA50"] - 1) * 100)
    if len(df) > 6 and not pd.isna(df["MA20"].iloc[-6]):
        ma20_slope_5d = float((last["MA20"] / df["MA20"].iloc[-6] - 1) * 100)
    else:
        ma20_slope_5d = 0.0

    in_uptrend = bool(last["Close"] > last["MA20"] > last["MA50"])
    ma20_rising = bool(ma20_slope_5d > 0.3)

    high_52w, pct_below_52w = compute_52w_high_distance(df)
    near_52w_high = pct_below_52w >= -15

    atr_14 = float(last["ATR"])
    atr_pct = float(atr_14 / last["Close"] * 100) if last["Close"] else 0.0

    rs_30d = compute_rs_vs_spy(df, spy_df, 30) if spy_df is not None else 0.0
    rs_leader = rs_30d > 3.0

    bo = detect_breakout(df)

    cross_today = bool(last["Close"] > last["MA20"] and prev["Close"] <= prev["MA20"])
    cross_recent = False
    for i in range(1, 11):
        if i + 1 > len(df):
            break
        a = df.iloc[-i]; b = df.iloc[-i - 1]
        if a["Close"] > a["MA20"] and b["Close"] <= b["MA20"]:
            cross_recent = True; break

    rsi_now = float(last["RSI"])
    rsi_rising = bool(last["RSI"] > prev["RSI"])
    macd_h = float(last["MACD_hist"])
    macd_positive = bool(macd_h > 0 and macd_h > prev["MACD_hist"])
    vol_ratio = float(last["Volume"] / last["VolAvg20"]) if last["VolAvg20"] else 0
    vol_strong = vol_ratio >= 1.5

    # Freshness & acceleration metrics
    days_above_ma50 = compute_days_above_ma50(df)
    days_since_ma50_cross = compute_days_since_ma50_cross_up(df, lookback=90)
    golden_cross_days = compute_golden_cross_days(df, lookback=90)
    roc = compute_roc_acceleration(df)
    vol_trend_ratio = compute_volume_trend(df)
    stage = classify_stage(days_above_ma50, days_since_ma50_cross,
                           golden_cross_days, roc["accelerating"],
                           near_52w_high, rsi_now, pct_above_ma20)

    if enrich:
        sector = fetch_sector(tk)
        fundamentals = fetch_fundamentals(tk)
        news_count, news_score = fetch_news_sentiment_finbert(tk)
        st_messages, st_bull_ratio, st_tagged = fetch_stocktwits(ticker)
        days_to_earn = fetch_days_to_earnings(tk)
        weekly_uptrend = check_weekly_uptrend(ticker)
        insider = fetch_insider_buying(ticker)
        reddit = fetch_reddit_mentions(ticker)
        x_data = fetch_x_mentions(ticker)
        time.sleep(0.3)
    else:
        sector = "Unknown"
        fundamentals = {
            "pe_ttm": None, "pe_forward": None,
            "revenue_growth_yoy": None, "earnings_growth_yoy": None,
            "profit_margin": None, "market_cap": None,
        }
        news_count, news_score = 0, 0.0
        st_messages, st_bull_ratio, st_tagged = 0, 0.5, 0
        days_to_earn = None
        weekly_uptrend = None
        insider = {"buy_count": 0, "total_value_usd": 0}
        reddit = {"count": 0, "score_total": 0}
        x_data = {"count": 0, "sentiment": 0.0, "source": "none"}

    score = 0.0
    reasons = []

    if in_uptrend and ma20_rising:
        score += 3.0
        reasons.append(f"In uptrend: price > MA20 > MA50, MA20 rising {ma20_slope_5d:+.2f}% over 5 sessions")

    # --- FRESHNESS BIAS: reward early-stage trends heavily ---
    if days_since_ma50_cross is not None and days_since_ma50_cross <= 15:
        score += 3.0
        reasons.append(f"Fresh MA50 reclaim: price crossed above {days_since_ma50_cross} sessions ago")
    elif days_since_ma50_cross is not None and days_since_ma50_cross <= 30:
        score += 1.5
        reasons.append(f"Recent MA50 reclaim ({days_since_ma50_cross} sessions ago)")

    if golden_cross_days is not None and golden_cross_days <= 20:
        score += 2.5
        reasons.append(f"Golden cross {golden_cross_days} sessions ago: MA20 crossed above MA50")
    elif golden_cross_days is not None and golden_cross_days <= 45:
        score += 1.0

    # --- ACCELERATION BIAS: reward trends picking up pace ---
    if roc["accelerating"]:
        score += 2.5
        reasons.append(f"Acceleration: 5d {roc['roc_5d']:+.1f}%, 10d {roc['roc_10d']:+.1f}%, 20d {roc['roc_20d']:+.1f}% — pace rising")

    if vol_trend_ratio > 1.3:
        score += 1.5
        reasons.append(f"Volume expanding: last 5 sessions averaging {vol_trend_ratio:.2f}× the 20-session base")
    elif vol_trend_ratio > 1.1:
        score += 0.5

    # --- MATURITY PENALTIES: discount late-stage setups ---
    if days_above_ma50 is not None and days_above_ma50 > 60:
        score -= 0.5
        reasons.append(f"Trend is {days_above_ma50}+ sessions old — late-stage, less room to run")

    if rsi_now > 72 and pct_above_ma20 > 8:
        score -= 1.5
        reasons.append(f"Overbought + extended (RSI {rsi_now:.1f}, +{pct_above_ma20:.1f}% above MA20) — pullback risk")

    if near_52w_high and rsi_now > 68 and (days_above_ma50 or 0) > 40:
        score -= 1.0
        reasons.append("Near 52w high + elevated RSI + mature trend — classic topping setup")

    # --- Existing signals (reduced weights on maturity-favoring ones) ---
    if rs_leader:
        score += 1.0  # reduced from 2.0
        reasons.append(f"Leadership: outperforming SPY by {rs_30d:+.2f}% over 30 sessions")
    elif rs_30d > 0:
        score += 0.3

    if enrich and weekly_uptrend is True:
        score += 1.5
        reasons.append("Weekly trend confirms: above 10-week MA and rising")
    elif enrich and weekly_uptrend is False:
        score -= 1.5
        reasons.append("Weekly trend is down — daily setup fights the bigger tide")

    if near_52w_high:
        score += 0.5  # reduced from 1.0 — less reward for already-extended names
        reasons.append(f"Near 52-week high (only {-pct_below_52w:.1f}% below)")
    elif -30 < pct_below_52w < -10:
        # Sweet spot: room to run, not yet broken out
        score += 0.5
        reasons.append(f"{-pct_below_52w:.1f}% below 52w high — runway intact")
    elif pct_below_52w < -30:
        # Far below 52w but if we've got fresh reversal signals, this is upside
        if (days_since_ma50_cross is not None and days_since_ma50_cross <= 20):
            reasons.append(f"{-pct_below_52w:.1f}% below 52w high BUT showing fresh reversal — turnaround setup")
        else:
            score -= 1.0
            reasons.append(f"{-pct_below_52w:.1f}% below 52-week high — long-term downtrend")

    if 0 < pct_above_ma20 <= 3:
        score += 2.0
        reasons.append(f"Healthy pullback: only +{pct_above_ma20:.2f}% above MA20")
    elif 3 < pct_above_ma20 <= 8:
        score += 1.0
        reasons.append(f"+{pct_above_ma20:.2f}% above MA20 — moderate extension")
    elif 8 < pct_above_ma20 <= 15:
        reasons.append(f"+{pct_above_ma20:.2f}% above MA20 — extended")
    elif pct_above_ma20 > 15:
        score -= 1.5
        reasons.append(f"+{pct_above_ma20:.2f}% above MA20 — overextended")

    if ma_separation >= 4:
        score += 1.5
        reasons.append(f"Established trend: MA20 +{ma_separation:.2f}% above MA50")
    elif ma_separation >= 1.5:
        score += 0.5

    if 50 <= rsi_now <= 75 and rsi_rising:
        score += 1.0
        reasons.append(f"RSI {rsi_now:.1f} and rising")
    elif rsi_now > 80:
        score -= 0.5
        reasons.append(f"RSI {rsi_now:.1f} extreme — wait for cooldown")

    if macd_positive:
        score += 1.0
        reasons.append(f"MACD histogram positive and expanding ({macd_h:+.3f})")

    if vol_strong:
        score += 1.5
        reasons.append(f"Volume {vol_ratio:.2f}x 20-day average")
    elif vol_ratio >= 1.0:
        score += 0.3

    if atr_pct < 1.0:
        score -= 0.5
        reasons.append(f"ATR only {atr_pct:.2f}% of price — low volatility")
    elif atr_pct > 6.0:
        score -= 1.0
        reasons.append(f"ATR {atr_pct:.2f}% of price — too volatile")

    if bo["is_breakout"] and bo["days_since"] is not None and bo["days_since"] <= 5:
        score += 2.0
        reasons.append(f"Fresh breakout from {bo['range_pct']:.1f}% consolidation ({bo['days_since']} sessions ago)")

    if cross_recent and pct_above_ma20 < 6:
        score += 0.3

    if enrich and st_tagged >= 5:
        if st_bull_ratio >= 0.70:
            score += 2.0
            reasons.append(f"StockTwits {st_bull_ratio*100:.0f}% bullish across {st_tagged} tagged messages")
        elif st_bull_ratio >= 0.55:
            score += 1.0
        elif st_bull_ratio <= 0.30:
            score -= 1.5
            reasons.append(f"StockTwits {(1-st_bull_ratio)*100:.0f}% bearish")
    if enrich and st_messages >= 25:
        score += 0.5

    if enrich and reddit["count"] >= 5:
        if reddit["score_total"] >= 200:
            score += 1.5
            reasons.append(f"Reddit buzz: {reddit['count']} posts with {reddit['score_total']} upvotes")
        else:
            score += 0.5
            reasons.append(f"Reddit chatter: {reddit['count']} recent posts")

    # X / Twitter sentiment (NEW)
    if enrich and x_data["count"] >= 5 and x_data["source"] != "none":
        x_sent = x_data["sentiment"]
        if x_sent > 0.30:
            score += 1.5
            reasons.append(f"X sentiment strongly positive across {x_data['count']} mentions (FinBERT {x_sent:+.2f})")
        elif x_sent > 0.10:
            score += 0.7
            reasons.append(f"X sentiment positive across {x_data['count']} mentions")
        elif x_sent < -0.30:
            score -= 1.5
            reasons.append(f"X sentiment strongly negative across {x_data['count']} mentions")
        elif x_sent < -0.10:
            score -= 0.7
        if x_data["count"] >= 25:
            # High X discussion volume bonus
            score += 0.3

    if news_score > 0.30:
        score += 1.5
        reasons.append(f"{news_count} headlines, strongly positive (FinBERT {news_score:+.2f})")
    elif news_score > 0.10:
        score += 0.7
        reasons.append(f"{news_count} headlines, mildly positive (FinBERT {news_score:+.2f})")
    elif news_score < -0.30:
        score -= 2.0
        reasons.append(f"{news_count} headlines, strongly negative (FinBERT {news_score:+.2f})")
    elif news_score < -0.10:
        score -= 1.0

    if enrich and insider["buy_count"] >= 1:
        if insider["total_value_usd"] >= 1_000_000:
            score += 2.0
            reasons.append(f"Insider buying: {insider['buy_count']} purchase(s) totaling ${insider['total_value_usd']:,} in last 30d")
        else:
            score += 1.0
            reasons.append(f"Some insider buying: {insider['buy_count']} purchase(s) recently")

    if enrich and days_to_earn is not None and 0 <= days_to_earn <= 3:
        reasons.append(f"⚠ Earnings in {days_to_earn} day(s) — excluded from final picks")

    name = ticker
    try:
        info_name = tk.info.get("shortName") or tk.info.get("longName")
        if info_name:
            name = info_name
    except Exception:
        pass

    chart_df = df.tail(60)
    chart_data = []
    for ts, row in chart_df.iterrows():
        ma20_val = row["MA20"]; ma50_val = row["MA50"]
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

    # Compute trade setup (stop-loss, target, R:R)
    setup = compute_trade_setup(df, float(last["Close"]), atr_14)

    return Signal(
        ticker=ticker, name=name, sector=sector,
        price=float(last["Close"]), ma20=float(last["MA20"]), ma50=float(last["MA50"]),
        cross_today=cross_today, cross_recent=cross_recent,
        pct_above_ma20=pct_above_ma20, pct_above_ma50=pct_above_ma50,
        ma20_slope_5d=ma20_slope_5d,
        in_uptrend=in_uptrend, ma20_rising=ma20_rising,
        rsi=rsi_now, rsi_rising=rsi_rising,
        macd_hist=macd_h, macd_positive=macd_positive,
        volume_ratio=vol_ratio, volume_strong=vol_strong,
        rs_vs_spy_30d=round(rs_30d, 2), rs_leader=rs_leader,
        weekly_uptrend=weekly_uptrend,
        pct_below_52w_high=round(pct_below_52w, 2), near_52w_high=near_52w_high,
        atr_14=round(atr_14, 4), atr_pct=round(atr_pct, 2),
        breakout_recent=bo["is_breakout"],
        breakout_days_since=bo["days_since"],
        consolidation_range_pct=bo["range_pct"],
        news_count_7d=news_count, news_score=news_score,
        st_message_count=st_messages, st_bullish_ratio=st_bull_ratio, st_tagged=st_tagged,
        reddit_mention_count=reddit["count"], reddit_score_total=reddit["score_total"],
        insider_buy_count_30d=insider["buy_count"],
        insider_buy_value_usd=insider["total_value_usd"],
        x_mention_count=x_data["count"],
        x_sentiment=x_data["sentiment"],
        x_source=x_data["source"],
        days_to_earnings=days_to_earn,
        composite_score=round(score, 2),
        reasons=reasons,
        chart_data=chart_data,
        stop_loss=setup["stop_loss"],
        stop_distance_pct=setup["stop_distance_pct"],
        target=setup["target"],
        target_distance_pct=setup["target_distance_pct"],
        rr_ratio=setup["rr_ratio"],
        pe_ttm=fundamentals["pe_ttm"],
        pe_forward=fundamentals["pe_forward"],
        revenue_growth_yoy=fundamentals["revenue_growth_yoy"],
        earnings_growth_yoy=fundamentals["earnings_growth_yoy"],
        profit_margin=fundamentals["profit_margin"],
        market_cap=fundamentals["market_cap"],
        days_above_ma50=days_above_ma50,
        days_since_ma50_cross=days_since_ma50_cross,
        golden_cross_days=golden_cross_days,
        roc_5d=roc["roc_5d"],
        roc_10d=roc["roc_10d"],
        roc_20d=roc["roc_20d"],
        roc_accelerating=roc["accelerating"],
        volume_trend_ratio=vol_trend_ratio,
        stage=stage,
    )


def select_top(signals: list, n: int = 3, max_per_sector: int = 1, earnings_window: int = 3) -> list:
    """Pick top N: skip earnings within window, max per sector, exclude weekly downtrend."""
    eligible = [s for s in signals if s.in_uptrend and s.ma20_rising]
    eligible = [s for s in eligible
                if s.days_to_earnings is None or s.days_to_earnings > earnings_window]
    eligible = [s for s in eligible if s.weekly_uptrend is not False]
    eligible.sort(key=lambda s: s.composite_score, reverse=True)
    chosen = []
    sector_count = {}
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

    # Fetch sectors FIRST while yfinance is fresh — these calls fail when
    # we put them after the heavy 100-stock scan (Yahoo throttles).
    print("Fetching sector ETFs for market regime...", file=sys.stderr)
    sectors = fetch_sector_etfs()
    regime = compute_regime(sectors)

    # Fetch SPY history once for relative strength comparisons
    print("Fetching SPY benchmark for relative strength...", file=sys.stderr)
    try:
        spy_df = yf.Ticker("SPY").history(period="1y", interval="1d", auto_adjust=False)
        if len(spy_df) < 50:
            spy_df = None
    except Exception:
        spy_df = None

    # Pre-warm FinBERT once (model load is the expensive part, not inference)
    print("Initializing sentiment scorer...", file=sys.stderr)
    get_sentiment_scorer()

    print(f"Scanning {len(tickers)} tickers (technicals only)...", file=sys.stderr)
    technical_signals = []
    for t in tickers:
        s = evaluate(t, enrich=False, spy_df=spy_df)
        if s is not None:
            technical_signals.append(s)

    candidates = [s for s in technical_signals if s.in_uptrend and s.ma20_rising]
    candidates.sort(key=lambda s: s.composite_score, reverse=True)
    enrich_count = min(len(candidates), max(args.top * 4, 8))
    top_candidates = candidates[:enrich_count]
    print(f"Enriching {len(top_candidates)} candidates with sentiment/earnings/sector/insider/reddit/weekly...", file=sys.stderr)

    signals = []
    for s in top_candidates:
        full = evaluate(s.ticker, enrich=True, spy_df=spy_df)
        if full is not None:
            signals.append(full)

    # Load history EARLY so cooldown can filter selection.
    history = load_history()

    # Apply ticker cooldown: don't re-pick anything we picked in the last COOLDOWN_DAYS days.
    cooldown = get_cooldown_tickers(history, days=COOLDOWN_DAYS)
    if cooldown:
        before_n = len(signals)
        excluded = [s for s in signals if s.ticker in cooldown]
        signals = [s for s in signals if s.ticker not in cooldown]
        if excluded:
            print(f"Cooldown filter ({COOLDOWN_DAYS}d): excluded {len(excluded)} ticker(s) — "
                  f"{', '.join(sorted(s.ticker for s in excluded))}",
                  file=sys.stderr)
        print(f"  {len(signals)}/{before_n} candidates remain after cooldown", file=sys.stderr)

    picks = select_top(signals, n=args.top)

    # Compute market breadth from the universe scan we already have
    breadth = compute_breadth(technical_signals)

    # Fetch VIX and HYG for richer regime context
    print("Fetching VIX and HYG for regime context...", file=sys.stderr)
    market_health = fetch_vix_and_hyg()

    # Fetch CNN Fear & Greed Index (composite of 7 sentiment factors)
    print("Fetching Fear & Greed Index...", file=sys.stderr)
    fear_greed = fetch_fear_greed()

    # Fetch market headlines (include today's pick tickers for relevance)
    print("Fetching top market headlines...", file=sys.stderr)
    pick_tickers = [p.ticker for p in picks] if picks else None
    market_news = fetch_market_news(top_n=10, pick_tickers=pick_tickers)

    # Combine into enhanced regime payload
    regime["breadth"] = breadth
    regime["vix"] = market_health.get("vix")
    regime["hyg"] = market_health.get("hyg")
    regime["fear_greed"] = fear_greed

    # ----- Performance tracking pipeline -----
    today_iso = datetime.now(timezone.utc).date().isoformat()
    # 1. Backfill realistic entries (signal close → next-day open) for yesterday's picks.
    history = update_history_realistic_entries(history)
    # 2. Recompute raw returns at 1d / 3d / 5d / 30d.
    history = update_history_returns(history, spy_df)
    # 3. Simulate stop & target exits using each pick's original setup.
    history = update_history_exits(history)
    # 4. Append today's new picks (entry_price = today's close; entry_realistic=False).
    history = append_picks_to_history(history, picks, today_iso)
    save_history(history)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market": args.market,
        "scanned": len(technical_signals),
        "enriched": len(signals),
        "universe_size": len(tickers),
        "picks": [asdict(p) for p in picks],
        "sectors": sectors,
        "regime": regime,
        "market_news": market_news,
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
