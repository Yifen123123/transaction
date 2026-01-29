#執行程式 python3 tech_analysis_b.py  --ticker AAPL --period 2y

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys
from datetime import datetime
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from backtest_module import BacktestConfig, run_backtest
from draw import plot_trades_on_price
from draw import plot_equity_and_drawdown


# ---- 中文字型（你已經設過可保留/修改）----
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    p = argparse.ArgumentParser("技術指標：Yahoo + Stooq 後備，含 MA/MACD/RSI + BBands + KDJ")
    p.add_argument("--ticker", default="AAPL", help="股票代號，例如 AAPL / TSM")
    p.add_argument("--period", default="2y", help="1y/2y/6mo/5d；若提供 start/end 則忽略")
    p.add_argument("--interval", default="1d", help="1d/1wk/1h …")
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--no-plot", dest="no_plot", action="store_true", help="只抓資料不畫圖（除錯用）")
    return p.parse_args()

def print_versions():
    import yfinance, pandas, numpy
    print(f"yfinance: {yfinance.__version__} | pandas: {pandas.__version__} | numpy: {numpy.__version__}")

def fetch_yahoo(ticker, period, interval, start, end) -> pd.DataFrame:
    import yfinance as yf
    print(f"嘗試 Yahoo 下載 {ticker} | interval={interval} | "
          f"{('start='+start+' end='+str(end)) if start else ('period='+period)}")
    if start:
        return yf.download(ticker, start=start, end=end, interval=interval, progress=False, threads=False)
    else:
        return yf.download(ticker, period=period, interval=interval, progress=False, threads=False)

def fetch_stooq(ticker) -> pd.DataFrame:
    from pandas_datareader import data as pdr
    sym = ticker if ticker.endswith(".US") else f"{ticker}.US"
    print(f"改用 Stooq 下載 {sym}")
    return pdr.DataReader(sym, "stooq").sort_index()

def normalize_single_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        last_level = df.columns.get_level_values(-1)
        if ticker in last_level:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df = df.T.groupby(level=0).first().T
    return df

def choose_price_column(df: pd.DataFrame) -> str:
    if "Adj Close" in df.columns and not df["Adj Close"].isna().all():
        return "Adj Close"
    return "Close"

def series_from(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

def add_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    加入：
    - MA20
    - RSI(14)
    - MACD(12,26,9)
    - Bollinger Bands(20, 2)：BB_MID/UPPER/LOWER/BB_PCT
    - KDJ(9,3,3)：K、D、J
    """
    import ta
    out = df.copy()

    price = series_from(out, price_col)
    high  = series_from(out, "High")
    low   = series_from(out, "Low")

    # --- MA20 ---
    out["MA20"] = price.rolling(20, min_periods=20).mean()

    # --- RSI(14) ---
    out["RSI"] = ta.momentum.RSIIndicator(price, window=14).rsi()

    # --- MACD(12,26,9) ---
    macd = ta.trend.MACD(price, window_fast=12, window_slow=26, window_sign=9)
    out["MACD"] = macd.macd()
    out["Signal"] = macd.macd_signal()

    # --- Bollinger Bands(20, 2) ---
    bb = ta.volatility.BollingerBands(close=price, window=20, window_dev=2)
    out["BB_MID"]   = bb.bollinger_mavg()
    out["BB_UPPER"] = bb.bollinger_hband()
    out["BB_LOWER"] = bb.bollinger_lband()
    out["BB_PCT"]   = bb.bollinger_pband()  # (Close - MID) / (UPPER - LOWER)

    # --- KDJ(9,3,3) ---
    # 使用 Stochastic Oscillator 作為 K、D，J = 3K - 2D
    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=price, window=9, smooth_window=3)
    out["K"] = stoch.stoch()         # %K
    out["D"] = stoch.stoch_signal()  # %D
    out["J"] = 3 * out["K"] - 2 * out["D"]

    return out

def plot_price_ma_bbands(df: pd.DataFrame, price_col: str, title: str):
    dfp = df.dropna(subset=[price_col, "MA20"])
    if dfp.empty:
        print("⚠️ 價格/MA 資料不足，無法畫圖。")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(dfp.index, dfp[price_col], label=price_col)
    plt.plot(dfp.index, dfp["MA20"], label="MA20")

    # 如果有 BBands，就一併畫
    if {"BB_UPPER", "BB_MID", "BB_LOWER"}.issubset(dfp.columns):
        plt.plot(dfp.index, dfp["BB_MID"], label="BB中軌")
        plt.plot(dfp.index, dfp["BB_UPPER"], label="BB上軌")
        plt.plot(dfp.index, dfp["BB_LOWER"], label="BB下軌")
        # 以區域填滿帶狀
        plt.fill_between(dfp.index, dfp["BB_LOWER"], dfp["BB_UPPER"], alpha=0.1, label="布林帶")

    plt.title(f"{title} - 價格 / MA20 / Bollinger Bands")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_macd(df: pd.DataFrame):
    need = ["MACD", "Signal"]
    if not set(need).issubset(df.columns):
        print("⚠️ MACD 欄位不足，略過 MACD 圖。")
        return
    dfp = df.dropna(subset=need)
    if dfp.empty:
        print("⚠️ MACD 資料不足。")
        return

    plt.figure(figsize=(14, 4))
    plt.plot(dfp.index, dfp["MACD"], label="MACD")
    plt.plot(dfp.index, dfp["Signal"], label="Signal")
    plt.title("MACD")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rsi(df: pd.DataFrame):
    if "RSI" not in df.columns:
        print("⚠️ 沒有 RSI 欄位，略過。")
        return
    dfp = df.dropna(subset=["RSI"])
    if dfp.empty:
        print("⚠️ RSI 資料不足。")
        return

    plt.figure(figsize=(14, 4))
    plt.plot(dfp.index, dfp["RSI"], label="RSI")
    plt.axhline(70, ls="--")
    plt.axhline(30, ls="--")
    plt.title("RSI")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_kdj(df: pd.DataFrame):
    need = ["K", "D", "J"]
    if not set(need).issubset(df.columns):
        print("⚠️ 沒有 K/D/J 欄位，略過。")
        return
    dfp = df.dropna(subset=need)
    if dfp.empty:
        print("⚠️ KDJ 資料不足。")
        return

    plt.figure(figsize=(14, 4))
    plt.plot(dfp.index, dfp["K"], label="%K")
    plt.plot(dfp.index, dfp["D"], label="%D")
    plt.plot(dfp.index, dfp["J"], label="%J")
    plt.axhline(80, ls="--")
    plt.axhline(20, ls="--")
    plt.title("KDJ")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    print_versions()

    # 1) 驗證日期參數
    if args.start:
        try:
            datetime.strptime(args.start, "%Y-%m-%d")
            if args.end:
                datetime.strptime(args.end, "%Y-%m-%d")
        except ValueError:
            print("❌ 日期格式錯誤，需 YYYY-MM-DD")
            sys.exit(2)

    # 2) 抓資料（yahoo → stooq 後備）
    try:
        df = fetch_yahoo(args.ticker, args.period, args.interval, args.start, args.end)
    except Exception as e:
        print("Yahoo 例外：", repr(e))
        df = pd.DataFrame()

    if df is None or df.empty:
        try:
            df = fetch_stooq(args.ticker)
        except Exception as e:
            print("Stooq 也失敗：", repr(e))
            sys.exit(1)

    if df is None or df.empty:
        print("⚠️ 仍無資料")
        sys.exit(0)

    # 3) 欄位正規化 + 選價位欄
    df = normalize_single_ticker(df, args.ticker)
    print(f"✅ rows={len(df)} | cols={list(df.columns)}")

    price_col = choose_price_column(df)
    price = series_from(df, price_col)
    if price.isna().all():
        print(f"⚠️ {price_col} 全為 NaN")
        sys.exit(0)

    # 4) 計算技術指標（一定要在回測前做）
    df = add_indicators(df, price_col)

    # 5) （可選）畫圖
    if not args.no_plot:
        plot_price_ma_bbands(df, price_col, f"{args.ticker}")
        plot_macd(df)
        plot_rsi(df)
        plot_kdj(df)

    # 6) 回測（需要 backtest_module）
    cfg = BacktestConfig(
        initial_cash=100_000,
        risk_pct=0.01,
        atr_stop_mult=3.0,
        atr_tp_mult=6.0,
        fee_bps=5.0,
        slippage_bps=5.0,
        allow_reentry_days=1,
        rsi_upper=75.0,
        rsi_lower=40.0,
    )
    result = run_backtest(df, price_col, cfg)
    trades = result["trades"]
    plot_trades_on_price(df, price_col, trades, title=args.ticker)
    plot_equity_and_drawdown(result["curve"], title=args.ticker)



    print("\n=== Backtest Stats ===")
    for k, v in result["stats"].items():
        print(f"{k:>20}: {v}")

    print("\n=== Trades (head) ===")
    trades = result["trades"]
    if trades.empty:
        print("（無交易）")
    else:
        print(trades.head())


if __name__ == "__main__":
    main()
