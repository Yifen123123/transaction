# backtest_module.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from dataclasses import dataclass

# ====== 指標：ATR ======
def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    計算 Average True Range (ATR)
    需要欄位：High / Low / Close
    """
    use = df.copy()
    high = use["High"].astype(float)
    low  = use["Low"].astype(float)
    prev_close = use["Close"].shift(1).astype(float)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    use["ATR"] = tr.rolling(window=window, min_periods=window).mean()
    return use

# ====== 參數設定 ======
@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0      # 初始資金
    risk_pct: float = 0.01               # 單筆交易最大風險（資金比例）
    atr_stop_mult: float = 3.0           # 停損: N * ATR
    atr_tp_mult: float = 6.0             # 停利: N * ATR
    fee_bps: float = 5.0                 # 單邊手續費(bps) 5 bps = 0.05%
    slippage_bps: float = 5.0            # 單邊滑價(bps)
    allow_reentry_days: int = 1          # 出場後冷卻天數
    rsi_upper: float = 75.0              # 訊號參數（可與主程式一致）
    rsi_lower: float = 40.0

# ====== 訊號生成（可換成你自己的規則）======
def generate_signals(df: pd.DataFrame, cfg: BacktestConfig | None = None) -> pd.DataFrame:
    """
    進場：收盤 > MA20 且 MACD > Signal 且 rsi_lower <= RSI < 70
    出場：收盤 < MA20 或 RSI > rsi_upper
    所有交易用「隔日開盤」執行，避免前視。
    """
    if cfg is None:
        cfg = BacktestConfig()

    sig = df.copy()
    sig["long_entry_cond"] = (
        (sig["Close"] > sig["MA20"])
        & (sig["MACD"] > sig["Signal"])
        & (sig["RSI"] >= cfg.rsi_lower)
        & (sig["RSI"] < 70)
    )
    sig["long_exit_cond"] = (sig["Close"] < sig["MA20"]) | (sig["RSI"] > cfg.rsi_upper)

    # 隔日開盤價執行
    sig["entry_price"] = sig["Open"].shift(-1)
    sig["exit_price_sig"] = sig["Open"].shift(-1)
    return sig

# ====== 回測主流程（單標的、僅做多、隔日開盤進出）======
def run_backtest(df: pd.DataFrame, price_col: str, cfg: BacktestConfig = BacktestConfig()) -> dict:
    """
    需求欄位：
      必要：Open/High/Low/Close、MA20、MACD、Signal、RSI
      本函式內會補 ATR(14)
    策略：
      - 進場：generate_signals() 規則；隔日開盤買
      - 出場：達停損/停利優先，否則符合 exit 訊號；隔日開盤賣
    交易成本：
      - 單邊手續費與滑價用 bps 設定
    回傳：
      - curve: 資金曲線 DataFrame(index=date, equity)
      - trades: 交易明細 DataFrame
      - stats: 總結指標 dict
    """
    use = df.copy()
    # 檢查必要欄位
    need_cols = ["Open", "High", "Low", "Close", "MA20", "MACD", "Signal", "RSI"]
    missing = [c for c in need_cols if c not in use.columns]
    if missing:
        raise ValueError(f"回測缺少必要欄位: {missing}")

    if "ATR" not in use.columns:
        use = add_atr(use, window=14)

    use = generate_signals(use, cfg)
    use = use.dropna(subset=["ATR", "entry_price", "exit_price_sig"])

    # 初始狀態
    cash = cfg.initial_cash
    position = 0
    entry_px = np.nan
    stop_px = np.nan
    tp_px = np.nan
    entry_date = None
    last_exit_idx = use.index[0] - pd.Timedelta(days=cfg.allow_reentry_days+1)

    curve = []     # (date, equity)
    trades = []    # dict 列表

    fee_rate = cfg.fee_bps / 10_000.0
    slip_rate = cfg.slippage_bps / 10_000.0

    # 逐日模擬（最後一天無法隔日成交）
    for i in range(len(use) - 1):
        row = use.iloc[i]
        idx = use.index[i]
        next_open = use["Open"].iloc[i + 1]
        next_high = use["High"].iloc[i + 1]
        next_low  = use["Low"].iloc[i + 1]

        # 有部位 → 先檢查隔日盤中 停損/停利，再看訊號出場
        if position > 0:
            exit_flag = False
            exit_reason = ""
            exit_px = None

            # 以次日 high/low 測是否觸及停損/停利
            if next_low <= stop_px:         # 停損
                exit_px = stop_px * (1 - fee_rate - slip_rate)
                exit_flag = True
                exit_reason = "stop"
            elif next_high >= tp_px:        # 停利
                exit_px = tp_px * (1 - fee_rate - slip_rate)
                exit_flag = True
                exit_reason = "takeprofit"
            elif row["long_exit_cond"]:     # 訊號出場（隔日開盤）
                exit_px = row["exit_price_sig"] * (1 - fee_rate - slip_rate)
                exit_flag = True
                exit_reason = "signal"

            if exit_flag:
                proceeds = position * exit_px
                fee = proceeds * fee_rate
                cash += proceeds - fee
                pnl = (exit_px - entry_px) * position - fee
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": use.index[i + 1],
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "shares": position,
                    "pnl": pnl,
                    "reason": exit_reason,
                })
                # 平倉
                position = 0
                entry_px = stop_px = tp_px = np.nan
                entry_date = None
                last_exit_idx = use.index[i + 1]

        # 無部位 → 檢查是否可進場（冷卻天數）
        if position == 0:
            cool_ok = (idx - last_exit_idx).days >= cfg.allow_reentry_days
            if row["long_entry_cond"] and cool_ok and not pd.isna(row["ATR"]):
                risk_amt = cash * cfg.risk_pct
                # 隔日開盤買
                buy_px = row["entry_price"] * (1 + fee_rate + slip_rate)
                stop_px = buy_px - cfg.atr_stop_mult * row["ATR"]
                tp_px   = buy_px + cfg.atr_tp_mult   * row["ATR"]
                risk_per_share = buy_px - stop_px
                shares = int(risk_amt // risk_per_share)
                if shares <= 0:
                    # 風險太小買不起一股（或最小單位），跳過
                    pass
                else:
                    cost = shares * buy_px
                    if cost > cash:
                        shares = int(cash // buy_px)
                        cost = shares * buy_px
                    if shares > 0:
                        fee = cost * fee_rate
                        cash -= (cost + fee)
                        position = shares
                        entry_px = buy_px
                        entry_date = use.index[i + 1]

        # 記錄資產（以當日收盤估值）
        mark_px = use["Close"].iloc[i]
        equity = cash + (position * mark_px if position > 0 else 0.0)
        curve.append((idx, equity))

    # 最後一天：若仍有部位，按最後收盤出清（簡化）
    if position > 0:
        last_idx = use.index[-1]
        last_px = use["Close"].iloc[-1] * (1 - fee_rate - slip_rate)
        proceeds = position * last_px
        fee = proceeds * fee_rate
        cash += proceeds - fee
        pnl = (last_px - entry_px) * position - fee
        trades.append({
            "entry_date": entry_date,
            "exit_date": last_idx,
            "entry_px": entry_px,
            "exit_px": last_px,
            "shares": position,
            "pnl": pnl,
            "reason": "EOD",
        })
        equity = cash
        curve.append((last_idx, equity))

    curve_df = pd.DataFrame(curve, columns=["date", "equity"]).set_index("date")
    stats = summarize_performance(curve_df, trades, cfg.initial_cash)
    trades_df = pd.DataFrame(trades)
    return {"curve": curve_df, "trades": trades_df, "stats": stats}

# ====== 績效彙總 ======
def summarize_performance(curve_df: pd.DataFrame, trades: list | pd.DataFrame, initial_cash: float) -> dict:
    eq = curve_df["equity"].astype(float)
    total_return = float(eq.iloc[-1] / initial_cash - 1.0)

    # 年化 CAGR
    if len(eq) > 1:
        days = (eq.index[-1] - eq.index[0]).days
        years = max(days / 365.25, 1e-9)
    else:
        years = 1e-9
    cagr = float((eq.iloc[-1] / initial_cash) ** (1 / years) - 1.0) if years > 0 else np.nan

    # 最大回撤
    peak = eq.cummax()
    drawdown = eq / peak - 1.0
    max_dd = float(drawdown.min())

    # Sharpe（以日報酬年化，無風險利率略）
    daily_ret = eq.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))
    else:
        sharpe = np.nan

    # 交易統計
    tr = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
    if tr.empty:
        n_trades = 0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = np.nan
        expectancy = 0.0
    else:
        n_trades = int(len(tr))
        wins = tr[tr["pnl"] > 0]["pnl"]
        losses = tr[tr["pnl"] <= 0]["pnl"]
        win_rate = float(len(wins) / n_trades) if n_trades > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(-losses.sum()) if len(losses) else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
        expectancy = float(win_rate * avg_win + (1 - win_rate) * avg_loss)

    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": total_return,
        "CAGR": cagr if np.isfinite(cagr) else np.nan,
        "max_drawdown": max_dd,
        "sharpe": sharpe if np.isfinite(sharpe) else np.nan,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
    }

# ====== 輸出工具（可選）======
def export_results(prefix: str, result: dict) -> None:
    """
    將回測結果輸出為 CSV 檔：{prefix}_curve.csv, {prefix}_trades.csv, {prefix}_stats.json
    """
    curve = result.get("curve")
    trades = result.get("trades")
    stats = result.get("stats")

    if isinstance(curve, pd.DataFrame) and not curve.empty:
        curve.to_csv(f"{prefix}_curve.csv")
    if isinstance(trades, pd.DataFrame):
        trades.to_csv(f"{prefix}_trades.csv", index=False)
    if isinstance(stats, dict):
        import json
        with open(f"{prefix}_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
