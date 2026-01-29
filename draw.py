import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ensure_datetime_cols(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    if not pd.api.types.is_datetime64_any_dtype(t["entry_date"]):
        t["entry_date"] = pd.to_datetime(t["entry_date"])
    if not pd.api.types.is_datetime64_any_dtype(t["exit_date"]):
        t["exit_date"] = pd.to_datetime(t["exit_date"])
    return t

def plot_trades_on_price(df: pd.DataFrame, price_col: str, trades: pd.DataFrame, title: str = ""):
    """
    在收盤價走勢上標出每筆交易的進/出場點，並以線連起來。
    - df：含價格與指標的 DataFrame（index 為日期）
    - price_col：'Adj Close' 或 'Close'
    - trades：run_backtest 回傳的 result['trades']
    """
    t = ensure_datetime_cols(trades)
    if t.empty:
        print("沒有交易紀錄可視化。")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df[price_col], label=price_col)

    # 進出場點與連線
    for _, r in t.iterrows():
        # 進場點（上三角）
        plt.scatter(r["entry_date"], r["entry_px"], marker="^", s=70, label=None)
        # 出場點（下三角）
        plt.scatter(r["exit_date"], r["exit_px"], marker="v", s=70, label=None)
        # 連線（從進場價到出場價）
        plt.plot([r["entry_date"], r["exit_date"]], [r["entry_px"], r["exit_px"]])

        # 註記：理由 + 損益（可視需要註解）
        ann = f'{r["reason"]}, PnL={r["pnl"]:.0f}'
        plt.annotate(ann, xy=(r["exit_date"], r["exit_px"]), xytext=(5, 5),
                     textcoords="offset points")

    plt.title(f"{title} 交易可視化（疊加於{price_col}）")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_equity_and_drawdown(curve_df: pd.DataFrame, title: str = "Equity Curve"):
    """
    curve_df：result['curve']，index=date, col='equity'
    會畫兩張圖：資金曲線 / 最大回撤
    """
    if curve_df is None or curve_df.empty:
        print("沒有資金曲線可視化。")
        return

    eq = curve_df["equity"].astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0  # drawdown (負值)

    # 資金曲線
    plt.figure(figsize=(14, 4))
    plt.plot(eq.index, eq.values, label="Equity")
    plt.title(f"{title} - 資金曲線")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 回撤曲線
    plt.figure(figsize=(14, 3))
    plt.plot(dd.index, dd.values, label="Drawdown")
    plt.title(f"{title} - 最大回撤曲線")
    plt.legend()
    plt.tight_layout()
    plt.show()
