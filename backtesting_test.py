from backtesting import Backtest, Strategy
from backtesting.test import SMA, GOOG  # 測試資料（Google 股價）

class SmaCross(Strategy):
    def init(self):
        # 定義指標（用內建 SMA）
        self.sma10 = self.I(SMA, self.data.Close, 10)
        self.sma20 = self.I(SMA, self.data.Close, 20)

    def next(self):
        # 如果 10MA 上穿 20MA，做多
        if self.sma10[-1] > self.sma20[-1] and self.sma10[-2] <= self.sma20[-2]:
            self.buy()

        # 如果 10MA 下穿 20MA，做空
        elif self.sma10[-1] < self.sma20[-1] and self.sma10[-2] >= self.sma20[-2]:
            self.sell()

# 建立回測
bt = Backtest(GOOG, SmaCross,
              cash=10_000,
              commission=.002,
              trade_on_close=False)

stats = bt.run()      # 執行策略
print(stats)          # 顯示績效
bt.plot()             # 畫圖
