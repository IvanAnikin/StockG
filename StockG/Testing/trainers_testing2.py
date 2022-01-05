
from StockG.Agents.Trader import technical

best, log_string, Return, signals_img, portfolio_img = technical.get_signals_loop()

print(best)
