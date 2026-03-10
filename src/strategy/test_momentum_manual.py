import pandas as pd

from src.data.features import add_basic_features
from src.data.loader import load_market_data
from src.engine.portfolio import Portfolio
from src.strategy.momentum import MomentumStrategy


df = load_market_data()
features_df = add_basic_features(df)

portfolio = Portfolio(initial_cash=50.0)
strategy = MomentumStrategy(
    max_open_positions=2,
    top_k=2,
    min_score=0.0,
)

test_date = pd.Timestamp("2025-06-02")
signals = strategy.generate_signals(
    decision_date=test_date,
    market_data=features_df,
    portfolio=portfolio,
)

print(f"Signals for {test_date.date()}:")
for signal in signals:
    print(signal)