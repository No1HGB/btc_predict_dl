import datetime
import pandas as pd

import config
from data_fetch import fetch_interval_data
from process import cal_value

"""
# 매일 오전 9시 이후 처리
symbol = config.symbol
before_df = pd.read_csv("data/origin.csv")
start_time = int(before_df.iloc[-1]["close_time"]) + 1
end_time = int(
    datetime.datetime.now(datetime.UTC)
    .replace(hour=0, minute=0, second=0, microsecond=0)
    .timestamp()
    * 1000
    - 1
)
append_df = fetch_interval_data(symbol, "5m", start_time, end_time)
append_df.to_csv("data/origin.csv", index=False, header=False, mode="a")
"""
df = pd.read_csv("data/origin.csv")
df = cal_value(df)
df.to_csv("data/conv2d.csv", index=False)
