import time, datetime
import pandas as pd

import config
from data_fetch import fetch_interval_data, fetch_data_start
from process import cal_value

symbol = config.symbol

"""
처음 스타트
"""

# 첫 시작,2019-12-25 00:00 - 2019-12-25 12:00
start_time = 1577199600000
end_time = 1577242800000
df_start = fetch_interval_data(
    symbol=symbol,
    interval="5m",
    start_time=start_time,
    end_time=end_time,
    type="future",
)
df_start.to_csv("data/origin.csv", index=False)


"""
데이터 추가 작업
"""

num = 30000
while num == 30000:
    before_df = pd.read_csv("data/origin.csv")
    start_time = int(before_df.iloc[-1]["close_time"]) + 1
    plus_term = int(datetime.timedelta(minutes=5).total_seconds() * num * 1000)
    end_datetime = datetime.datetime.now(datetime.UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - datetime.timedelta(days=1)

    if start_time + plus_term >= int(end_datetime.timestamp() * 1000):
        num = (start_time + plus_term - int(end_datetime.timestamp() * 1000)) / int(
            datetime.timedelta(minutes=5).total_seconds() * 1000
        )
        print(num)

    after_df = fetch_data_start(
        symbol=symbol,
        interval="5m",
        start_time=start_time,
        numbers=num,
        type="future",
    )
    after_df.to_csv("data/origin.csv", index=False, header=False, mode="a")

    print(f"Data Shape:{after_df.shape}")
    end = after_df.iloc[-1]["close_time"]
    end_date = datetime.datetime.fromtimestamp(int(end / 1000))
    print(f"Time:{end_date}")

    time.sleep(61)


"""
데이터 처리 후 저장
"""

df = pd.read_csv("data/origin.csv")
df = cal_value(df)
df.to_csv("data/conv2d.csv", index=False)
