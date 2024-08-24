import time
import pandas as pd
import datetime
from binance.spot import Spot
from binance.um_futures import UMFutures


# fetch_data 를 위한 함수, 정해진 개수의 데이터 가져옴
def fetch_one_data(
    symbol: str,
    interval: str,
    end_time: int,
    limit: int,
    type: str = "future",
) -> pd.DataFrame:

    client = UMFutures()
    if type == "spot":
        client = Spot()

    bars = client.klines(
        symbol=symbol, interval=interval, endTime=end_time, limit=limit
    )
    df = pd.DataFrame(
        bars,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df.drop(
        [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
        axis=1,
        inplace=True,
    )

    # 모든 열을 숫자로 변환
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# interval 에 따른 정해진 개수의 데이터 가져오기
def fetch_data(
    symbol: str, interval: str, numbers: int, type: str = "future"
) -> pd.DataFrame:

    end_datetime = datetime.datetime.now(datetime.UTC)

    if interval == "4h":
        now = datetime.datetime.now(datetime.UTC)
        now_hour = now.hour
        if 0 <= now_hour < 12:
            end_datetime = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            end_datetime = now.replace(hour=12, minute=0, second=0, microsecond=0)
    elif interval == "1h":
        now = datetime.datetime.now(datetime.UTC)
        end_datetime = now.replace(minute=0, second=0, microsecond=0)

    end_time = int(end_datetime.timestamp() * 1000 - 1)

    data = []

    cnt: int = 1500
    if type == "spot":
        cnt: int = 1000

    while numbers > 0:
        if numbers < cnt:
            num = numbers
        else:
            num = cnt

        df = fetch_one_data(
            symbol=symbol,
            interval=interval,
            end_time=end_time,
            limit=num,
            type=type,
        )
        if df.empty:
            break

        data.insert(0, df)

        end_time = int(df.iloc[0]["open_time"]) - 1
        numbers -= num

    data_combined = pd.concat(data, axis=0, ignore_index=True)
    data_combined.dropna(axis=0, inplace=True)
    data_combined.reset_index(drop=True, inplace=True)

    return data_combined


# 시작 시간과 마지막 시간 사이의 데이터를 가져오는 함수
def fetch_interval_data(
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int,
    type: str = "future",
) -> pd.DataFrame:

    client = UMFutures()
    cnt: int = 1500
    if type == "spot":
        client = Spot()
        cnt: int = 1000

    bars = client.klines(
        symbol=symbol,
        interval=interval,
        startTime=start_time,
        endTime=end_time,
        limit=cnt,
    )
    df = pd.DataFrame(
        bars,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df.drop(
        [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
        axis=1,
        inplace=True,
    )

    # 모든 열을 숫자로 변환
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# start time 기준 fetch one data
def fetch_one_data_start(
    symbol: str,
    interval: str,
    start_time: int,
    limit: int,
    type: str = "future",
) -> pd.DataFrame:

    client = UMFutures()
    if type == "spot":
        client = Spot()

    bars = client.klines(
        symbol=symbol, interval=interval, startTime=start_time, limit=limit
    )
    df = pd.DataFrame(
        bars,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df.drop(
        [
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
        axis=1,
        inplace=True,
    )

    # 모든 열을 숫자로 변환
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


# start time 기준 fetch data
def fetch_data_start(
    symbol: str, interval: str, start_time: int, numbers: int, type: str = "future"
) -> pd.DataFrame:

    data = []
    st_time = start_time

    cnt: int = 1500
    if type == "spot":
        cnt: int = 1000

    while numbers > 0:
        if numbers < cnt:
            num = numbers
        else:
            num = cnt

        df = fetch_one_data_start(
            symbol=symbol,
            interval=interval,
            start_time=st_time,
            limit=num,
            type=type,
        )
        if df.empty:
            break

        data.append(df)
        st_time = int(df.iloc[-1]["close_time"] + 1)
        numbers -= num

    data_combined = pd.concat(data, axis=0, ignore_index=True)
    data_combined.dropna(axis=0, inplace=True)
    data_combined.reset_index(drop=True, inplace=True)

    return data_combined
