import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from keras import callbacks

from data_fetch import fetch_data


def cal_log_value(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA10"] = df["close"].ewm(alpha=(2 / 11), adjust=False).mean()
    df["EMA20"] = df["close"].ewm(alpha=(2 / 21), adjust=False).mean()
    df["EMA50"] = df["close"].ewm(alpha=(2 / 51), adjust=False).mean()
    df["EMA200"] = df["close"].ewm(alpha=(2 / 201), adjust=False).mean()
    df["volume_MA50"] = df["volume"].rolling(window=50).mean()
    df["volume_ratio"] = df["volume"] / df["volume"].shift(1)

    # 하이킨아시
    df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    df["ha_open"] = 0.0
    df.at[0, "ha_open"] = df.iloc[0]["open"]
    df["ha_high"] = 0.0
    df["ha_low"] = 0.0
    for i in range(1, len(df)):
        df.at[i, "ha_open"] = (df.at[i - 1, "ha_open"] + df.at[i - 1, "ha_close"]) / 2
        df.at[i, "ha_high"] = max(
            df.at[i, "high"], df.at[i, "ha_open"], df.at[i, "ha_close"]
        )
        df.at[i, "ha_low"] = min(
            df.at[i, "low"], df.at[i, "ha_open"], df.at[i, "ha_close"]
        )
    df.at[0, "ha_high"] = df.at[0, "high"]
    df.at[0, "ha_low"] = df.at[0, "low"]

    # null값 행 및 0값 행 제거
    df.dropna(axis=0, inplace=True, how="any")
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["volume"] > 0)].copy()

    # 필요한 값 계산
    df["delta"] = np.log(df["close"] / df["open"])
    df["up_delta"] = np.log(df["high"] / df[["open", "close"]].max(axis=1))
    df["down_delta"] = np.log(df["low"] / df[["open", "close"]].min(axis=1))
    df["d10"] = np.log(df["close"] / df["EMA10"])
    df["d20"] = np.log(df["close"] / df["EMA20"])
    df["d50"] = np.log(df["close"] / df["EMA50"])
    df["d200"] = np.log(df["close"] / df["EMA200"])
    df["volume_delta"] = df["volume"] / df["volume_MA50"]
    df["ha_delta"] = np.log(df["ha_close"] / df["ha_open"])
    df["ha_up_delta"] = np.log(df["ha_high"] / df[["ha_open", "ha_close"]].max(axis=1))
    df["ha_down_delta"] = np.log(df["ha_low"] / df[["ha_open", "ha_close"]].min(axis=1))

    df["t_long"] = (
        (df["delta"] > 0)
        & (df["delta"].shift(1) > 0)
        & (df["delta"].abs() > df["delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)
    df["r_long"] = (
        (df["delta"] > 0)
        & (df["delta"].shift(1) < 0)
        & (df["delta"].abs() > df["delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)
    df["t_short"] = (
        (df["delta"] < 0)
        & (df["delta"].shift(1) < 0)
        & (df["delta"].abs() > df["delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)
    df["r_short"] = (
        (df["delta"] < 0)
        & (df["delta"].shift(1) > 0)
        & (df["delta"].abs() > df["delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)

    df["ha_t_long"] = (
        (df["ha_delta"] > 0)
        & (df["ha_delta"].shift(1) > 0)
        & (df["ha_delta"].abs() > df["ha_delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)
    df["ha_r_long"] = (
        (df["ha_delta"] > 0)
        & (df["ha_delta"].shift(1) < 0)
        & (df["ha_delta"].abs() > df["ha_delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)
    df["ha_t_short"] = (
        (df["ha_delta"] < 0)
        & (df["ha_delta"].shift(1) < 0)
        & (df["ha_delta"].abs() > df["ha_delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)
    df["ha_r_short"] = (
        (df["ha_delta"] < 0)
        & (df["ha_delta"].shift(1) > 0)
        & (df["ha_delta"].abs() > df["ha_delta"].shift(1).abs())
        & (df["volume_ratio"] > 1)
    ).astype(int)

    df.drop(
        [
            "open_time",
            "volume",
            "close_time",
            "EMA10",
            "EMA20",
            "EMA50",
            "EMA200",
            "volume_MA50",
        ],
        axis=1,
        inplace=True,
    )
    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def make_data(df: pd.DataFrame, columns: list):

    x_data = []
    y_data = []
    selected_df = df[columns]
    for i in range(df.shape[0] - 7):
        open = df.iloc[i + 1]["open"]
        close = df.iloc[i + 7]["close"]
        if close - open > 0:
            y_data.append(1)
        else:
            y_data.append(0)

        x = selected_df.iloc[i].values
        x_data.append(x)

    real_x_data = []
    real_x = selected_df.iloc[-1].values
    real_x_data.append(real_x)

    return np.array(x_data), np.array(y_data), np.array(real_x_data)


columns = [
    "delta",
    "up_delta",
    "down_delta",
    "d10",
    "d20",
    "d50",
    "d200",
    "ha_delta",
    "ha_up_delta",
    "ha_down_delta",
    "volume_delta",
    "volume_ratio",
    "t_long",
    "r_long",
    "t_short",
    "r_short",
    "ha_t_long",
    "ha_t_short",
    "ha_r_long",
    "ha_r_short",
]
df = fetch_data("BTCUSDT", "4h", 400, "spot")
df = cal_log_value(df)
X_data, y_data, real_x_data = make_data(df, columns)
print(X_data.shape)

X_train = X_data[:-70]
y_train = y_data[:-70]
X_test = X_data[-70:]
y_test = y_data[-70:]


model = keras.Sequential()
model.add(Input(shape=(len(columns),), name="input"))

# 첫 번째 은닉층: 입력 차원은 20, 활성화 함수는 ReLU
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.1))  # 드롭아웃을 추가하여 과적합 방지

# 두 번째 은닉층: 32개의 뉴런, ReLU 활성화 함수
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.1))

# 출력층: 이진 분류를 위한 시그모이드 활성화 함수
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일: 손실 함수는 이진 크로스엔트로피, 옵티마이저는 Adam
model.compile(optimizer="adamw", loss="binary_crossentropy", metrics=["accuracy"])

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=100,
    verbose=1,
    validation_split=0.1,
    callbacks=[early_stopping],
)

train_result = model.evaluate(X_test, y_test)
print(f"Train Results: {train_result}")

y_pred = model.predict(X_test)
real_y_pred = model.predict(real_x_data)
print("Original Data:", y_test)
print("Predicted Data:", y_pred)
print("Real Predicted Data:", real_y_pred)
