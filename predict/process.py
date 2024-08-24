import numpy as np
import pandas as pd
import tensorflow as tf


def cal_value(df: pd.DataFrame) -> pd.DataFrame:
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["MA200"] = df["close"].rolling(window=200).mean()
    df["volume_MA50"] = df["volume"].rolling(window=50).mean()
    df["volume_MA200"] = df["volume"].rolling(window=200).mean()
    df.dropna(axis=0, inplace=True, how="any")

    # 0 값을 가지는 행 제거
    df = df[(df["open"] > 0) & (df["close"] > 0) & (df["volume"] > 0)].copy()

    df["delta"] = df["close"] / df["open"]
    df["up_delta"] = df["high"] / df[["open", "close"]].max(axis=1)
    df["down_delta"] = df["low"] / df[["open", "close"]].min(axis=1)
    df["d50"] = df["close"] / df["MA50"]
    df["d200"] = df["close"] / df["MA200"]

    df["volume_delta"] = df["volume"] / df["volume"].shift(1)
    df["volume_d50"] = df["volume"] / df["volume_MA50"]
    df["volume_d200"] = df["volume"] / df["volume_MA200"]

    df.drop(["MA50", "MA200", "volume_MA50", "volume_MA200"], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True, how="any")
    df.reset_index(drop=True, inplace=True)

    return df


def create_x_data(future: pd.DataFrame, window_size=864):
    future_val = future[["up_delta", "delta", "down_delta", "volume_ratio"]]

    x_data = []
    for i in range(len(future_val) - 2 * window_size + 1):
        future_slice = future_val.iloc[i : i + window_size].values
        x_data.append(future_slice)
    return np.array(x_data)


def create_y_data(future: pd.DataFrame, window_size=864):
    future_val = future[["up_delta", "delta", "down_delta", "volume_ratio"]]

    y_data = []
    for i in range(window_size, len(future) - window_size + 1):
        day_num = int(window_size / 3)
        future_slice = future_val.iloc[i : i + day_num]
        delta_vector = future_slice["delta"].values
        y_data.append(delta_vector)
    return np.array(y_data)


# 메모리 효율화를 위한 x,y data
def generate_x_data_conv2d(
    future: pd.DataFrame, x_cols: list, x_days: int, y_days: int
):
    window_size = x_days * 24 * 12
    future_val = future[x_cols].values.astype(np.float32)

    for i in range(len(future_val) - window_size - y_days * 24 * 12 + 1):
        yield future_val[i : i + window_size]


def generate_y_data_conv2d(
    future: pd.DataFrame, y_cols: list, x_days: int, y_days: int
):
    window_size = x_days * 24 * 12  # x_days 만큼의 데이터 크기
    future_val = future[y_cols].values.astype(np.float32)

    for i in range(window_size, len(future_val) - y_days * 24 * 12 + 1):
        window = future_val[i : i + y_days * 24 * 12]

        open_first = window[0][0]  # 첫 번째 open 값
        close_last = window[-1][1]  # 마지막 close 값
        high_max = window[:, 2].max()  # high의 최댓값
        low_min = window[:, 3].min()  # low의 최솟값

        min_open_close = min(open_first, close_last)
        max_open_close = max(open_first, close_last)

        low_ratio = low_min / min_open_close
        close_open_ratio = close_last / open_first
        high_ratio = high_max / max_open_close

        yield [low_ratio, close_open_ratio, high_ratio]


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data_file_path,
        cluster_file_path,
        cluster_num,
        x_days,
        y_days,
        x_cols,
        y_cols,
        test_size,
        batch_size,
        indices_split_size,
        is_validation=False,
        validation_split=0.2,
    ):
        super().__init__()
        self.data_file_path = data_file_path
        self.cluster_file_path = cluster_file_path
        self.cluster_num = cluster_num
        self.x_days = x_days
        self.y_days = y_days
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.window_size = x_days * 24 * 12
        self.y_window_size = y_days * 24 * 12
        self.batch_size = batch_size
        self.test_size = test_size
        self.indices_split_size = indices_split_size  # 한 번에 처리할 인덱스 수
        self.is_validation = is_validation
        self.validation_split = validation_split

        # 모든 속성 초기화 이후 호출
        self.indices = self._get_indices()
        self.current_index_batch = 0  # 현재 처리 중인 인덱스 배치
        self.loaded_data = None  # 메모리에 로드된 데이터
        self._prepare_next_batch()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        if index >= len(self):  # 인덱스 범위를 초과하는 경우
            raise IndexError("Index out of range")

        # 현재 배치가 로드된 데이터 범위를 초과하는 경우
        if index * self.batch_size >= len(self.loaded_data[0]):
            self.current_index_batch += 1
            self._prepare_next_batch()

        # 현재 로드된 데이터에서 배치 추출
        start = (
            index % (len(self.loaded_data[0]) // self.batch_size)
        ) * self.batch_size
        x_batch = self.loaded_data[0][start : start + self.batch_size]
        y_batch = self.loaded_data[1][start : start + self.batch_size]

        x_batch = x_batch.reshape(
            (x_batch.shape[0], x_batch.shape[1], x_batch.shape[2], 1)
        )

        return np.array(x_batch), np.array(y_batch)

    def _get_indices(self):
        # 클러스터 파일을 읽어 클러스터링된 인덱스 반환
        cluster_data = pd.read_csv(self.cluster_file_path)[["Cluster"]]
        cluster_array = cluster_data.values
        indices = np.where(cluster_array[:, 0] == self.cluster_num)[0][
            : -self.test_size
        ]
        # train, validation 분리
        total_len = len(indices)
        split_idx = int(total_len * (1 - self.validation_split))
        train_indices = indices[:split_idx]  # 처음 80%
        val_indices = indices[split_idx:]  # 마지막 20%

        if self.is_validation:
            return val_indices

        return train_indices

    def _prepare_next_batch(self):
        # 인덱스 배열을 나누어 처리
        start_idx = self.current_index_batch * self.indices_split_size
        end_idx = min(
            (self.current_index_batch + 1) * self.indices_split_size, len(self.indices)
        )
        divided_indices = self.indices[start_idx:end_idx]

        if len(divided_indices) == 0:
            return  # 더 이상 처리할 데이터가 없는 경우

        # 필요한 원본 데이터의 범위 계산
        start_data_idx = divided_indices[0]
        end_data_idx = divided_indices[-1] + self.window_size + self.y_window_size

        # 원본 데이터에서 해당 범위만큼 로드
        data_chunk = pd.read_csv(
            self.data_file_path,
            usecols=self.x_cols + self.y_cols,
            skiprows=list(range(1, int(start_data_idx) + 1)),
            nrows=int(end_data_idx - start_data_idx),
            header=0,
        )

        # x_data와 y_data 생성
        x_data = []
        y_data = []
        for idx in divided_indices:
            local_idx = idx - start_data_idx
            # x_data 생성
            x_val = data_chunk[self.x_cols].values
            x_chunk = x_val[local_idx : local_idx + self.window_size]
            x_data.append(x_chunk)

            # y_data 생성
            y_val = data_chunk[self.y_cols].values
            y_chunk = y_val[
                local_idx
                + self.window_size : local_idx
                + self.window_size
                + self.y_window_size
            ]
            open_first = y_chunk[0][0]  # 첫 번째 open 값
            close_last = y_chunk[-1][1]  # 마지막 close 값
            high_max = y_chunk[:, 2].max()  # high의 최댓값
            low_min = y_chunk[:, 3].min()  # low의 최솟값

            min_open_close = min(open_first, close_last)
            max_open_close = max(open_first, close_last)

            low_ratio = low_min / min_open_close
            close_open_ratio = close_last / open_first
            high_ratio = high_max / max_open_close
            # y_data.append([low_ratio, close_open_ratio, high_ratio])
            y_data.append([close_open_ratio])

        self.loaded_data = (np.array(x_data), np.array(y_data))


def make_test_data(
    data_file_path,
    cluster_file_path,
    cluster_num,
    x_days,
    y_days,
    x_cols,
    y_cols,
    test_size,
):
    window_size = x_days * 24 * 12
    y_window_size = y_days * 24 * 12

    # 클러스터 파일을 읽어 클러스터링된 인덱스 반환
    cluster_data = pd.read_csv(cluster_file_path)[["Cluster"]]
    cluster_array = cluster_data.values
    test_indices = np.where(cluster_array[:, 0] == cluster_num)[0][-test_size:]

    # 필요한 원본 데이터의 범위 계산
    start_data_idx = test_indices[0]
    end_data_idx = test_indices[-1] + window_size + y_window_size

    # 원본 데이터에서 해당 범위만큼 로드
    data_chunk = pd.read_csv(
        data_file_path,
        usecols=x_cols + y_cols,
        skiprows=list(range(1, int(start_data_idx) + 1)),
        nrows=int(end_data_idx - start_data_idx),
        header=0,
    )

    # x_data와 y_data 생성
    x_data = []
    y_data = []
    for idx in test_indices:
        local_idx = idx - start_data_idx
        # x_data 생성
        x_val = data_chunk[x_cols].values
        x_chunk = x_val[local_idx : local_idx + window_size]
        x_data.append(x_chunk)

        # y_data 생성
        y_val = data_chunk[y_cols].values
        y_chunk = y_val[
            local_idx + window_size : local_idx + window_size + y_window_size
        ]
        open_first = y_chunk[0][0]  # 첫 번째 open 값
        close_last = y_chunk[-1][1]  # 마지막 close 값
        high_max = y_chunk[:, 2].max()  # high의 최댓값
        low_min = y_chunk[:, 3].min()  # low의 최솟값

        min_open_close = min(open_first, close_last)
        max_open_close = max(open_first, close_last)

        low_ratio = low_min / min_open_close
        close_open_ratio = close_last / open_first
        high_ratio = high_max / max_open_close
        # y_data.append([low_ratio, close_open_ratio, high_ratio])
        y_data.append([close_open_ratio])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))

    return x_data, y_data
