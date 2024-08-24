import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow import keras

from process import generate_y_data_conv2d, generate_x_data_conv2d


def make_result(y_data: np.array):
    results = []
    for window in y_data:
        prices = [(0, 10000)]
        ups = []
        downs = []

        for i in range(len(window)):
            test = window[i]
            open = prices[i - 1][1]
            close = prices[i - 1][1] * test[1]
            prices.append((open, close))
            downs.append(min(open, close) * test[0])
            ups.append(max(open, close) * test[2])

        results.append((min(downs), prices[-1][1], max(ups)))

    return results


def plot_result(test_results: list, pred_results: list):
    if len(test_results) != len(pred_results) or len(test_results[0]) != len(
        pred_results[0]
    ):
        raise ValueError("Test results and predicted results are not equal")

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    for i in range(len(test_results[0])):
        x_values = list(range(len(test_results)))
        test_y_values = [x[i] for x in test_results]
        pred_y_values = [y[i] for y in pred_results]

        axs[i].plot(
            x_values, test_y_values, marker="o", linestyle="-", label=f"test{i + 1}"
        )
        axs[i].plot(
            x_values, pred_y_values, marker="o", linestyle="-", label=f"pred{i + 1}"
        )
        axs[i].set_title(f"Graph {i + 1}: test{i + 1} vs pred{i + 1}")
        axs[i].set_xlabel("Index")
        axs[i].set_ylabel(f"Value {i + 1}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()


# 프로젝트 설정
# drive_dir = "drive/My Drive/Colab Notebooks/"
model_dir = "model/conv2d_0.keras"
data_dir = "data/conv2d.csv"
cluster_dir = "data/clustered_data_7d.csv"


# 변수 설정
test_cnt: int = 120
x_days: int = 7
x_cols: list = ["volume_ratio", "down_delta", "delta", "up_delta"]
y_cols: list = ["down_delta", "delta", "up_delta"]
cluster_num: int = 0

# 데이터 가져오기
data = pd.read_csv(data_dir)

# 데이터 전처리 및 생성
x_gen = generate_x_data_conv2d(data, x_cols, x_days, 1)
y_gen = generate_y_data_conv2d(data, y_cols, x_days, 1)
x_data = np.array(list(x_gen), dtype=np.float32)
y_data = np.array(list(y_gen), dtype=np.float32)
if len(x_data) != len(y_data):
    raise Exception("Data size mismatch")

# 분류 결과 가져오기
cluster = pd.read_csv(cluster_dir)

# 분류
cluster_lst_x = []
cluster_lst_y = []
cluster = cluster[-len(x_data) :]
for i in range(len(x_data)):
    if cluster["Cluster"].iloc[i] == cluster_num:
        cluster_lst_x.append(x_data[i])
        cluster_lst_y.append(y_data[i])
x_data = np.array(cluster_lst_x)
y_data = np.array(cluster_lst_y)

# 데이터를 Conv2D 입력에 맞게 4차원으로 변환
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
y_data = y_data.reshape((y_data.shape[0], y_data.shape[1], y_data.shape[2], 1))

# 학습 데이터, 테스트 데이터 분리
x_data_learn = x_data[:-test_cnt]
y_data_learn = y_data[:-test_cnt]
x_data_test = x_data[-test_cnt:]
y_data_test = y_data[-test_cnt:]

model = keras.saving.load_model(model_dir)
model.summary()

# 예측(테스트 데이터)
y_result = model.predict(x_data_test)
y_test = y_data_test.reshape(
    (y_data_test.shape[0], y_data_test.shape[1], y_data_test.shape[2])
)
y_pred = y_result.reshape(
    (y_data_test.shape[0], y_data_test.shape[1], y_data_test.shape[2])
)

# 예측 결과 출력
test_results: list = make_result(y_test)
pred_results: list = make_result(y_pred)
plot_result(test_results, pred_results)
test_means = np.mean(test_results, axis=0)
pred_means = np.mean(pred_results, axis=0)
print(f"Test Means:{test_means}")
print(f"Test Last:{test_results[-1]}")
print(f"Pred Means:{pred_means}")
print(f"Test Results:{test_results}")
print(f"Pred Results:{pred_results}")
