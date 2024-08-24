import numpy as np
import pandas as pd
from keras import callbacks
from keras_tuner import RandomSearch
import keras_tuner as kt

import models
from model_test import plot_result
from process import generate_x_data_conv2d, generate_y_data_conv2d


# 프로젝트 설정
# drive_dir = "drive/My Drive/Colab Notebooks/"
project_name = "conv2d_0_search"
model_dir = "model/conv2d_0_search.keras"
data_dir = "data/conv2d.csv"
cluster_dir = "data/clustered_data_7d.csv"
hyperparam_dir = "hyperparam"

# 변수 설정
test_cnt: int = 120
epochs: int = 100
x_days: int = 7
x_cols: list = [
    "volume_d200",
    "volume_d50",
    "volume_delta",
    "d200",
    "d50",
    "down_delta",
    "delta",
    "up_delta",
]
y_cols: list = ["open", "close", "high", "low"]
activation: str = "relu"
cluster_num: int = 0

# 데이터 가져오기
data = pd.read_csv(data_dir, usecols=x_cols + y_cols, dtype=np.float32)

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
cluster = cluster[-len(x_data) :]
# 클러스터 번호에 해당하는 인덱스만 추출
indices = cluster.index[cluster["Cluster"] == cluster_num].tolist()
# 해당 인덱스를 사용하여 x_data와 y_data 필터링
x_data = x_data[indices]
y_data = y_data[indices]

# 데이터를 Conv2D 입력에 맞게 4차원으로 변환
x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
x_shape_input = (x_data.shape[1], x_data.shape[2], 1)
y_shape_input = (None, y_data.shape[1])

# 학습 데이터, 테스트 데이터 분리
x_data_learn = x_data[:-test_cnt]
y_data_learn = y_data[:-test_cnt]
x_data_test = x_data[-test_cnt:]
y_data_test = y_data[-test_cnt:]

print(f"Shape: {x_data_learn.shape},{y_data_learn.shape}")
print(f"Test Shape: {x_data_test.shape},{y_data_test.shape}")

# RandomSearch 객체 생성
hypermodel = models.Conv2DHyperModel(
    x_shape_input=x_shape_input,
    y_shape_input=y_shape_input,
    name=project_name,
    activation=activation,
)

# 하이퍼파라미터 서치 객체(튜너) 생성
tuner = RandomSearch(
    hypermodel,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=1,
    directory=hyperparam_dir,
    project_name=project_name,
)

# 조기 종료 콜백 설정
early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# 하이퍼파라미터 튜닝 수행
hp = kt.HyperParameters()
tuner.search(
    x_data_learn,
    y_data_learn,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping],
)

# 최적 하이퍼파라미터
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Conv1 Filter: {best_hps.get('conv1_filters')}")
for i in range(best_hps.get("num_conv_layers")):
    print(f"Conv{i + 2} Filter: {best_hps.get('one_conv' + str(i + 2) + '_filters')}")
    print(f"Conv{i + 2} Filter: {best_hps.get('conv' + str(i + 2) + '_filters')}")


# 최적의 하이퍼파라미터로 모델 학습
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    x_data_learn,
    y_data_learn,
    epochs=epochs,
    callbacks=[early_stopping],
    validation_split=0.2,
)  # validation_split 적용 여부 고려


# 모델 평가(학습 데이터)
result = model.evaluate(x_data_learn, y_data_learn, return_dict=True)
print(f"Results: {result}")
model.summary()  # 파라미터 수 및 깊이 수 확인
model.save(model_dir)


# 예측(테스트 데이터)
pred_results = model.predict(x_data_test)
test_results = y_data_test.copy()

# 예측 결과 출력
plot_result(test_results, pred_results)
test_means = np.mean(test_results, axis=0)
pred_means = np.mean(pred_results, axis=0)
print(f"Test Means:{test_means}")
print(f"Test Last:{test_results[-1]}")
print(f"Pred Means:{pred_means}")
