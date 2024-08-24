import numpy as np
from keras import callbacks

from models import Conv2DModel
from process import DataGenerator, make_test_data
from model_test import plot_result


# 프로젝트 설정
# drive_dir = "drive/My Drive/Colab Notebooks/"
project_name = "conv2d_0_20d"
model_dir = "model/conv2d_0_20d.keras"
data_dir = "data/conv2d.csv"
cluster_dir = "data/cluster_20d.csv"

# 변수 설정
cluster_num: int = 0
test_size: int = 120
epochs: int = 100
batch_size: int = 32
indices_split_size: int = 5000
x_days: int = 20
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

# 데이터 생성
train_generator = DataGenerator(
    data_file_path=data_dir,
    cluster_file_path=cluster_dir,
    cluster_num=cluster_num,
    x_days=x_days,
    y_days=1,
    x_cols=x_cols,
    y_cols=y_cols,
    test_size=test_size,
    batch_size=batch_size,
    indices_split_size=indices_split_size,
)
val_generator = DataGenerator(
    data_file_path=data_dir,
    cluster_file_path=cluster_dir,
    cluster_num=cluster_num,
    x_days=x_days,
    y_days=1,
    x_cols=x_cols,
    y_cols=y_cols,
    test_size=test_size,
    batch_size=batch_size,
    indices_split_size=indices_split_size,
    is_validation=True,
)

x_test, y_test = make_test_data(
    data_file_path=data_dir,
    cluster_file_path=cluster_dir,
    cluster_num=cluster_num,
    x_days=x_days,
    y_days=1,
    x_cols=x_cols,
    y_cols=y_cols,
    test_size=test_size,
)

model = Conv2DModel(
    x_days=x_days,
    x_cols=x_cols,
).build()

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)  # 조기 종료 콜백 설정

model.fit(
    train_generator,
    epochs=epochs,
    callbacks=[early_stopping],
    validation_data=val_generator,
)

# 모델 평가
train_result = model.evaluate(train_generator, return_dict=True)
val_result = model.evaluate(val_generator, return_dict=True)
print(f"Train Results: {train_result}")
print(f"Val Results: {val_result}")
model.summary()  # 파라미터 수 및 깊이 수 확인
model.save(model_dir)

# 예측(테스트 데이터)
pred_results = model.predict(x_test)
test_results = y_test.copy()

# 예측 결과 출력
plot_result(test_results, pred_results)
test_means = np.mean(test_results, axis=0)
pred_means = np.mean(pred_results, axis=0)
print(f"Test Means:{test_means}")
print(f"Test Last:{test_results[-1]}")
print(f"Pred Means:{pred_means}")
print(f"Pred Last:{pred_results[-1]}")
