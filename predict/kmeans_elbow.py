import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

from process import generate_x_data_conv2d

# 디렉토리 설정
# drive_dir = "drive/My Drive/Colab Notebooks/"
data_dir = "data/conv2d.csv"

# 변수 설정
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
batch_size: int = 256 * 256

# 원본 데이터
df = pd.read_csv(data_dir, usecols=x_cols, dtype=np.float32)

# x_data 데이터 생성
gen = generate_x_data_conv2d(df, x_cols, x_days, 1)
data = np.array(list(gen), dtype=np.float32)
data_reshaped = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

# SSE 값을 저장할 리스트 초기화
sse = []

# 클러스터 개수 범위 설정 (예: 1부터 10까지)
k_range = range(1, 11)

# 각 클러스터 개수에 대해 미니배치 K-means 클러스터링 수행 및 SSE 계산
for k in k_range:
    minibatch_kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=batch_size,
        init="k-means++",
        n_init="auto",
    )
    minibatch_kmeans.fit(data_reshaped)
    sse.append(minibatch_kmeans.inertia_)  # SSE 값을 리스트에 추가

# 엘보우 방법 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal Number of Clusters using MiniBatch K-means")
plt.show()
