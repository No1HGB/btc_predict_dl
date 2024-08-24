import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from process import generate_x_data_conv2d

# 프로젝트 설정
# drive_dir = "drive/My Drive/Colab Notebooks/"
data_dir = "data/conv2d.csv"
data_reshaped_save_dir = "data/clustered_data_fit_7d.csv"
model_dir = "model/kmeans_model_fit_7d.pkl"

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
clusters: int = 8
plot_data_cnt: int = 12000

# 원본 데이터
df = pd.read_csv(data_dir, usecols=x_cols, dtype=np.float32)

# x_data 데이터 생성
gen = generate_x_data_conv2d(df, x_cols, x_days, 1)
data = np.array(list(gen), dtype=np.float32)
data_reshaped = data.reshape(data.shape[0], data.shape[1] * data.shape[2])

# K-Means 클러스터링 수행 (K=4)
kmeans = KMeans(
    n_clusters=clusters,
    random_state=42,
    init="k-means++",
    n_init="auto",
).fit(data_reshaped)

# 클러스터 할당 결과
labels = kmeans.labels_

# 클러스터 할당 결과를 csv 파일로 저장
data_reshaped_zero = np.zeros((data_reshaped.shape[0], 1))
df_reshaped = pd.DataFrame(data_reshaped_zero, columns=["Value"])
df_reshaped["Cluster"] = labels
df_reshaped.drop(columns=["Value"], inplace=True)
df_reshaped.to_csv(data_reshaped_save_dir, index=False)

# K-Means 모델 저장
joblib.dump(kmeans, model_dir)
print("Model saved successfully!")

# 마지막 n개의 데이터에 대한 클러스터 할당 결과
subset_labels = labels[-plot_data_cnt:]

# 마지막 n개의 데이터
subset_data = data_reshaped[-plot_data_cnt:]

# PCA를 사용하여 차원 축소 (4개의 주요 특성)
pca = PCA(n_components=4)
subset_data_pca = pca.fit_transform(subset_data)

# 데이터의 차원 수 (특성 수) - 4로 고정
num_features = subset_data_pca.shape[1]

# 특성 쌍 조합에 따른 2D 산점도 그리기
fig, axes = plt.subplots(nrows=num_features, ncols=num_features, figsize=(20, 20))
fig.suptitle("K-Means Clustering with Multiple 2D Scatter Plots")

# 각 클러스터에 대한 색상 설정
colors = [
    "#1f77b4",  # 파란색
    "#ff7f0e",  # 주황색
    "#2ca02c",  # 녹색
    "#d62728",  # 빨간색
    "#9467bd",  # 보라색
    "#8c564b",  # 갈색
    "#e377c2",  # 분홍색
    "#7f7f7f",  # 회색
]

for i in range(num_features):
    for j in range(num_features):
        ax = axes[i, j]
        if i == j:
            ax.text(
                0.5,
                0.5,
                f"Feature {i+1}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            for cluster_idx in range(clusters):
                cluster_points = subset_data_pca[subset_labels == cluster_idx]
                ax.scatter(
                    cluster_points[:, i],
                    cluster_points[:, j],
                    c=colors[cluster_idx],
                    label=f"Cluster {cluster_idx}",
                    alpha=0.6,
                )
            if j == 0:
                ax.set_ylabel(f"Feature {i+1}")
            if i == num_features - 1:
                ax.set_xlabel(f"Feature {j+1}")

# 범례는 하나만 추가
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.show()
