from sklearn.cluster import KMeans
import numpy as np


# 计算标准拉普拉斯矩阵
def compute_laplacian(W):
    D = np.diag(W.sum(axis=1))  # 度矩阵，D[i, i] 是第 i 个节点的度数
    L = D - W  # 标准拉普拉斯矩阵
    return L


# 特征值分解并选择 k 个最小特征值对应的特征向量
def compute_eigenvectors(L, k):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvectors[:, :k]


# 使用 K-means 聚类
def perform_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    return kmeans.fit_predict(X)


# 主函数
def spectral_clustering(W, n_clusters):
    L = compute_laplacian(W)
    k = n_clusters  # 聚类数就是选取的特征向量个数
    eigenvectors = compute_eigenvectors(L, k)
    labels = perform_kmeans(eigenvectors, n_clusters)
    return labels
