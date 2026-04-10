import os

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import wasserstein_distance
from joblib import Parallel, delayed  # 用于并行化


# 计算 SSIM
def calculate_ssim(X):
    num_bands = X.shape[0]
    ssim_matrix = np.zeros((num_bands, num_bands))

    def compute_ssim(i, j):
        return ssim(X[i], X[j], data_range=1)

    results = Parallel(n_jobs=-1)(
        delayed(compute_ssim)(i, j) for i in range(num_bands) for j in range(i + 1, num_bands))

    idx = 0
    for i in range(num_bands):
        for j in range(i + 1, num_bands):
            ssim_matrix[i, j] = results[idx]
            ssim_matrix[j, i] = results[idx]  # 对称填充下三角矩阵
            idx += 1

    return ssim_matrix


# 计算 IcSDD
def calculate_icSDD(X, grp, ssim_matrix):
    num_bands = X.shape[0]
    icSDDs = np.zeros(num_bands)

    for i in range(num_bands):
        same_cluster = grp == grp[i]  # 同簇波段
        different_cluster = grp != grp[i]  # 不同簇波段

        positive_similarities = ssim_matrix[i, same_cluster]
        negative_similarities = ssim_matrix[i, different_cluster]

        # 计算 IcSDD
        icSDD = wasserstein_distance(positive_similarities, negative_similarities)
        icSDDs[i] = icSDD

    return icSDDs
