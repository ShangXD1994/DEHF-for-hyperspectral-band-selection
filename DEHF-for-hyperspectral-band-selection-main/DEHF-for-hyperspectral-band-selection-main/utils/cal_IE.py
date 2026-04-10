import numpy as np


# 计算特征矩阵的熵
def Entrop(X):
    G = 256
    L, N = X.shape
    rak_val = np.zeros(L)
    minX = np.min(X)
    maxX = np.max(X)
    edge = np.linspace(minX, maxX, G)
    for i in range(L):
        histX, _ = np.histogram(X[i, :], bins=edge, density=False)
        histX = histX / histX.sum()
        rak_val[i] = -np.sum(histX * np.log2(histX + np.finfo(float).eps))
    return rak_val
