import torch


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def cal_weights_via_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors + 1]  # top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

    sum_top_k = torch.sum(sorted_distances[:, 1:num_neighbors + 1],
                          dim=1)  # sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    weights -= torch.diag(torch.diag(weights))
    if links != 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()
    return weights, raw_weights


def spatial_similarity(X, k):
    """
    根据特征矩阵X构建k近邻空间矩阵
    :param X: d x n 特征矩阵，d为维度，n为样本数量
    :param k: 选择的近邻数量
    :return: n x n 的矩阵
    """
    n = X.shape[1]

    # 计算所有样本间的平方距离矩阵（n x n）
    distance_matrix = distance(X, X, square=True)

    # 对每行距离排序，获取排序后的索引（排除自身）
    _, sorted_indices = torch.sort(distance_matrix, dim=1)  # sorted_indices形状为[n, n]

    # 提取每个样本的k近邻索引（跳过第0个自身）
    knn_indices = sorted_indices[:, :k + 1]  # 形状变为[n, k+1]

    # 创建全零矩阵
    A = torch.zeros((n, n), dtype=torch.long).cuda()

    # 生成行索引矩阵
    row_indices = torch.arange(n).unsqueeze(1).expand(n, k + 1).cuda()

    # 计算绝对差值并填充
    abs_diffs = torch.abs(row_indices - knn_indices)
    A[row_indices, knn_indices] = abs_diffs

    row_sums = A.sum(dim=1)
    sigma = row_sums / (k + 1)

    # 使用计算出来的sigma更新矩阵
    spa_matrix = torch.exp(-A / (2 * sigma ** 2))

    return spa_matrix


def dot_product(z, k):
    distances = distance(z.t(), z.t())
    softmax = torch.nn.Softmax(dim=1)
    adj1 = softmax(-distances)
    adj1 = adj1 * spatial_similarity(z.t(), k)
    return adj1
