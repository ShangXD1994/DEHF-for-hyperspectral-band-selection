import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import torch
import scipy.io as sio
from models import DEHF
from utils import graph_construction as graph
from utils import cluster
from utils import data
import torch.nn.functional as F
from torch.optim import Adam
from utils.cal_IE import Entrop
from utils import metrics
import matplotlib
matplotlib.use("Agg")  # 不需要交互式显示

import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'


def check_gradients(model):
    """
    Checks if any of the gradients are NaN, Inf
    """
    for param in model.parameters():
        if param.grad is not None:
            # Check if the gradients contain NaN or Inf
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradient for parameter {param}")
            if torch.isinf(param.grad).any():
                print(f"Inf detected in gradient for parameter {param}")


def train_dtfu(dataset, n_input, n_z, lr, k, epoches, lamb_da, mu):
    num = dataset.x.shape[0]
    model = DEHF.DTFU(512, 256, 128, 128, 256, 512,
                      n_input=n_input,
                      n_z=n_z,
                      k=k, num_bands=num, mu=mu).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=lr)

    # KNN Graph
    data = torch.Tensor(dataset.x).to(device)
    weights, raw_weights = graph.cal_weights_via_CAN(data.t(), k)

    spatial_matrix = graph.spatial_similarity(data.t(), k)
    weights = weights * spatial_matrix

    L = DEHF.get_Laplacian_from_weights(weights)

    print("completed!")

    losses = []
    for epoch in range(epoches):
        x_bar, z, A_pred, Z, Z_ = model(data, L)

        re_loss = F.mse_loss(x_bar, data)
        re_graphloss = torch.sum(weights * torch.log(weights / (A_pred + 10 ** -10) + 10 ** -10))
        re_graphloss = re_graphloss / num

        loss = lamb_da * re_loss + re_graphloss
        losses.append(loss.item())
        print('{} loss: {}, re_loss:{},re_graphloss:{}'.format(epoch, loss.item(), re_loss.item(), re_graphloss.item()))
        optimizer.zero_grad()
        loss.backward()
        check_gradients(model)
        optimizer.step()
    return A_pred, Z_, Z, losses


if __name__ == "__main__":

    # name = 'DC191'
    name = 'IP220'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Original_Img = sio.loadmat('data/Indian_pines.mat')['indian_pines']

    new_l = []
    c = 0
    for k in range(Original_Img.shape[2]):
        l = []
        for i in range(Original_Img.shape[0]):
            for j in range(Original_Img.shape[1]):
                l.append(Original_Img[i][j][k])
                c = c + 1
        new_l.append(l)
    X = np.array(new_l)

    print('data.shape=', X.shape)
    # X归一化
    X_min = X.min(axis=0)  # 对每一列计算最小值
    X_max = X.max(axis=0)  # 对每一列计算最大值
    X = (X - X_min) / (X_max - X_min)  # 对每一列分别进行归一化

    y = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']
    y = y.reshape(X.shape[1])

    dataset = data.load_data(X, y)
    k = 10
    lr = 1e-4
    n_input = X.shape[1]
    n_z = 64
    epoches = 100
    mu = 0.8
    lamb_da = 1000
    A_pred, Z_, Z, losses = train_dtfu(dataset, n_input, n_z, lr, k, epoches, lamb_da, mu)
    A_pred = A_pred.detach().cpu().numpy()
    Z = Z.detach().cpu().numpy()
    sio.savemat('A_pred.mat', {'A': A_pred})

    # 可视化损失函数
    plt.plot(range(1, epoches + 1), losses, marker='o', linestyle='-', color='b', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    # Multi-dimensional Priority-Based Band Selection
    W = sio.loadmat('A_pred')['A']
    W = (W + W.T) / 2

    with open("selected_bands.txt", "w") as file:
        for num_bands in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:

            grp = cluster.spectral_clustering(W, num_bands)

            # 计算 SSIM
            ssim_matrix = metrics.calculate_ssim(X)

            # 计算 IcSDD 和熵
            icSDDs = metrics.calculate_icSDD(X, grp, ssim_matrix)
            entropies = Entrop(X)

            # 归一化 IcSDD 和熵
            icSDDs = (icSDDs - icSDDs.min()) / (icSDDs.max() - icSDDs.min())
            entropies = (entropies - entropies.min()) / (entropies.max() - entropies.min())

            # 合并值
            values = icSDDs + entropies

            # 获取每个类别的最大值及其索引
            max_values = {}
            for i, label in enumerate(grp):
                if label not in max_values or values[i] > max_values[label][0]:
                    max_values[label] = (values[i], i)

            # 存储每个类别最大值样本的索引
            selected_bands = [index for _, index in max_values.values()]
            selected_bands.sort()
            selected_bands = [x + 1 for x in selected_bands]

            print(f"num_bands:{num_bands},{selected_bands}")
            file.write(f"num_bands:{num_bands}, {selected_bands}\n")
