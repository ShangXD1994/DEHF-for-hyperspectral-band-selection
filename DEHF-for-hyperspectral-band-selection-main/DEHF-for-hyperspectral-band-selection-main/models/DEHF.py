from models.AE import AE
import torch.nn as nn
import torch
from models.GNN import GNNLayer
from models.HFF import FusionLayer
import scipy.io as sio
from utils.graph_construction import dot_product


def get_Laplacian_from_weights(weights):
    degree = torch.sum(weights, dim=1).pow(-0.5)
    return (weights * degree).t() * degree


class DTFU(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, k, num_bands, mu):
        super(DTFU, self).__init__()

        # autoencoder for intra information
        self.ael = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)


        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_dec_1)

        self.gnn_7 = GNNLayer(n_dec_1, n_dec_2)
        self.gnn_8 = GNNLayer(n_dec_2, n_dec_3)
        self.gnn_9 = GNNLayer(n_dec_3, n_input)

        self.fuse1 = FusionLayer(n_enc_1)
        self.fuse2 = FusionLayer(n_enc_2)
        self.fuse3 = FusionLayer(n_enc_3)
        self.fuse4 = FusionLayer(n_z)

        self.fuse5 = FusionLayer(n_dec_1)
        self.fuse6 = FusionLayer(n_dec_2)
        self.fuse7 = FusionLayer(n_dec_3)
        self.fuse8 = FusionLayer(n_input)

        # degree
        self.k = k
        self.num_bands = num_bands
        self.mu = mu

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, dec_1, dec_2, dec_3 = self.ael(x)

        sigma = self.mu

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.fuse1(h, tra1)
        adj1 = dot_product(h, self.k)
        adj1 = (1 - sigma) * adj + sigma * adj1
        adj1 = get_Laplacian_from_weights(adj1)  # 归一化

        h = self.gnn_2(h, adj1)
        h = self.fuse2(h, tra2)
        adj2 = dot_product(h, self.k)
        adj2 = (1 - sigma) * adj + sigma * adj2
        adj2 = get_Laplacian_from_weights(adj2)

        h = self.gnn_3(h, adj2)
        h = self.fuse3(h, tra3)
        adj3 = dot_product(h, self.k)
        adj3 = (1 - sigma) * adj + sigma * adj3
        adj3 = get_Laplacian_from_weights(adj3)

        h = self.gnn_4(h, adj3)
        h = self.fuse4(h, z)
        adj4 = dot_product(h, self.k)
        adj4 = (1 - sigma) * adj + sigma * adj4
        adj4 = get_Laplacian_from_weights(adj4)

        h1 = h
        h = self.gnn_5(h, adj4)
        h = self.fuse5(h, dec_1)
        adj5 = dot_product(h, self.k)
        adj5 = (1 - sigma) * adj + sigma * adj5
        adj5 = get_Laplacian_from_weights(adj5)

        h = self.gnn_7(h, adj5)
        h = self.fuse6(h, dec_2)
        adj6 = dot_product(h, self.k)
        adj6 = (1 - sigma) * adj + sigma * adj6
        adj6 = get_Laplacian_from_weights(adj6)

        h = self.gnn_8(h, adj6)
        h = self.fuse7(h, dec_3)
        adj7 = dot_product(h, self.k)
        adj7 = (1 - sigma) * adj + sigma * adj7
        adj7 = get_Laplacian_from_weights(adj7)

        h = self.gnn_9(h, adj7)
        h = self.fuse8(h, x_bar)
        A_pred = dot_product(h, self.k)
        A_pred = (1 - sigma) * adj + sigma * A_pred

        return x_bar, z, A_pred, h, h1
