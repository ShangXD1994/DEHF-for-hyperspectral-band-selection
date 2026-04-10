import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # Encoder Layers
        self.enc_1 = Linear(n_input, n_enc_1)
        self.bn1 = nn.BatchNorm1d(n_enc_1)  # BatchNorm after Linear
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.bn2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.bn3 = nn.BatchNorm1d(n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        # Decoder Layers
        self.dec_1 = Linear(n_z, n_dec_1)
        self.bn4 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.bn5 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.bn6 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        # Encoder: Linear + BatchNorm + ReLU
        enc_h1 = F.relu(self.bn1(self.enc_1(x)))
        enc_h2 = F.relu(self.bn2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.bn3(self.enc_3(enc_h2)))
        z = self.z_layer(enc_h3)

        # Decoder: Linear + BatchNorm + ReLU
        dec_h1 = F.relu(self.bn4(self.dec_1(z)))
        dec_h2 = F.relu(self.bn5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.bn6(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h1, dec_h2, dec_h3
