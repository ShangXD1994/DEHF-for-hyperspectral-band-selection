import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, last_dim, n_num):
        super(AttentionLayer, self).__init__()
        self.n_num = n_num
        self.fc1 = nn.Linear(n_num * last_dim, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, n_num)
        self.attention = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.T = 10
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        attention_sample = self.attention(x / self.T)
        attention_view = torch.mean(attention_sample, dim=0, keepdim=True).squeeze()
        return attention_view


class FusionLayer(nn.Module):
    def __init__(self, last_dim, n_num=2):
        super(FusionLayer, self).__init__()
        self.n_num = n_num
        self.attentionLayer = AttentionLayer(last_dim, n_num)

    def forward(self, x, k):
        y = torch.cat((x, k), 1)
        weights = self.attentionLayer(y)
        x_TMP = weights[0] * x + weights[1] * k
        return x_TMP
