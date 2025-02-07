import torch
import torch.nn as nn

# 定义模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc1 = nn.Linear(40, 1000)
        self.fc2 = nn.Linear(1000, 1)
        self.div = 1000

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        x = x / self.div
        return x