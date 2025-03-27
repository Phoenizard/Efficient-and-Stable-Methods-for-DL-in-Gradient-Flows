import torch
import torch.nn as nn

class SinCosModel(nn.Module):
    def __init__(self, m):
        super(SinCosModel, self).__init__()
        # 定义第一层参数：W 的形状为 (2, m)
        # 其中 2 表示输入值和偏置项
        self.W = nn.Parameter(torch.empty(2, m))
        # 定义第二层参数：a 的形状为 (m, 1)
        self.a = nn.Parameter(torch.empty(m, 1))
        # 使用 He 初始化（Kaiming 初始化）
        nn.init.kaiming_normal_(self.W, nonlinearity='relu')
        nn.init.kaiming_normal_(self.a, nonlinearity='relu')
    
    def forward(self, x):
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, ones], dim=1)
        hidden = torch.relu(x_aug @ self.W)
        out = hidden @ self.a
        return out


class ClassificationModel(nn.Module):
    def __init__(self, m, inputs=1, outputs=10):
        super(ClassificationModel, self).__init__()
        # 定义第一层参数：W 的形状为 (D + 1, m)
        # 其中 2 表示输入值和偏置项
        self.W = nn.Parameter(torch.empty(inputs + 1, m))
        # 定义第二层参数：a 的形状为 (m, 1)
        self.a = nn.Parameter(torch.empty(m, outputs))
        # 使用 He 初始化（Kaiming 初始化）
        nn.init.kaiming_normal_(self.W, nonlinearity='relu')
        nn.init.kaiming_normal_(self.a, nonlinearity='relu')
    
    def forward(self, x):
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, ones], dim=1)
        hidden = torch.relu(x_aug @ self.W)
        out = hidden @ self.a
        # 使用 softmax 函数将输出转换为概率
        prob = torch.softmax(out, dim=1)
        return prob


if __name__ == '__main__':
    model = SinCosModel(m=64)
    print(model)