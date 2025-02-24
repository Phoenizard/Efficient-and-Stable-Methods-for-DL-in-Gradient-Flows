import torch
import torch.nn as nn

class RelativeErrorLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(RelativeErrorLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # 计算每个样本的相对误差: |y_true - y_pred| / (|y_true| + eps)
        rel_error = torch.abs(y_true - y_pred) / (torch.abs(y_true) + self.eps)
        # 对所有样本求平均
        return torch.mean(rel_error)

# 示例使用方法：
if __name__ == "__main__":
    # 构造示例数据
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0])
    y_pred = torch.tensor([2.1, 3.9, 5.8, 8.2])
    
    # 初始化损失函数
    criterion = RelativeErrorLoss()
    
    # 计算损失
    loss = criterion(y_pred, y_true)
    print("Relative Error Loss:", loss.item())