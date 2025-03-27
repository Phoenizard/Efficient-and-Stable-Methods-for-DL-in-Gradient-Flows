import numpy as np
import math
import torch
from model.LinearModel import SinCosModel
import torchvision
import torchvision.transforms as transforms


def flatten_params(W: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """将 W 和 a 展平并拼接为一维向量。"""
    return torch.cat([W.view(-1), a.view(-1)])

def unflatten_params(theta: torch.Tensor, W_shape, a_shape):
    """将一维向量 theta 拆分并 reshape 回 W 和 a 的形状。"""
    w_size = W_shape[0] * W_shape[1]
    W_new = theta[:w_size].view(*W_shape)
    a_new = theta[w_size:].view(*a_shape)
    return W_new, a_new

def flatten_grad(model):
    """从 model.W.grad, model.a.grad 中取出梯度并展平。"""
    gW = model.W.grad
    ga = model.a.grad
    return torch.cat([gW.view(-1), ga.view(-1)])

def compute_jacobian(theta, X, model):
    """
    计算 f(theta) 关于 theta 的雅各比矩阵，其中
      - theta: 展平后的模型参数，形状 (num_parameters,)
      - X: 当前 batch 的输入，形状 (batch_size, input_dim)
      - model: 模型对象，用于提供参数维度信息
    返回：
      - J: 雅各比矩阵，形状 (batch_size, num_parameters)
    """
    # 定义以 theta 为自变量的函数 f_theta
    def f_theta(theta_vec):
        W_temp, a_temp = unflatten_params(theta_vec, model.W.shape, model.a.shape)
        ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
        x_aug = torch.cat([X, ones], dim=1)
        hidden = torch.relu(x_aug @ W_temp)
        out = hidden @ a_temp  # shape: (batch_size, 1)
        return out.squeeze(-1)  # shape: (batch_size,)

    # 计算雅各比矩阵，输出形状为 (num_parameters, batch_size)，需要转置
    J = torch.autograd.functional.jacobian(f_theta, theta, vectorize=True)
    return J
