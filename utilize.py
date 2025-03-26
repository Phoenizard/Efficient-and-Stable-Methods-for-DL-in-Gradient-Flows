import numpy as np
import math
import torch

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