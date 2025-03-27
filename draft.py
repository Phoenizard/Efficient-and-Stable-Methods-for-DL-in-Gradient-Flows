import torch
import torch.nn.functional as F

# 参数设定: N=4, D=1, m=1
N, D, m = 4, 1, 1

# 手动设定输入矩阵 x (包含偏置项)
x = torch.tensor([[1, 1],
                  [2, 1],
                  [3, 1],
                  [4, 1]], dtype=torch.float32)

# 手动设定权重矩阵 W 和参数 A
W = torch.tensor([[2.0], [3.0]], requires_grad=True)  # 形状: [2, 1]
A = torch.tensor([[4.0]], requires_grad=True)           # 形状: [1, 1]

# 定义函数 f(x;theta)=ReLU(xW)*A
def f(x, W, A):
    return torch.relu(x @ W) @ A  # 结果形状: [4, 1]

# 为了计算雅可比，定义函数 func 接受 W, A 为独立参数，并返回一维张量（每个样本对应一个标量）
def func(W, A):
    return f(x, W, A).squeeze()  # squeeze 后形状为 [4]

# 计算雅可比矩阵：返回一个元组 (jac_W, jac_A)
jac = torch.autograd.functional.jacobian(func, (W, A))

# 将各部分雅可比展开并拼接
jac_W = jac[0].reshape(N, -1)  # 对 W 的雅可比，形状应为 [4, 2]
jac_A = jac[1].reshape(N, -1)  # 对 A 的雅可比，形状应为 [4, 1]

# 拼接后得到对所有参数的雅可比矩阵，形状 [4, 3]
J = torch.cat([jac_W, jac_A], dim=1)

print("雅可比矩阵 J (shape {}):".format(J.shape))
print(J)