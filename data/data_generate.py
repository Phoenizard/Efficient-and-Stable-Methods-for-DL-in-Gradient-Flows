import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

def SinCos_dataset(num_points=1000, x_range=(-2 * np.pi, 2 * np.pi)):
    # 生成 x 值
    x = np.linspace(x_range[0], x_range[1], num_points, dtype=np.float32)
    y = np.sin(x) + np.random.normal(0, 0.01, size=x.shape)
    # 转换为 torch 张量，并扩展维度使其符合 (N, 1) 的形状
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    # 打乱数据集
    idx = np.random.permutation(num_points)
    x_tensor, y_tensor = x_tensor[idx], y_tensor[idx]
    # 分割数据集
    n_train = int(num_points * 0.8)
    x_train, x_test = x_tensor[:n_train], x_tensor[n_train:]
    y_train, y_test = y_tensor[:n_train], y_tensor[n_train:]
    return (x_train, y_train), (x_test, y_test)

def Gaussian_dataset(num_points=1000, var=1, mean=0):
    # y = exp(x^2) + noise
    x = np.random.normal(mean, var, num_points).astype(np.float32)
    y = np.exp(-x ** 2) + np.random.normal(0, 0.01, size=x.shape)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    idx = np.random.permutation(num_points)
    x_tensor, y_tensor = x_tensor[idx], y_tensor[idx]
    n_train = int(num_points * 0.8)
    x_train, x_test = x_tensor[:n_train], x_tensor[n_train:]
    y_train, y_test = y_tensor[:n_train], y_tensor[n_train:]
    return (x_train, y_train), (x_test, y_test)

def main():
    #=======================SinCos Data Generate================================
    # (x_train, y_train), (x_test, y_test) = SinCos_dataset()
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    # torch.save((x_train, y_train), 'data/SinCos_train_data.pt')
    # torch.save((x_test, y_test), 'data/SinCos_test_data.pt')
    # print("Data saved.")
    #=======================Gaussian Data Generate===============================
    (x_train, y_train), (x_test, y_test) = Gaussian_dataset()
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    torch.save((x_train, y_train), 'data/Gaussian_train_data.pt')
    torch.save((x_test, y_test), 'data/Gaussian_test_data.pt')
    print("Data saved.")

if __name__ == '__main__':
    main()