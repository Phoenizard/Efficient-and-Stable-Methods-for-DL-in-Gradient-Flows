import numpy as np
import torch

def generate_dataset(num_points=1000, x_range=(-2 * np.pi, 2 * np.pi)):
    # 生成 x 值
    x = np.linspace(x_range[0], x_range[1], num_points, dtype=np.float32)
    y = np.sin(x) + np.cos(x) + np.random.normal(0, 0.1, x.shape)
    # 转换为 torch 张量，并扩展维度使其符合 (N, 1) 的形状
    x_tensor = torch.tensor(x).unsqueeze(1)
    y_tensor = torch.tensor(y).unsqueeze(1)
    return x_tensor, y_tensor

def main():
    # 生成数据集
    x_tensor, y_tensor = generate_dataset(num_points=1000)
    # 保存数据到文件
    torch.save((x_tensor, y_tensor), 'dataset.pt')
    print("数据集已保存至 dataset.pt")

if __name__ == '__main__':
    main()