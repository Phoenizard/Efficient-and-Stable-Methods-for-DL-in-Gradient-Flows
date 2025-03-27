import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value


def download_mnist(data_dir='./data'):
    # 定义数据预处理，转换为 Tensor 格式
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 下载 MNIST 训练集和测试集
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    
    print(f"MNIST 数据集已下载到目录 {data_dir}")
    print(f"训练集样本数量：{len(train_set)}")
    print(f"测试集样本数量：{len(test_set)}")


def load_mnist_flat(data_dir='./data'):
    # 定义数据预处理：先转为 tensor，再将图片展平成一维向量（28x28=784）
    transform = transforms.Compose([
        transforms.ToTensor(),                      # 转为 tensor，形状为 (1, 28, 28)
        transforms.Lambda(lambda x: x.view(-1))       # 将 tensor 展平成一维向量，形状为 (784,)
    ])
    
    # 下载 MNIST 数据集（训练集和测试集）
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)
    
    # 构造 X_train, Y_train
    X_train = []
    Y_train = []
    for img, label in train_dataset:
        X_train.append(img.numpy())
        Y_train.append(label)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # 构造 X_test, Y_test
    X_test = []
    Y_test = []
    for img, label in test_dataset:
        X_test.append(img.numpy())
        Y_test.append(label)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_mnist_flat()
    # Convert to torch.Tensor
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test)
    print("X_train shape:", X_train.shape)  # 应为 (60000, 784)
    print("Y_train shape:", Y_train.shape)  # 应为 (60000,)
    print("X_test shape:", X_test.shape)    # 应为 (10000, 784)
    print("Y_test shape:", Y_test.shape)    # 应为 (10000,)
    # Save as .pt file
    torch.save((X_train, Y_train), 'data/MNIST_train_data.pt')
    torch.save((X_test, Y_test), 'data/MNIST_test_data.pt')