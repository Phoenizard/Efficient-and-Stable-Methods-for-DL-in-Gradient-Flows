import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

data = pd.read_csv('data/Exampe1_D40.csv')
features = data.columns[:-1]
X = data[features].values
y = data['y'].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# 划分训练集和测试集
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)

x_train, x_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# 创建数据加载器
dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)  # 小批量大小为10，数据打乱
