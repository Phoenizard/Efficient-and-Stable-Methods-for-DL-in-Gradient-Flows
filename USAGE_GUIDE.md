# 使用指南 (Usage Guide)

## 项目重构说明

本项目已经重构，所有优化算法已封装为标准函数接口，每个实验都有对应的独立脚本。

## 文件结构

```
.
├── algorithms.py                      # 所有优化算法的函数实现
├── generate_data.py                   # 生成所有实验数据
├── experiment_regression_1.py         # 实验1: Sin+Cos回归任务
├── experiment_regression_2.py         # 实验2: 二次函数回归任务
├── experiment_regression_3.py         # 实验3: 高斯函数回归任务
├── experiment_classification_mnist.py # 实验4: MNIST分类任务
├── model/LinearModel.py               # 神经网络模型定义
├── utilize.py                         # 工具函数
├── data/                              # 数据目录
└── results/                           # 实验结果目录
```

## 快速开始

### 1. 生成实验数据

首先运行数据生成脚本:

```bash
python generate_data.py
```

这将在 `data/` 目录下生成以下数据文件:
- `Example1_train_data.pt`, `Example1_test_data.pt` (40D Sin+Cos函数)
- `Example2_train_data.pt`, `Example2_test_data.pt` (40D 二次函数)
- `Example3_train_data.pt`, `Example3_test_data.pt` (40D 高斯函数)
- `Gaussian_train_data.pt`, `Gaussian_test_data.pt` (1D 高斯函数，用于兼容)

### 2. 运行实验

#### 实验1: Sin+Cos回归

```bash
python experiment_regression_1.py
```

比较的算法:
- SGD
- Adam
- SAV
- ExpSAV
- IEQ (Full Jacobian)
- IEQ Adaptive

结果保存在: `results/experiment_1/`

#### 实验2: 二次函数回归

```bash
python experiment_regression_2.py
```

比较的算法: 同实验1

结果保存在: `results/experiment_2/`

#### 实验3: 高斯函数回归

```bash
python experiment_regression_3.py
```

比较的算法:
- SGD
- Adam
- SAV
- ExpSAV
- IEQ Adaptive

(由于计算成本，跳过了IEQ Full Jacobian)

结果保存在: `results/experiment_3/`

#### 实验4: MNIST分类

首先确保MNIST数据存在:

```bash
cd data/MNIST
python MNIST.py  # 如果数据不存在
cd ../..
```

然后运行实验:

```bash
python experiment_classification_mnist.py
```

比较的算法:
- SGD
- SAV
- ExpSAV
- IEQ Adaptive

结果保存在: `results/experiment_mnist/`

## 使用算法模块

你也可以直接在自己的代码中使用 `algorithms.py` 中的函数:

```python
from algorithms import sav_regression, adam_regression
import torch

# 加载数据
(x_train, y_train) = torch.load('data/Example1_train_data.pt')
(x_test, y_test) = torch.load('data/Example1_test_data.pt')

# 运行SAV算法
hist = sav_regression(
    x_train, y_train, x_test, y_test,
    m=100,           # 神经元数量
    batch_size=256,  # 批量大小
    C=100,           # SAV常数
    lambda_=4,       # 线性算子系数
    dt=0.1,          # 时间步长(学习率)
    num_epochs=5000, # 训练轮数
    device='cuda'    # 使用GPU
)

# 访问结果
print(f"最终训练损失: {hist['train_loss'][-1]}")
print(f"最终测试损失: {hist['test_loss'][-1]}")

# 获取训练好的模型
model = hist['model']
```

## 算法函数接口

所有算法函数都返回一个字典 `hist`，包含:

### 回归任务
```python
{
    'train_loss': [...]  # 训练损失列表
    'test_loss': [...]   # 测试损失列表
    'model': model       # 训练好的模型
}
```

### 分类任务
```python
{
    'train_loss': [...]      # 训练损失列表
    'test_loss': [...]       # 测试损失列表
    'test_accuracy': [...]   # 测试准确率列表
    'model': model           # 训练好的模型
}
```

## 可用的算法函数

### 回归任务
- `sgd_regression()` - 标准梯度下降
- `adam_regression()` - Adam优化器
- `sav_regression()` - SAV方法
- `esav_regression()` - ExpSAV方法 (改进的数值稳定性)
- `ieq_regression()` - IEQ方法 (完整Jacobian)
- `ieq_adaptive_regression()` - IEQ自适应方法

### 分类任务
- `sgd_classification()` - 标准梯度下降
- `sav_classification()` - SAV方法
- `esav_classification()` - ExpSAV方法
- `ieq_classification()` - IEQ方法 (完整Jacobian)
- `ieq_adaptive_classification()` - IEQ自适应方法

## GPU加速

所有算法都支持GPU加速。只需确保安装了支持CUDA的PyTorch，脚本会自动检测并使用GPU:

```python
device = 'cuda' if torch.cuda.is_available() else \
         ('mps' if torch.backends.mps.is_available() else 'cpu')
```

## 实验结果

每个实验脚本运行后会生成:
1. `results.pt` - 包含所有算法结果的PyTorch文件
2. `loss_comparison.png` 或 `metrics_comparison.png` - 损失函数/指标对比图

## 论文对应关系

- **Example 1** (experiment_regression_1.py): 论文公式(25), Section 3.1.1
- **Example 2** (experiment_regression_2.py): 论文公式(26), Section 3.1.2
- **Example 3** (experiment_regression_3.py): 论文公式(27), Section 3.1.3
- **Example 4** (experiment_classification_mnist.py): 论文Section 3.2

## 注意事项

1. 确保有足够的GPU内存运行大规模实验 (特别是Example 3)
2. IEQ Full Jacobian方法计算成本高，适合小批量
3. 所有随机种子已固定 (np.random.seed(0), torch.manual_seed(0)) 以确保可重复性
4. 实验参数已根据论文设置，但可以根据需要调整

## 疑难解答

### MNIST数据未找到
```bash
cd data/MNIST
python MNIST.py
cd ../..
```

### CUDA内存不足
降低批量大小或神经元数量:
```python
hist = sav_regression(..., m=50, batch_size=128)
```

### 实验运行时间过长
减少训练轮数:
```python
hist = sav_regression(..., num_epochs=100)
```

## 引用

如果使用本代码，请引用原论文:

```
Ziqi Ma, Zhiping Mao, Jie Shen. "Efficient and stable SAV-based methods for gradient flows
arising from deep learning." Journal of Computational Physics, 505, 112911, 2024.
```
