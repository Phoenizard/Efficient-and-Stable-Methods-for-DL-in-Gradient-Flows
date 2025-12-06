"""
Optimization algorithms for gradient flows in deep learning.
All algorithms return a history dictionary with training and test losses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
from model import LinearModel
from utilize import flatten_params, unflatten_params, flatten_grad, compute_jacobian, compute_jacobian_C


def sgd_regression(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                   learning_rate=0.1, num_epochs=100, device='cuda'):
    """
    SGD optimization for regression tasks.

    Args:
        x_train, y_train: Training data
        x_test, y_test: Test data
        m: Number of neurons
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Number of epochs
        device: 'cuda' or 'cpu'

    Returns:
        hist: Dictionary with 'train_loss' and 'test_loss' lists
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LinearModel.SinCosModel(m=m)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(x_test), y_test).item()
            test_losses.append(test_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'model': model}


def sgd_classification(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                       learning_rate=0.1, num_epochs=100, inputs=784, outputs=10, device='cuda'):
    """
    SGD optimization for classification tasks.

    Returns:
        hist: Dictionary with 'train_loss', 'test_loss', and 'test_accuracy' lists
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LinearModel.ClassificationModel(m=m, inputs=inputs, outputs=outputs)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test Accuracy: {accuracy:.2f}%")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'test_accuracy': test_accuracies, 'model': model}


def adam_regression(x_train, y_train, x_test, y_test, m=100, batch_size=64,
                    learning_rate=0.1, num_epochs=50000, device='cuda'):
    """Adam optimization for regression tasks."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LinearModel.SinCosModel(m=m)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(x_test), y_test).item()
            test_losses.append(test_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'model': model}


def sav_regression(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                   C=100, lambda_=4, dt=0.1, num_epochs=50000, device='cuda'):
    """
    SAV (Scalar Auxiliary Variable) optimization for regression tasks.

    Args:
        C: SAV constant to ensure loss + C >= 0
        lambda_: Coefficient for linear operator
        dt: Time step (learning rate)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LinearModel.SinCosModel(m=m)
    model.to(device)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    r = None

    for epoch in range(num_epochs):
        r = None
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)
            if r is None:
                r = math.sqrt(loss.item() + C)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                theta_n = flatten_params(model.W, model.a)
                grad_n = flatten_grad(model)
                inv_operator = 1.0 / (1.0 + dt * lambda_)
                grad_scaled = grad_n * inv_operator

                alpha = dt / math.sqrt(loss.item() + C)
                theta_n_2 = - alpha * grad_scaled

                dot_val = torch.dot(grad_n, grad_scaled)
                denom = 1.0 + dt * dot_val / (2.0 * (loss.item() + C))
                r = r / denom

                theta_n_plus_1 = theta_n + r * theta_n_2

                W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

        with torch.no_grad():
            model.eval()
            train_loss = criterion(model(x_train), y_train).item()
            test_loss = criterion(model(x_test), y_test).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'model': model}


def sav_classification(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                       C=100, lambda_=4, dt=0.1, num_epochs=100, inputs=784, outputs=10, device='cuda'):
    """SAV optimization for classification tasks."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LinearModel.ClassificationModel(m=m, inputs=inputs, outputs=outputs)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_accuracies = []
    r = None

    for epoch in range(num_epochs):
        r = None
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)
            if r is None:
                r = math.sqrt(loss.item() + C)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                theta_n = flatten_params(model.W, model.a)
                grad_n = flatten_grad(model)
                inv_operator = 1.0 / (1.0 + dt * lambda_)
                grad_scaled = grad_n * inv_operator

                alpha = dt / math.sqrt(loss.item() + C)
                theta_n_2 = - alpha * grad_scaled

                dot_val = torch.dot(grad_n, grad_scaled)
                denom = 1.0 + dt * dot_val / (2.0 * (loss.item() + C))
                r = r / denom

                theta_n_plus_1 = theta_n + r * theta_n_2

                W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            train_losses.append(train_loss)

            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            test_loss /= len(test_dataset)
            test_losses.append(test_loss)
            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test Accuracy: {accuracy:.2f}%")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'test_accuracy': test_accuracies, 'model': model}


def esav_regression(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                    C=1, lambda_=1, dt=0.1, num_epochs=50000, device='cuda'):
    """
    ExpSAV (Exponential SAV) optimization for regression tasks.
    Uses exponential auxiliary variable for better numerical stability.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LinearModel.SinCosModel(m=m)
    model.to(device)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    r = None

    for epoch in range(num_epochs):
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)
            if r is None:
                r = C * math.exp(loss.item())

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                theta_n = flatten_params(model.W, model.a)
                grad_n = flatten_grad(model)
                inv_operator = 1.0 / (1.0 + dt * lambda_)
                grad_scaled = grad_n * inv_operator

                alpha = dt / (C * math.exp(loss.item()))
                theta_n_2 = - alpha * grad_scaled

                dot_val = torch.dot(grad_n, grad_scaled)
                denom = 1.0 + dt * dot_val
                r = r / denom

                theta_n_plus_1 = theta_n + r * theta_n_2

                W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

        with torch.no_grad():
            model.eval()
            train_loss = criterion(model(x_train), y_train).item()
            test_loss = criterion(model(x_test), y_test).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'model': model}


def esav_classification(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                        C=1, lambda_=0.0, dt=0.1, num_epochs=100, inputs=784, outputs=10, device='cuda'):
    """ExpSAV optimization for classification tasks."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LinearModel.ClassificationModel(m=m, inputs=inputs, outputs=outputs)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_accuracies = []
    r = None

    for epoch in range(num_epochs):
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)
            if r is None:
                r = C * math.exp(loss.item())

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                theta_n = flatten_params(model.W, model.a)
                grad_n = flatten_grad(model)
                inv_operator = 1.0 / (1.0 + dt * lambda_)
                grad_scaled = grad_n * inv_operator

                alpha = dt / (C * math.exp(loss.item()))
                theta_n_2 = - alpha * grad_scaled

                dot_val = torch.dot(grad_n, grad_scaled)
                denom = 1.0 + dt * dot_val
                r = r / denom

                theta_n_plus_1 = theta_n + r * theta_n_2

                W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            train_losses.append(train_loss)

            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            test_loss /= len(test_dataset)
            test_losses.append(test_loss)
            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test Accuracy: {accuracy:.2f}%")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'test_accuracy': test_accuracies, 'model': model}


def ieq_regression(x_train, y_train, x_test, y_test, m=100, batch_size=64,
                   dt=0.1, num_epochs=50000, device='cuda'):
    """
    IEQ (Invariant Energy Quadratization) with full Jacobian for regression tasks.
    Uses exact solution via (I + Δt J J^T)^{-1}.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LinearModel.SinCosModel(m=m)
    model.to(device)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        for X, Y in train_loader:
            theta_n = flatten_params(model.W, model.a)
            theta_n = theta_n.clone().detach().requires_grad_(True)

            f_theta = model(X).squeeze(-1)
            q_n = f_theta - Y.squeeze(-1)

            J = compute_jacobian(theta_n, X, model)
            batch_size_curr = X.size(0)
            I = torch.eye(batch_size_curr, device=X.device, dtype=X.dtype)
            JJt = J @ J.t()
            A_mat = I + dt * JJt
            q_np1 = torch.linalg.solve(A_mat, q_n)

            theta_np1 = theta_n - dt * (J.t() @ q_np1)

            with torch.no_grad():
                W_new, a_new = unflatten_params(theta_np1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

        with torch.no_grad():
            model.eval()
            train_loss = criterion(model(x_train), y_train).item()
            test_loss = criterion(model(x_test), y_test).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'model': model}


def ieq_classification(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                       dt=0.1, num_epochs=100, inputs=784, outputs=10, device='cuda'):
    """IEQ with full Jacobian for classification tasks."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Convert labels to one-hot encoding
    y_train_onehot = nn.functional.one_hot(y_train, num_classes=outputs)
    y_test_onehot = nn.functional.one_hot(y_test, num_classes=outputs)

    train_dataset = TensorDataset(x_train, y_train_onehot)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LinearModel.ClassificationModel(m=m, inputs=inputs, outputs=outputs)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        for X, Y in train_loader:
            theta_n = flatten_params(model.W, model.a)
            theta_n = theta_n.clone().detach().requires_grad_(True)

            f_theta = model(X)
            q_n = (f_theta - Y).reshape(-1)

            J = compute_jacobian_C(theta_n, X, model)
            batch_size_curr, _ = f_theta.shape
            total_dim = batch_size_curr * outputs
            I = torch.eye(total_dim, device=X.device, dtype=X.dtype)
            JJt = J @ J.t()
            A_mat = I + dt * JJt
            q_np1 = torch.linalg.solve(A_mat, q_n)

            theta_np1 = theta_n - dt * (J.t() @ q_np1)

            with torch.no_grad():
                W_new, a_new = unflatten_params(theta_np1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

        model.eval()
        test_loss = 0.0
        accuracy = 0
        total = 0
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            train_losses.append(train_loss)

            for batch_x, batch_y in test_loader:
                outputs_pred = model(batch_x)
                loss = criterion(outputs_pred, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs_pred.data, 1)
                total += batch_y.size(0)
                accuracy += (predicted == batch_y).sum().item()

            test_loss /= len(test_dataset)
            test_losses.append(test_loss)
            acc_percent = 100 * accuracy / total
            test_accuracies.append(acc_percent)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test Accuracy: {acc_percent:.2f}%")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'test_accuracy': test_accuracies, 'model': model}


def ieq_adaptive_regression(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                            dt=0.1, epsilon=1e-8, num_epochs=50000, device='cuda'):
    """
    IEQ Adaptive optimization for regression tasks.
    Uses adaptive scaling factor α^n for O(n) complexity instead of O(n³) full Jacobian.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LinearModel.SinCosModel(m=m)
    model.to(device)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    q = None

    for epoch in range(num_epochs):
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)

            if q is None:
                q = pred.detach() - Y

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                theta_n = flatten_params(model.W, model.a)
                grad_n = flatten_grad(model)

                grad_norm_sq = torch.norm(grad_n) ** 2
                q_norm_sq = torch.norm(q) ** 2
                alpha = 1.0 / (1.0 + dt * grad_norm_sq / (q_norm_sq + epsilon))

                theta_n_plus_1 = theta_n - dt * alpha * grad_n

                W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

                q = q / (1.0 + dt * grad_norm_sq / (q_norm_sq + epsilon))

        with torch.no_grad():
            model.eval()
            train_loss = criterion(model(x_train), y_train).item()
            test_loss = criterion(model(x_test), y_test).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'model': model}


def ieq_adaptive_classification(x_train, y_train, x_test, y_test, m=100, batch_size=256,
                                dt=0.1, epsilon=1e-8, num_epochs=100, inputs=784, outputs=10, device='cuda'):
    """IEQ Adaptive optimization for classification tasks."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LinearModel.ClassificationModel(m=m, inputs=inputs, outputs=outputs)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_accuracies = []
    q = None

    for epoch in range(num_epochs):
        for X, Y in train_loader:
            pred = model(X)
            loss = criterion(pred, Y)

            if q is None:
                y_onehot = nn.functional.one_hot(Y, num_classes=outputs).float()
                q = pred.detach() - y_onehot

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                theta_n = flatten_params(model.W, model.a)
                grad_n = flatten_grad(model)

                grad_norm_sq = torch.norm(grad_n) ** 2
                q_norm_sq = torch.norm(q) ** 2
                alpha = 1.0 / (1.0 + dt * grad_norm_sq / (q_norm_sq + epsilon))

                theta_n_plus_1 = theta_n - dt * alpha * grad_n

                W_new, a_new = unflatten_params(theta_n_plus_1, model.W.shape, model.a.shape)
                model.W.copy_(W_new)
                model.a.copy_(a_new)

                q = q / (1.0 + dt * grad_norm_sq / (q_norm_sq + epsilon))

        model.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            train_losses.append(train_loss)

            for batch_x, batch_y in test_loader:
                outputs_pred = model(batch_x)
                loss = criterion(outputs_pred, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs_pred.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            test_loss /= len(test_dataset)
            test_losses.append(test_loss)
            acc_percent = 100 * correct / total
            test_accuracies.append(acc_percent)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test Accuracy: {acc_percent:.2f}%")

    return {'train_loss': train_losses, 'test_loss': test_losses, 'test_accuracy': test_accuracies, 'model': model}
