import torch
import torch.nn as nn
import torch.optim as optim
import data
import matplotlib.pyplot as plt
import numpy as np
# TODO testirati i jupyter kod sa zadacima


def MSE(y_gt, y_predicted):
    return torch.mean((y_gt - y_predicted) ** 2)


def linear_regression(X, Y_, param_niter=100, param_delta=0.1):
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    optimizer = optim.SGD([a, b], lr=param_delta)

    for i in range(param_niter):
        # simple regression model
        Y = a*X + b

        # Mean Square Error
        loss = MSE(Y_, Y)

        # gradient calculation
        loss.backward()

        # update of gradients
        optimizer.step()

        if i % 10 == 0:
            print(f'\nstep: {i}, loss:{loss}, a:{a}, b {b}')
            diff = Y - Y_
            my_grad_a = 2 * torch.mean(diff * X)
            my_grad_b = 2 * torch.mean(diff)
            print(f"\tstep: {i}, my a grad: {my_grad_a.detach().numpy()}, my b grad: {my_grad_b.detach().numpy()}")
            print(f"\tstep: {i}, pytorch a : {a.grad.detach().numpy()[0]}, pytorch b: {b.grad.detach().numpy()[0]}")

        # gradients to zero for next pass
        optimizer.zero_grad()

    return a, b


if __name__ == '__main__':
    np.random.seed(100)
    x, y = data.sample_lr_data(100, 15, 2)
    plt.scatter(x, y)
    plt.show()
    X = torch.Tensor(x)
    Y_ = torch.Tensor(y)
    a, b = linear_regression(X, Y_, 1000, 0.1)
    y2 = X*a + b
    y2 = y2.detach().numpy()
    plt.scatter(x, y)
    plt.scatter(x, y2)
    plt.show()
