import torch
import torch.nn as nn
import torch.optim as optim


def MSE(y_gt, y_predicted):
    return torch.mean((y_gt - y_predicted) ** 2)


def linear_regression(X, Y, param_niter=100, param_delta=0.1):
    a = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    optimizer = optim.SGD([a, b], lr=param_delta)

    for i in range(param_niter):
        # simple regression model
        Y_ = a*X + b

        diff = (Y-Y_)

        # Mean Square Error
        loss = MSE(Y_, Y)

        # gradient calculation
        loss.backward()

        # update of gradients
        optimizer.step()

        # gradients to zero for next pass
        optimizer.zero_grad()

        print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')
        my_grad_a = 2 * torch.mean(diff * X)
        my_grad_b = 2 * torch.mean(diff)
        print(f"\tstep: {i}, a_calculated:{my_grad_a}, b_calculated:{my_grad_b}")
        print(f"\tstep: {i}, a_calculated:{a.grad}, b_calculated:{b.grad}")
