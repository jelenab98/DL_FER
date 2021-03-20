import torch.nn as nn
import numpy as np
import torch


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        super().__init__()

        self.w = nn.Parameter(torch.randn(D, C), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(C), requires_grad=True)

    def forward(self, X):
        return torch.softmax(X.mm(self.w) + self.b, dim=1)

    @staticmethod
    def get_loss(X, Yoh_):
        return -torch.mean(torch.sum(Yoh_ * torch.log(X), dim=1))


def decfun(model, X):
    def classify(X):
        return np.argmax(model.forward(torch.Tensor(X)).detach().cpu().numpy(), axis=1)
    return classify


def train(model, X, Yoh_, param_niter=100, param_delta=0.1, param_lambda=1e-3, print_step=50):
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)
    for epoch in range(param_niter):
        probits = model.forward(X)

        loss = model.get_loss(probits, Yoh_) + param_lambda * torch.norm(model.w)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % print_step == 0:
            print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss))


def eval(model, X):
    X = torch.Tensor(X)
    return model.forward(X).detach().cpu().numpy()
