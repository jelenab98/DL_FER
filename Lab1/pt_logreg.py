import torch
import torch.nn as nn
import numpy as np
import data
import matplotlib.pyplot as plt


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


def train(model, X, Yoh_, param_niter=100, param_delta=0.1, param_lambda=0):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)
    for epoch in range(param_niter):
        probits = model.forward(X)

        loss = model.get_loss(probits, Yoh_) + param_lambda * torch.norm(model.w)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss))


def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    X = torch.Tensor(X)
    return model.forward(X).detach().cpu().numpy()


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    x, Y_ = data.sample_gauss_2d(3, 100)
    Yoh = data.class_to_onehot(Y_)
    X = torch.Tensor(x)
    Yoh_ = torch.Tensor(Yoh)
    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 1000, 0.5)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, recall = data.eval_perf_multi(Y, Y_)
    print("Accuracy: {}, precision: {}, recall: {}".format(accuracy, precision, recall))

    # iscrtaj rezultate, decizijsku plohu
    fun = decfun(ptlr, x)
    bbox = (np.min(x, axis=0), np.max(x, axis=0))

    data.graph_surface(fun, bbox, offset=0.5)
    data.graph_data(x, Y_, Y)

    plt.show()