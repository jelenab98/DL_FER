import torch
import torch.nn as nn
import numpy as np
import data
import matplotlib.pyplot as plt


class PTDeep(nn.Module):
    def __init__(self, neurons, activation):
        super().__init__()

        weights, biases = [], []
        for idx in range(len(neurons) - 1):
            weights.append(nn.Parameter(nn.init.xavier_normal_(torch.zeros((neurons[idx], neurons[idx + 1]),
                                                                           dtype=torch.float)), requires_grad=True))
            biases.append(nn.Parameter(torch.zeros((1, neurons[idx + 1]), dtype=torch.float), requires_grad=True))

        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)
        self.activation = activation

    def forward(self, X):
        s = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            s = self.activation(s.mm(w) + b)
        return torch.softmax(s.mm(self.weights[-1]) + self.biases[-1], dim=1)

    @staticmethod
    def get_loss(X, Yoh_):
        return -torch.mean(torch.sum(Yoh_ * torch.log(X+1e-13), dim=1))

    def get_norm(self):
        norm = 0
        for weights in self.weights:
            norm += torch.norm(weights)
        return norm


def decfun(model, X):
    def classify(X):
        return np.argmax(model.forward(torch.Tensor(X)).detach().cpu().numpy(), axis=1)
    return classify


def train(model, X, Yoh_, param_niter=100, param_delta=0.1, param_lambda=1e-4):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)
    for epoch in range(param_niter):
        probits = model.forward(X)

        loss = model.get_loss(probits, Yoh_) + param_lambda * model.get_norm()
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
    x, Y_ = data.sample_gmm_2d(6, 2, 10)
    Yoh = data.class_to_onehot(Y_)
    X = torch.Tensor(x)
    Yoh_ = torch.Tensor(Yoh)
    # definiraj model:
    ptlr = PTDeep([2, 10, 10, 2], torch.relu)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 6400, 0.1, 1e-4)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, M = data.eval_perf_multi(Y, Y_)
    #avg_precision = data.eval_AP(Y_[probs.argsort()])
    avg_precision = 0
    print("Accuracy: {}, precision: {}, average precision: {}".format(accuracy, precision,
                                                                      avg_precision))

    # iscrtaj rezultate, decizijsku plohu
    fun = decfun(ptlr, x)
    bbox = (np.min(x, axis=0), np.max(x, axis=0))

    data.graph_surface(fun, bbox, offset=0.5)
    data.graph_data(x, Y_, Y)

    plt.show()