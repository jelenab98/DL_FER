import torch.nn as nn
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PTDeep(nn.Module):
    def __init__(self, neurons, activation, cuda=True):
        super().__init__()

        weights, biases = [], []

        # Xavier za težine, nula za biase
        for idx in range(len(neurons) - 1):
            if cuda:
                weights.append(nn.Parameter(nn.init.xavier_normal_(torch.zeros((neurons[idx], neurons[idx + 1]),
                                                                               dtype=torch.float).to(device)),
                                            requires_grad=True))
                biases.append(nn.Parameter(torch.zeros((1, neurons[idx + 1]), dtype=torch.float).to(device),
                                           requires_grad=True, ))
            else:
                weights.append(nn.Parameter(nn.init.xavier_normal_(torch.zeros((neurons[idx], neurons[idx + 1]),
                                                                               dtype=torch.float)), requires_grad=True))
                biases.append(nn.Parameter(torch.zeros((1, neurons[idx + 1]), dtype=torch.float), requires_grad=True))

        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases)
        self.activation = activation

    def forward(self, X):
        """
        Unaprijedni prolaz. Iterativno se radi W*s + b i to se propusta kroz aktivaciju i finalno kroz softmax
        :param X:
        :return:
        """
        s = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            s = self.activation(s.mm(w) + b)
        return torch.softmax(s.mm(self.weights[-1]) + self.biases[-1], dim=1)

    @staticmethod
    def get_loss(X, Yoh_):
        """
        Unakrsna entropija pomocu one hot encoded oznaka i primjera.
        :param X:
        :param Yoh_:
        :return:
        """
        return -torch.mean(torch.sum(Yoh_ * torch.log(X+1e-13), dim=1))

    def count_params(self):
        """
        Pomocna metoda za ispis slojeva, velicina tenzora i broja parametara
        :return:
        """
        tensor_shapes = [(p[0], p[1][0].shape) for p in self.named_parameters()]
        total_parameters = np.sum([p.numel() for p in self.parameters()])
        return tensor_shapes, total_parameters

    def get_norm(self):
        """
        Pomocna metoda koja vraca sumu normi svih tezina.
        :return:
        """
        norm = 0

        for weights in self.weights:
            norm += torch.norm(weights)

        return norm


def decfun(model, X):
    def classify(X):
        return np.argmax(model.forward(torch.Tensor(X).to(device)).detach().cpu().numpy(), axis=1)
    return classify


def train(model, X, Yoh_, param_niter=100, param_delta=0.1, param_lambda=1e-4, print_step=100, printing=True,
          cuda=True):
    if cuda:
        X = X.to(device)
        Yoh_ = Yoh_.to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)
    losses = []

    for epoch in range(param_niter):
        probits = model.forward(X)

        loss = model.get_loss(probits, Yoh_) + param_lambda * model.get_norm()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % print_step == 0 and printing:
            print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss))

        losses.append(loss)

    return losses


def eval(model, X, cuda=True):
    X = torch.Tensor(X)
    if cuda:
        X = X.to(device)
    return model.forward(X).detach().cpu().numpy()
