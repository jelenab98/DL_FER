import torch
import torchvision
import numpy as np
import pt_deep


def load_mnist(dataset_root="./data/"):
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    x_train, y_train, x_test, y_test = load_mnist()
    # definiraj model:
    ptlr = pt_deep.PTDeep([784, 10], torch.relu)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    pt_deep.train(ptlr, x_train, y_train, 1000, 0.1, 1e-4)

    # dohvati vjerojatnosti na skupu za učenje
    probs = pt_deep.eval(ptlr, x_train)
    Y = np.argmax(probs, axis=1)



