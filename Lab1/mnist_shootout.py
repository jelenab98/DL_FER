import torch
import torchvision
import numpy as np
import pt_deep
import matplotlib.pyplot as plt
import data

from torchvision import datasets
from sklearn import svm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist(dataset_root="./data/"):

    mnist_train = datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = datasets.MNIST(dataset_root, train=False, download=False)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    return x_train, y_train, x_test, y_test


def show_first_16_images(x, y):
    fig = plt.figure(figsize=(16, 16))
    for i in range(1, 17):
        plt.subplot(4, 4, i)
        plt.imshow(x[i - 1])
        plt.title(y[i - 1].detach().numpy())
    plt.show()


def show_weights(weights):
    fig = plt.figure(figsize=(16, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow((weights[:, i].detach().cpu().numpy()).reshape(28, 28))
    plt.show()


def show_loss(losses, labels):
    fig = plt.figure(figsize=(16, 10))
    for loss, label in zip(losses, labels):
        plt.plot(range(len(loss)), loss, label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss functions")
    plt.title("Loss function over the epochs")
    plt.legend()
    plt.show()


def train_weights_regularization(x_train, y_oh_train):
    arh = [784, 10]
    lambdas = [0, 1e-3, 0.1, 1]
    x_train = x_train.reshape(len(x_train), 784)
    for param_lambda in lambdas:
        print("Regularizacijski koeficijent je ", param_lambda)
        model = pt_deep.PTDeep(arh, torch.relu).to(device)
        pt_deep.train(model, x_train, y_oh_train, 5000, 0.1, param_lambda, 500, printing=False)
        show_weights(model.weights[0])


def train_test_regularization(x_train, y_train, y_oh_train, x_test, y_test, y_oh_test):
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    arh = [784, 100, 100, 10]
    lambdas = [0, 1e-3, 0.1, 1]
    losses = []
    for param_lambda in lambdas:
        print("Model with architecture {} and lambda {}".format(arh, param_lambda))
        model = pt_deep.PTDeep(arh, torch.relu).to(device)
        losses.append(pt_deep.train(model, x_train, y_oh_train, 3001, 0.1, param_lambda, 1000))

        probs = pt_deep.eval(model, x_train)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_train)
        print("Train| Accuracy: {}, precision_recall: {}".format(accuracy, precision))

        probs = pt_deep.eval(model, x_test)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_test)
        print("Test| Accuracy: {}, precision_recall: {}\n".format(accuracy, precision))
    return losses, lambdas


def train_early_stopping(x_train, y_oh_train, x_valid, y_valid, param_delta,
                         param_niter, param_lambda, print_step, save_path):
    x_train = x_train.reshape(len(x_train), 784)
    x_valid = x_valid.reshape(len(x_valid), 784)
    arh = [784, 100, 100, 10]
    model = pt_deep.PTDeep(arh, torch.relu).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)
    losses = []
    valid_acc = 0
    best_epoch = -1
    best_model = None
    for epoch in range(param_niter):
        probits = model.forward(x_train)

        loss = model.get_loss(probits, y_oh_train) + param_lambda * model.get_norm()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if epoch % print_step == 0:
            print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss))
        losses.append(loss)

        probs = pt_deep.eval(model, x_valid)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_valid)
        if accuracy > valid_acc:
            best_epoch = epoch
            best_model = model
            torch.save(model, save_path)

    return losses, best_epoch, best_model


def evaluate_init_model(x_train, y_train, y_oh_train, x_test, y_test, y_oh_test):
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)

    model = pt_deep.PTDeep([784, 100, 100, 10], torch.relu).to(device)

    probits = model.forward(x_train.to(device))
    loss = model.get_loss(probits, y_oh_train.to(device))
    probs = pt_deep.eval(model, x_train)
    Y = np.argmax(probs, axis=1)
    accuracy, precision, M = data.eval_perf_multi(Y, y_train)
    print("Train | Loss: {}, accuracy: {:.2f}%".format(loss, 100 * accuracy))

    probits = model.forward(x_test.to(device))
    loss = model.get_loss(probits, y_oh_test.to(device))
    probs = pt_deep.eval(model, x_test)
    Y = np.argmax(probs, axis=1)
    accuracy, precision, M = data.eval_perf_multi(Y, y_test)
    print("Test  | Loss: {}, accuracy: {:.2f}%".format(loss, 100 * accuracy))


def train_adam(x_train, y_oh_train, param_delta=1e-4, param_niter=3000, param_lambda=1e-4, print_step=1000,
               scheduler=False, param_betas=(0.9, 0.999), param_gamma=1-1e-4):
    x_train = x_train.reshape(len(x_train), 784)
    arh = [784, 100, 100, 10]
    model = pt_deep.PTDeep(arh, torch.relu).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=param_delta,
                                 weight_decay=param_lambda, betas=param_betas)
    if scheduler:
        scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param_gamma)
    losses = []
    for epoch in range(param_niter):
        probits = model.forward(x_train)

        loss = model.get_loss(probits, y_oh_train)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if epoch % print_step == 0:
            print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss))
        losses.append(loss)
        if scheduler:
            scheduler_lr.step()

    return losses


def train_multiple_architectures(x_train, y_train, y_oh_train, x_test, y_test):
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    architectures = ([784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10])
    losses = []
    train_stats = []
    test_stats = []
    param_niter = [3000, 3000, 3000, 5000]
    param_delta = [0.1, 0.1, 0.1, 0.05]
    param_lambda = 1e-2
    print_step = 1000
    for arh, epochs, lr in zip(architectures, param_niter, param_delta):
        print("Starting with architecture: ", arh)
        model = pt_deep.PTDeep(arh, torch.relu).to(device)
        losses.append(pt_deep.train(model, x_train, y_oh_train, epochs, lr, param_lambda, print_step))

        probs = pt_deep.eval(model, x_train)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_train)
        print("\nTrain| Accuracy: {}, precision_recall: {}".format(accuracy, precision))
        train_stats.append((accuracy, precision))

        probs = pt_deep.eval(model, x_test)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_test)
        print("\nTest| Accuracy: {}, precision_recall: {}\n".format(accuracy, precision))
        test_stats.append((accuracy, precision))
    return architectures, losses, train_stats, test_stats


def svm(x_train, y_train, x_test, y_test):
    x_train, y_train = x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()
    x_test, y_test = x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy()
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    model = svm.SVC(kernel="linear", decision_function_shape="ovo").fit(x_train, y_train)
    accuracy, precision, M = data.eval_perf_multi(model.predict(x_test), y_test)
    print("Linear SVM | Accuracy: {}, precision_recall: {}".format(accuracy, precision))

    model = svm.SVC(kernel="rbf", decision_function_shape="ovo").fit(x_train, y_train)
    accuracy, precision, M = data.eval_perf_multi(model.predict(x_test), y_test)
    print("Radial SVM | Accuracy: {}, precision_recall: {}".format(accuracy, precision))


def train_mb(model, x_train, y_train, param_delta, param_lambda, print_step, param_niter):
    x_train = x_train.reshape(len(x_train), 784)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta, weight_decay=param_lambda)
    losses = []
    for epoch in range(param_niter):

        probits = model.forward(x_train)

        loss = model.get_loss(probits, y_train)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if epoch % print_step == 0:
            print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss))
        losses.append(loss)
    return losses
