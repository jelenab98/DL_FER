from torchvision import datasets
from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np
import pt_deep
import torch
import data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist(dataset_root="./data/"):
    """
    Ucitavanje podataka.
    :param dataset_root:
    :return:
    """

    mnist_train = datasets.MNIST(dataset_root, train=True, download=False)
    mnist_test = datasets.MNIST(dataset_root, train=False, download=False)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    return x_train, y_train, x_test, y_test


def show_weights(weights):
    """
    Pomocna funkcija za prikaz naucenih matrica tezina
    :param weights:
    :return:
    """
    fig = plt.figure(figsize=(16, 8))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow((weights[:, i].detach().cpu().numpy()).reshape(28, 28))
    plt.show()


def show_loss(losses, labels):
    """
    Pomocna funkcija za prikaz kretanja modela kroz epohe ucenja
    :param losses:
    :param labels:
    :return:
    """
    fig = plt.figure(figsize=(16, 10))
    for loss, label in zip(losses, labels):
        plt.plot(range(len(loss)), loss, label=label)
    plt.xlabel("Epochs")
    plt.ylabel("Loss functions")
    plt.title("Loss function over the epochs")
    plt.legend()
    plt.show()


def train_weights_regularization(x_train, y_oh_train):
    """
    Utjecaj regularizacija na naucene tezine za pojedinu znamenku. S povecanjem regulariacije, matrice su zagladene,
    odnosno preciznije su i bolje su definirani oblici pojedine znamenke koja se detektira.
    :param x_train:
    :param y_oh_train:
    :return:
    """
    arh = [784, 10]
    lambdas = [0, 1e-3, 0.1, 0.9]
    x_train = x_train.reshape(len(x_train), 784)
    for param_lambda in lambdas:
        print("Regularizacijski koeficijent je ", param_lambda)
        model = pt_deep.PTDeep(arh, torch.relu, cuda=True).to(device)
        pt_deep.train(model, x_train, y_oh_train, 5000, 0.1, param_lambda, 500, printing=False, cuda=True)
        show_weights(model.weights[0])


def train_test_regularization(x_train, y_train, y_oh_train, x_test, y_test):
    """
    Utjecaj regularizacije na performanse modela. S povecanjem lambde se model pojednostavljuje i ne generalizira
    tako dobro. Za manje vrijednosti se dobije blaga poboljsanja za razliku od lambda=0.
    :param x_train:
    :param y_train:
    :param y_oh_train:
    :param x_test:
    :param y_test:
    :return:
    """
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    arh = [784, 100, 10]
    lambdas = [0, 1e-3, 0.1, 0.9]
    losses = []
    for param_lambda in lambdas:
        print("Model with architecture {} and lambda {}".format(arh, param_lambda))
        model = pt_deep.PTDeep(arh, torch.relu, cuda=True).to(device)
        losses.append(pt_deep.train(model, x_train, y_oh_train, 3001, 0.1, param_lambda, 1000, cuda=True))

        probs = pt_deep.eval(model, x_train, cuda=True)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_train)
        print("\nTrain| Accuracy: {}, precision_recall: {}".format(accuracy, precision))

        probs = pt_deep.eval(model, x_test, cuda=True)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_test)
        print("\nTest| Accuracy: {}, precision_recall: {}\n".format(accuracy, precision))
    return losses, lambdas


def train_early_stopping(x_train, y_oh_train, x_valid, y_valid, param_delta,
                         param_niter, param_lambda, print_step, save_path):
    """
    early stopping kako bi se sprječila pojava prenaučenosti i gubitka sposobnosti generalizacije. U pravilu se postiže
    bolja performansa budući da je vraćen model koji nije počeo gubiti na generalizaciji.
    :param x_train:
    :param y_oh_train:
    :param x_valid:
    :param y_valid:
    :param param_delta:
    :param param_niter:
    :param param_lambda:
    :param print_step:
    :param save_path:
    :return:
    """
    arh = [784, 100, 10]
    model = pt_deep.PTDeep(arh, torch.relu, cuda=True).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)
    losses = []
    valid_acc = 0
    best_epoch = -1
    best_model = None
    x_train = x_train.to(device)
    y_oh_train = y_oh_train.to(device)
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


def train_test_early_stopping(x_train, y_oh_train, y_train, x_train_smaller, y_oh_smaller, x_valid, y_valid, x_test,
                              y_oh_test, y_test, param_delta=0.1, param_niter=3001, param_lambda=1e-4, print_step=1000,
                              save_path="./saved_model.pth"):

    x_train = x_train.reshape(len(x_train), 784)
    x_valid = x_valid.reshape(len(x_valid), 784)
    x_train_smaller = x_train_smaller.reshape(len(x_train_smaller), 784)
    x_test = x_test.reshape(len(x_test), 784)

    losses, best_epoch, model = train_early_stopping(x_train_smaller, y_oh_smaller, x_valid, y_valid, param_delta,
                                                     param_niter, param_lambda, print_step, save_path)

    print("Best model is from epoch ", best_epoch)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)

    probits = model.forward(x_train.to(device))

    loss = model.get_loss(probits, y_oh_train.to(device)) + param_lambda * model.get_norm()
    loss.backward()

    optimizer.step()

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


def evaluate_init_model(x_train, y_train, y_oh_train, x_test, y_test, y_oh_test):
    """
    Evaluacija modela bez učenja. Točnost modela trebala bi biti 1/C % buduci da model za svaki primjer bira jednu od
    C klasa koju ce dodijeliti.
    :param x_train:
    :param y_train:
    :param y_oh_train:
    :param x_test:
    :param y_test:
    :param y_oh_test:
    :return:
    """
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


def train_adam(x_train, y_oh_train, y_train, x_test, y_test, y_oh_test, param_delta=1e-4, param_niter=3000,
               param_lambda=1e-4, print_step=1000, scheduler=False, param_gamma=1-1e-4):
    """
    Trening s Adamom uz ili bez koristenja lr schedulera koji postepeno smanjuje stopu ucenja.
    :param x_train:
    :param y_oh_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param y_oh_test:
    :param param_delta:
    :param param_niter:
    :param param_lambda:
    :param print_step:
    :param scheduler:
    :param param_gamma:
    :return:
    """
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    arh = [784, 100, 10]
    model = pt_deep.PTDeep(arh, torch.relu).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=param_delta)
    if scheduler:
        scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param_gamma)
    losses = []
    for epoch in range(param_niter):
        probits = model.forward(x_train.to(device))

        loss = model.get_loss(probits, y_oh_train.to(device)) + model.get_norm() * param_lambda
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        if epoch % print_step == 0:
            print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss))
        losses.append(loss)
        if scheduler:
            scheduler_lr.step()

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

    return losses


def train_multiple_architectures(x_train, y_train, y_oh_train, x_test, y_test):
    """
    Usporedba razlicitih modela i njihovih performansi. Dublji modeli su slozeniji, sporije se treniraju, ali daju
    bolje rezultate.
    :param x_train:
    :param y_train:
    :param y_oh_train:
    :param x_test:
    :param y_test:
    :return:
    """
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    architectures = ([784, 10], [784, 100, 10], [784, 100, 100, 10], [784, 100, 100, 100, 10])
    losses = []
    train_stats = []
    test_stats = []
    param_niter = [3001, 3001, 3001, 5001]
    param_delta = [0.1, 0.1, 0.1, 0.05]
    param_lambda = 1e-4
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


def svm2(x_train, y_train, x_test, y_test):
    """
    Usporedba SVM-a s dubokim modelima. Ocekujemo da RBF daje bolje rezultate. Problem s SVM-om je da treniranje
    može potrajati dosta dugo.
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    x_train, y_train = x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy()
    x_test, y_test = x_test.detach().cpu().numpy(), y_test.detach().cpu().numpy()
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    model = svm.SVC(kernel="linear", decision_function_shape="ovo").fit(x_train, y_train)
    accuracy, precision, M = data.eval_perf_multi(model.predict(x_test), y_test)
    print("Linear SVM | Accuracy: {}, precision_recall: {}".format(accuracy, precision))

    model = svm.SVC(kernel="rbf", decision_function_shape="ovo").fit(x_train, y_train)
    accuracy, precision, M = data.eval_perf_multi(model.predict(x_test), y_test)
    print("\nRBF SVM | Accuracy: {}, precision_recall: {}".format(accuracy, precision))


def train_mb(x_train, y_train, y_gt, x_test, y_oh_test, y_test, param_delta, param_lambda, print_step, param_niter):
    """
    SGD train loop, odnosno ucenje po mini grupama. Ne uci se na potpunom gradijentu nego na djelomicno izracunatom.
    Model bolje uci na taj nacin. Potrebno je manje epoha za postici bolje performanse, a mijenja se i raspon stope
    učenja. Iako, vrtilo se dosta sporije nego prije što se tiče jedne epohe.
    :param x_train:
    :param y_train:
    :param y_gt:
    :param x_test:
    :param y_oh_test:
    :param y_test:
    :param param_delta:
    :param param_lambda:
    :param print_step:
    :param param_niter:
    :return:
    """
    x_train = x_train.reshape(len(x_train), 784)
    x_test = x_test.reshape(len(x_test), 784)
    losses = []
    batch_sizes = (64, 256, 1000)

    for batch_size in batch_sizes:
        model = pt_deep.PTDeep([784, 100, 10], torch.relu).to(device)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=param_delta)
        tmp_losses = []
        print("Starting with the batch size of ", batch_size)
        for epoch in range(param_niter):
            permuted_indices = torch.randperm(len(x_train))
            x_train_permuted = x_train[permuted_indices]
            y_train_permuted = y_train[permuted_indices]

            x_train_in_batches = [x_i for x_i in torch.split(x_train_permuted, batch_size)]
            y_train_in_batches = [y_i for y_i in torch.split(y_train_permuted, batch_size)]

            loss_log = 0
            for i, (x, y) in enumerate(zip(x_train_in_batches, y_train_in_batches)):
                probits = model.forward(x.to(device))

                loss = model.get_loss(probits, y.to(device)) + param_lambda * model.get_norm()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                loss_log += loss
                if i % 100 == 0 and epoch % print_step == 0:
                    print("Step {}, loss: {}".format(i, loss))
            loss_log /= len(x_train_in_batches)
            if epoch % print_step == 0:
                print("Epoch {}/{}, loss = {}".format(epoch, param_niter, loss_log))
            tmp_losses.append(loss_log)

        probits = model.forward(x_train.to(device))
        loss = model.get_loss(probits, y_train.to(device))
        probs = pt_deep.eval(model, x_train)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_gt)
        print("Train | Loss: {}, accuracy: {:.2f}%".format(loss, 100 * accuracy))

        probits = model.forward(x_test.to(device))
        loss = model.get_loss(probits, y_oh_test.to(device))
        probs = pt_deep.eval(model, x_test)
        Y = np.argmax(probs, axis=1)
        accuracy, precision, M = data.eval_perf_multi(Y, y_test)
        print("Test  | Loss: {}, accuracy: {:.2f}%".format(loss, 100 * accuracy))

        losses.append(tmp_losses)
    return losses, batch_sizes
