import numpy as np


def reLu(x):
    """
    Pomocna funkcija za izracunavanje zglobnice.
    :param x: tenzor x
    :return:
    """
    return np.maximum(0, x)


def softmax(x):
    """
    Izracun softmaxa kao e^x/(suma e^x). Na taj nacin spoljustuje izlaz na interval 0-1.
    :param x:
    :return:
    """
    x_exp = np.exp(x)
    return x_exp/(np.sum(x_exp, axis=1))[:, np.newaxis]


def cross_entropy_loss(probits, Y_):
    """
    Funkcija gubitka unakrsne entropije dobivena po formuli -1/N * suma(log(P(Y=yi|xi)))
    :param probits: probiti dobivei iz softmaxa
    :param Y_: one hot encoded oznake kako se ne bi trebalo prebirati indekse iz x
    :return:
    """
    N = probits.shape[0]
    return np.mean(np.sum(-np.log(probits[range(N), Y_] + 1e-13)))


def fcann2_train(X, Y_, hidden_neurons=5, param_niter=1000, param_delta=0.01, param_lambda=0.05, print_step=100):
    """
    Train funkcija za model s jednim skrivenim slojem
    :param X:
    :param Y_:
    :param hidden_neurons:
    :param param_niter:
    :param param_delta:
    :param param_lambda:
    :param print_step:
    :return:
    """
    N, d = X.shape
    C = max(Y_) + 1

    # weights initialization (Xavier distribution for weights, zeros for biases)
    w_1 = np.random.normal(loc=0, scale=1/np.mean([d, hidden_neurons]), size=(d, hidden_neurons))
    b_1 = np.zeros((1, hidden_neurons))
    w_2 = np.random.normal(loc=0, scale=1/np.mean([hidden_neurons, C]), size=(hidden_neurons, C))
    b_2 = np.zeros((1, C))

    for epoch in range(param_niter):

        # forward pass
        s1 = np.dot(X, w_1) + b_1  # s1 = W1*x + b1
        h1 = reLu(s1)  # h1 = reLu(s1)
        s2 = np.dot(h1, w_2) + b_2  # s2 = W2*h1 + b2
        s2 -= s2.max()  # zastita od overflowa
        probits = softmax(s2) # P(Y|x) = softmax(s2)

        predicted_classes = np.argmax(probits, axis=1)

        # Cross Entropy loss
        loss = cross_entropy_loss(probits, Y_) + param_lambda * (np.linalg.norm(w_1) + np.linalg.norm(w_2))

        if epoch % print_step == 0:
            print("Epoch {}/{}, loss: {}".format(epoch, param_niter, loss))

        # calculation of gradients
        grads_w_2_tmp = probits
        grads_w_2_tmp[range(N), Y_] -= 1    # predicted classes - y_gt
        grads_w_2_tmp /= N
        grads_b_2 = np.sum(grads_w_2_tmp, axis=0)  # Pij - Yij
        grads_w_2 = np.dot(h1.T, grads_w_2_tmp)  # (Pij - Yij) * h1.T

        grads_w_1_tmp = np.dot(grads_w_2_tmp, w_2.T)  # (P-Y)*W
        grads_w_1_tmp[s1 <= 0.0] = 0.0  # prehodno * diagonalna[s1 >0]
        grads_w_1 = np.dot(X.T, grads_w_1_tmp)
        grads_b_1 = np.sum(grads_w_1_tmp, axis=0)  # dL/ds1

        # updating the weights

        w_1 -= param_delta * grads_w_1
        b_1 -= param_delta * grads_b_1

        w_2 -= param_delta * grads_w_2
        b_2 -= param_delta * grads_b_2

    return w_1, b_1, w_2, b_2


def fcann2_classify(X, w_1, b_1, w_2, b_2):
    """
    Klasifikacija primjera na temelju najvece vrijednosti softmaxa.
    :param X:
    :param w_1:
    :param b_1:
    :param w_2:
    :param b_2:
    :return:
    """
    h1 = reLu(np.dot(X, w_1) + b_1)
    s2 = np.dot(h1, w_2) + b_2
    probits = softmax(s2)
    predicted_classes = np.argmax(probits, axis=1)
    return predicted_classes


def decfun(X, w1, b1, w2, b2):
    def classify(X):
        return fcann2_classify(X, w1, b1, w2, b2)
    return classify
