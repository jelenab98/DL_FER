import numpy as np
import data
import matplotlib.pyplot as plt

# TODO popraviti gradijente za fcann u treningu i testirati sve


def reLu(x):
    return np.maximum(0, x)


def softmax(x):
    x_exp = np.exp(x)
    return x_exp/(np.sum(x_exp, axis=1))[:, np.newaxis]


def cross_entropy_loss(probits, Y_):
    N = probits.shape[0]
    return np.mean(np.sum(-np.log(probits[range(N), Y_])))


def fcann2_train(X, Y_, hidden_neurons=5, param_niter=1000, param_delta=0.01, param_lambda=0.05):
    N, d = X.shape
    C = max(Y_) + 1

    # weights initialization (Xavier distribution for weights, zeros for biases)
    w_1 = np.random.normal(loc=0, scale=1/np.mean([d, hidden_neurons]), size=(d, hidden_neurons))
    b_1 = np.zeros((1, hidden_neurons))
    w_2 = np.random.normal(loc=0, scale=1/np.mean([hidden_neurons, C]), size=(hidden_neurons, C))
    b_2 = np.zeros((1, C))


    for epoch in range(param_niter):

        # forward pass
        h1 = reLu(np.matmul(X, w_1) + b_1)
        s2 = np.matmul(h1, w_2) + b_2
        probits = softmax(s2)
        predicted_classes = np.argmax(probits, axis=1)

        # Cross Entropy loss
        loss = cross_entropy_loss(probits, Y_)

        if epoch % 100 == 99:
            print("Epoch {}/{}, loss: {}".format(epoch, param_niter, loss))

        # calculation of gradients
        grads_w_2 = (predicted_classes - Y_) * h1.T
        grads_b_2 = predicted_classes - Y_

        grads_w_1 = np.matmul(X.T, np.dot(np.dot((predicted_classes - Y_), w_2), ))
        grads_b_1 = 0

        # updating the weights

        w_1 -= param_delta * grads_w_1 + param_lambda * np.linalg.norm(w_1)
        b_1 -= param_delta * grads_b_1

        w_2 -= param_delta * grads_w_2 + param_lambda * np.linalg.norm(w_2)
        b_2 -= param_delta * grads_b_2

    return w_1, b_1, w_2, b_2


def fcann2_classify(X, w_1, b_1, w_2, b_2):
    h1 = np.max(0, np.dot(X, w_1) + b_1)
    s2 = np.dot(h1, w_2) + b_2
    probits = softmax(s2)
    predicted_classes = np.argmax(probits, axis=1)
    return predicted_classes


def decfun(X, w1, b1, w2, b2):
    def classify(X):
        return fcann2_classify(X, w1, b1, w2, b2)
    return classify


if __name__ == '__main__':
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    w1, b1, w2, b2 = fcann2_train(X, Y_, 5, 100000, 0.05, 1e-3)
    Y = fcann2_classify(X, w1, b1, w2, b2)

    fun = decfun(X, w1, b1, w2, b2)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))

    data.graph_surface(fun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)

    plt.show()