import numpy as np


def reLu(x):
    return np.max(0, x)


def softmax(x):
    x_exp = np.exp(x)
    return x_exp/np.sum(x_exp, axis=1)


def cross_entropy_loss(probits, Y_):
    N = probits.shape[0]
    return np.mean(np.sum(-np.log(probits[range(N), Y_])))


def fcann2_train(X, Y_, hidden_neurons=5, param_niter=1000, param_delta=0.01, param_lambda=0.05):
    N, d = X.shape
    _, C = Y_.shape

    # weights initialization (Xavier distribution for weights, zeros for biases)
    w_1, b_1 = np.random.uniform((d, hidden_neurons)), np.zeros((1, d))
    w_2, b_2 = np.random.uniform((hidden_neurons, C)), np.zeros((1, C))

    for epoch in range(param_niter):

        # forward pass
        h1 = np.max(0, np.dot(X, w_1) + b_1)
        s2 = np.dot(h1, w_2) + b_2
        probits = softmax(s2)
        predicted_classes = np.argmax(probits, axis=1)

        # Cross Entropy loss
        loss = cross_entropy_loss(probits, Y_)

        if epoch % 100 == 99:
            print("Epoch {}/{}, loss: {}".format(epoch, param_niter, loss))

        # calculation of gradients
        grads_w_2 = (predicted_classes - Y_) * h1.T
        grads_b_2 = predicted_classes - Y_

        grads_w_1 = 0
        grads_b_1 = 0

        # updating the weights

        w_1 -= param_delta * grads_w_1 + param_lambda * np.linalg.norm(w_1)
        b_1 -= param_delta * grads_b_1 + param_lambda * np.linalg.norm(b_1)

        w_2 -= param_delta * grads_w_2 + param_lambda * np.linalg.norm(w_2)
        b_2 -= param_delta * grads_b_2 + param_lambda * np.linalg.norm(b_2)

    return w_1, b_1, w_2, b_2


def fcann2_classify(X, w_1, b_1, w_2, b_2):
    h1 = np.max(0, np.dot(X, w_1) + b_1)
    s2 = np.dot(h1, w_2) + b_2
    probits = softmax(s2)
    predicted_classes = np.argmax(probits, axis=1)
    return predicted_classes
