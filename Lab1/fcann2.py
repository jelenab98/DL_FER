import numpy as np


def reLu(x):
    return np.maximum(0, x)


def softmax(x):
    x_exp = np.exp(x)
    return x_exp/(np.sum(x_exp, axis=1))[:, np.newaxis]


def cross_entropy_loss(probits, Y_):
    N = probits.shape[0]
    return np.mean(np.sum(-np.log(probits[range(N), Y_])))


def fcann2_train(X, Y_, hidden_neurons=5, param_niter=1000, param_delta=0.01, param_lambda=0.05, print_step=100):
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

        if epoch % print_step == 0:
            print("Epoch {}/{}, loss: {}".format(epoch, param_niter, loss))

        # calculation of gradients
        grads_w_2_tmp = probits
        grads_w_2_tmp[range(N), Y_] -= 1    # predicted classes - y_gt
        grads_w_2_tmp /= N
        grads_b_2 = np.sum(grads_w_2_tmp, axis=0)
        grads_w_2 = np.matmul(h1.T, grads_w_2_tmp)

        grads_w_1_tmp = np.matmul(grads_w_2_tmp, w_2.T)
        grads_w_1_tmp[h1 <= 0.0] = 0.0
        grads_w_1 = np.mean(np.matmul(X.T, grads_w_1_tmp), axis=0)
        grads_b_1 = np.sum(grads_w_1_tmp, axis=0)

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
