from sklearn import svm
import numpy as np
import data
import matplotlib.pyplot as plt


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma="auto", param_svm_kernel="linear"):
        self.svm = svm.SVC(C=param_svm_c, kernel=param_svm_kernel,
                           gamma=param_svm_gamma).fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        return self.svm.decision_function(X)

    def support(self):
        return self.svm.support_


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(6, 2, 10)
    # definiraj model:
    ptlr = KSVMWrap(X, Y_, 1, "auto", "rbf")

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    predictions = ptlr.predict(X)
    scores = ptlr.get_scores(X)
    support_v = ptlr.support()


    # iscrtaj rezultate, decizijsku plohu
    bbox = (np.min(X, axis=0), np.max(X, axis=0))

    data.graph_surface(ptlr.predict, bbox, offset=0.5)
    data.graph_data(X, Y_, predictions)

    plt.show()
