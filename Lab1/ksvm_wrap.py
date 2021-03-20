from sklearn import svm


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
