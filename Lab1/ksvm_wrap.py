from sklearn import svm


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma="auto", param_svm_kernel="linear"):
        # inicijalizacija i fittanje
        self.svm = svm.SVC(C=param_svm_c, kernel=param_svm_kernel,
                           gamma=param_svm_gamma).fit(X, Y_)

    def predict(self, X):
        """
        Predikcija na temelju danih primjera
        :param X:
        :return:
        """
        return self.svm.predict(X)

    def get_scores(self, X):
        """
        Klasifikacija na temelju danih primjera
        :param X:
        :return:
        """
        return self.svm.decision_function(X)

    def support(self):
        """
        Support vectors koji su vazni za klasifikaciju.
        :return:
        """
        return self.svm.support_
