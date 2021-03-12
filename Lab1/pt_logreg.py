class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        # ...

    def forward(self, X):

    # unaprijedni prolaz modela: izračunati vjerojatnosti
    #   koristiti: torch.mm, torch.softmax
    # ...

    def get_loss(self, X, Yoh_):


# formulacija gubitka
#   koristiti: torch.log, torch.mean, torch.sum
# ...


def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC], type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    # ...

    # petlja učenja
    # ispisujte gubitak tijekom učenja
    # ...


def eval(model, X):
    """Arguments:
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()

if __name__ == "__main__":
  # inicijaliziraj generatore slučajnih brojeva
  np.random.seed(100)

  # instanciraj podatke X i labele Yoh_

  # definiraj model:
  ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

  # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
  train(ptlr, X, Yoh_, 1000, 0.5)

  # dohvati vjerojatnosti na skupu za učenje
  probs = eval(ptlr, X)

  # ispiši performansu (preciznost i odziv po razredima)

  # iscrtaj rezultate, decizijsku plohu