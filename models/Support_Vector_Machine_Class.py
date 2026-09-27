import numpy as np

rng = np.random.default_rng(42)

class SVM():
    """ Support vector machine class with gaussian/RBF kernel and a simplified smo algorithm
    """
    def __init__(self, sigma=1):
      """
      When creating the model, set self.sigma=sigma

      Args:
      sigma: standard deviation, used for gaussian kernel
      """

      self.sigma=sigma
      self.name = "Support Vector Machine" # will be useful when we create a decision boundary function

    def kernel(self, X, x1):
        """
        Vectorized Gaussian/RBF kernel function

        Args:
        X: training examples features. Shape (m, n)
        x1: vector that the kernel is taken wrt or matrix of new features. Shape (m1, n)

        Returns:
        If m > 1; vector of values of kernel function between each example in training set and x1. Shape (m,)
        If m = 1; value of gaussian kernel between two vectors
        If m1 > 1; matrix of vector values of kernel function between each example in training set and each new feature vector. Shape (m, m1)
        """

        if len(X.shape)==1:
          m = X.shape[0]
          n=0
        else:
          m, n = X.shape

        if len(x1.shape)==1:
          m1 = x1.shape[0]
        else:
          m1, n = x1.shape


        if len(x1.shape) == 1:
            if n > 1:
                Z = X - x1.reshape(1,n)
                return np.exp(-(np.sum(Z**2, axis=1) / (2*self.sigma**2)))
            else:
                return np.exp(-((X - x1).dot(X - x1) / (2*self.sigma**2)))
        else:
            X = X.reshape(1, m, n)
            x1 = x1.reshape(m1, 1, n)
            Z = X - x1
            return np.exp(-((np.sum(Z**2, axis=2)).T / (2*self.sigma**2)))

    def f(self, X, Y, x1):
        """
        Decision function

        Args:
        X: training examples features. Shape (m, n)
        Y: training examples labels. Shape (m,)
        x1: new vector or matrix of new features. Shape (m1, n)

        Returns:
        If m1 = 1; decision function score
        If m1 > 1; decision function score for each m1 feature vectors"""

        m, n = X.shape

        # check if alpha and b have been created
        if self.alpha.any() == None:
            self.alpha = np.zeros(m)
        if self.b == None:
            self.b = 0


        return (self.alpha * Y) @ (self.kernel(X, x1)) + self.b

    def E(self, X, Y, j):
        """
        Error between prediction and true label

        Args:
        X: training examples features. Shape (m, n)
        Y: training examples labels. Shape (m,)
        j: the vector we are evaluating f to lives within X, this is that index

        Returns:
        error, distance between prediction and true label"""


        return self.f(X, Y, X[j]) - Y[j]

    def eta(self, x1, x2):
        """
        eta, second derivative of objective function along the linear constraint. Will be used to update alpha_j

        Args:
        X: training examples feaatures. Shape (m, n)
        x1: vector corresponding to ith entry of X. Shape (n,)
        x2: vector corresponding to jth entry of X. Shape (n,)

        Returns:
        eta, used to update alpha_j
        """

        return 2 * self.kernel(x1, x2) - self.kernel(x1, x1) - self.kernel(x2, x2)

    def fit(self, X, Y, C=1.0, tol=1e-3, max_passes=20):
        """ fits the models parameters (alpha's and b) to train the svm

        Args:
        X: training examples features. Shape (m, n)
        Y: training examples labels. Shape (m,)
        C: regularization parameter
        tol: level of accuracy for kkt conditions to be satisfied
        max_passes: how many passes without the alpha's updating before we move on
        """

        m, n = np.shape(X)

        # let's change Y so that it fits our convention of y_i is in {-1, 1}
        Y[Y == 0] = -1

        self.alpha = np.zeros(m)
        self.b = 0
        passes = 0
        exit_while = False

        while(passes < max_passes):
            num_changed_alphas = 0
            for i in range(m):
                E_i = self.E(X, Y, i)
                if (((Y[i] * E_i) < -tol and self.alpha[i] < C) or ((Y[i] * E_i) > tol and self.alpha[i] > 0)):
                    index_list = np.delete(np.arange(len(self.alpha)), i)
                    j = rng.choice(index_list)
                    E_j = self.E(X, Y, j)
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    if Y[i] == Y[j]:
                        L = max(0, self.alpha[i] + self.alpha[j] - C)
                        H = min(C, self.alpha[i] + self.alpha[j])
                    else:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(C, C + self.alpha[j] - self.alpha[i])
                    if L == H:
                        continue
                    eta = self.eta(X[i], X[j])
                    if eta >= 0:
                        continue
                    self.alpha[j] -= (Y[j]*(E_i - E_j))/eta
                    unconstrained_alpha_j = self.alpha[j]
                    if self.alpha[j] > H:
                        self.alpha[j] = H
                    elif self.alpha[j] < L:
                        self.alpha[j] = L
                    if np.abs(self.alpha[j] - alpha_j_old) < 10e-5:
                        continue
                    self.alpha[i] += Y[i] * Y[j] * (alpha_j_old - self.alpha[j])
                    b_1 = self.b - E_i - Y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i], X[i]) - Y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[i], X[j])
                    b_2 = self.b - E_j - Y[i] * (self.alpha[i] - alpha_i_old) * self.kernel(X[i], X[j]) - Y[j] * (self.alpha[j] - alpha_j_old) * self.kernel(X[j], X[j])
                    if 0 < self.alpha[i] < C:
                        self.b = b_1
                    elif 0 < self.alpha[j] < C:
                        self.b = b_2
                    else:
                        self.b = (b_1 + b_2)/2
                    num_changed_alphas += 1
            if exit_while==True:
              break
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0


    def predict(self, X, Y, x1):
        """
        prediction function for new examples

        Args:
        X: training examples features. Shape (m, n)
        Y: training examples labels. Shape (m,)
        x1: new features. Shape (m1, n)

        Returns:
        vector of shape (m,) with predicted class"""

        # let's make sure that Y matches our previous convention
        Y[Y==0] = -1

        return self.f(X, Y, x1) >= 0
