import numpy as np

rng = np.random.default_rng(42)

class Logistic_Regression():
    """
    Logistic regression with batch gradient descent as the optimizer
    """
    def __init__(self, eps=10e-5):
        self.eps = eps
        self.name = "Logistic Regression" # Will be useful when we create a decision boundary function

    def fit(self, X, Y, lr=0.01):
        """
        Run batch gradient descent until convergence

        Args:
        X: training example features. Shape (m, n)
        Y: training example labels. Shape (m,)
        lr: learning rate, typicaly 0.01
        """

        self.theta = rng.uniform(0, 1, size=X.shape[1]+1)

        # let's make X fit the shape of theta
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        def h(theta, X):
            """ Hypothesis function with the sigmoid

            Args:
            theta: parameters of our model. Shape (n+1,)
            X: set of examples. Shape (m,n)


            Returns:
            probability of a positive example for each example
            """

            return 1 / (1 + np.exp(-(X @ theta)))

        def gradient(theta, X, Y):
           """ Gradient of the loss function wrt theta

            Args:
            theta: parameters of our model. Shape (n+1,)
            X: training example features. Shape (m,n)
            Y: training example labels. Shape (m,)

            Returns:
            gradient of loss function
            """
           m = X.shape[0]
           return 1/m * X.T @ (h(theta, X) - Y)

        while True:
            # save old value of theta
            old_theta = np.copy(self.theta)

            # compute gradient
            grad = gradient(self.theta, X, Y)

            # update theta values
            self.theta -= lr*grad

            # compare with previous
            if np.linalg.norm((old_theta - self.theta), 1) <= self.eps:
                break


    def predict(self, X):
        """ Predicts the label given a new x

        Args:
        X: set of examples. Shape (m,n)

        Returns:
        vector of shape (m,) with predicted class"""
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((ones, X))

        return X @ self.theta >= 0
    