import numpy as np

rng = np.random.default_rng(42)

class NeuralNetwork():
    def __init__(self):
        """
        initializes neural network, wights using Xavier Uniform Initialization and biases to 0
        """
        self.W1 = rng.uniform(-np.sqrt(6/5), np.sqrt(6/5), size=(3, 2))
        self.b1 = np.zeros((3, 1))
        self.W2 = rng.uniform(-np.sqrt(6/5), np.sqrt(6/5), size=(2, 3))
        self.b2 = np.zeros((2, 1))
        self.W3 = rng.uniform(-np.sqrt(6/3), np.sqrt(6/3), size=(1, 2))
        self.b3 = np.zeros((1, 1))

        self.name = "Neural Network" # Will be useful when we create a decision boundary function

    def sigmoid(self, X):
        """
        sigmoid activation function

        Args:
        X: matrix of values. Shape (n, m)

        Returns:
        sigmoid function applied to each value. Shape (n, m)
        """

        return 1 / (1 + np.exp(-X))

    def forward(self, X):
        """
        forward pass

        Args:
        X: set of feature vectors. Shape (n, m)

        Returns:
        a3: probability of positive label. Shape (1, m)
        a2: sigmoid of layer two. Shape (2, m)
        a1: sigmoid of layer 1. Shape (3, m)
        """

        z1 = self.W1 @ X + self.b1
        a1 = self.sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = self.sigmoid(z2)
        z3 = self.W3 @ a2 + self.b3
        a3 = self.sigmoid(z3)

        return a3, a2, a1

    def loss(self, X, Y):
        """
        Binary cross-entropy loss

        Args:
        X: set of feature vectors. Shape (n, m)
        Y: set of labels for those feature vectors. Shape (m,)

        Returns:
        loss of model evaluated at current parameters
        """
        m = X.shape[1]
        a3 = self.forward(X)[0]
        return -1/m * np.sum(Y*np.log(a3) + (1-Y)* np.log(1-a3))

    def predict(self, X):
        """
        Predict the labels for a set of feature vectors

        Args:
        X: set of feature vectors. Shape (n, m)

        Returns:
        vector of labels. Shape (m,)"""
        m = X.shape[1]

        return self.forward(X)[0].reshape(m) >= 0.5

    def acc(self, X, Y):
        """
        accuracy of model on set X with labels Y

        Args:
        X: set of feature vectors. Shape (n, m)
        Y: set of labels. Shape (m,)

        Returns:
        accuracy of models predictions
        """
        a3 = self.forward(X)[0].reshape(X.shape[1])

        return np.mean(self.predict(X) == Y)


    def fit(self, X, Y, epochs, lr = 0.01, update_lr=False):
        """
        batch gradient descent optimizer

        Args:
        X: training examples features. Shape (n, m)
        Y: training examples labels. Shape (m,)
        epochs: number of loops to go through
        lr: learning rate
        gather_info: whether you want to recieve accuracy and loss per epoch or not

        Returns:
        nothing, unless gather_info is true, in that case, returns accuracy per epoch and loss per epoch in a list
        """
        m = X.shape[1]
        acc_list = [0] * epochs
        loss_list = [0] * epochs

        for epoch in range(epochs):
            a3, a2, a1 = self.forward(X)
            b1_grad = lr/m * np.sum(self.W2.T @ ((self.W3.T @ (a3 - Y)) * (a2 * (1 - a2))) * (a1 * (1 - a1)), keepdims=True, axis=1)
            W1_grad = lr/m * (self.W2.T @ (self.W3.T @ (a3 - Y) * (a2 * (1 - a2))) * (a1 * (1 - a1))) @ X.T
            b2_grad = lr/m * np.sum((self.W3.T @ (a3 - Y)) * (a2 * (1-a2)), keepdims=True, axis=1)
            W2_grad = lr/m * (self.W3.T @ (a3 - Y) * a2 * (1-a2)) @ a1.T
            b3_grad = lr/m * np.sum(a3 - Y, keepdims=True, axis=1)
            W3_grad = lr/m * (a3 - Y) @ a2.T

            self.b1 -= b1_grad
            self.W1 -= W1_grad
            self.b2 -= b2_grad
            self.W2 -= W2_grad
            self.b3 -= b3_grad
            self.W3 -= W3_grad

            if epoch % (epochs/20) == 0:
                print(f"epoch: {epoch}: Loss; {self.loss(X, Y)}, Training Accuracy; {self.acc(X, Y)}")

            if update_lr == True:
                if epoch % (epochs/5) == 0:
                    lr *= 0.5



