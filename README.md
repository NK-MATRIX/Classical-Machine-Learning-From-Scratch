# Classical-Machine-Learning-From-Scratch
Implementations of a logistic regression classifier, support vector machine, and a neural network from scratch using NumPy with derivations of their updates, and a Jupyter notebook detailing their performance on various datasets.

## Implemented Models and Mathematics
- **Logistic Regression:** A linear classifier, whose loss is derived from the Bernoulli distribution. Implemented with regular batch gradient descent.
- **Support Vector Machine:** A classifier that solves the regularized optimal margin problem using the Gaussian/RBF kernel. Implemented with a simplified version of the SMO algorithm.
- **Neural Network:** A three-layer multilayer perceptron (MLP) using the sigmoid activation function. Implemented with batch gradient descent.

## Purpose
This repository was created to implement classical models from their mathematical foundations rather than rely on libraries to do so. This project was inspired by my self-study of CS229.

## Notation
To ensure clarity, please note the following notation when going through the derivations:

* **Feature Dimensions:** Within our implementations, we use the convention that $X \in \mathbb{R}^{m \times n}$ for the logistic regression and support vector machine classes whereas $X \in \mathbb{R}^{n \times m}$ for our neural network class.
    * $m$ denotes the number of training examples.
    * $n$ denotes the number of features.
* **Label Dimensions:** $Y \in \mathbb{R}^{m}$ for all models. Within the SVM, $Y \in \\{-1, 1\\}$ whereas in the neural network and logistic regression classifier, $Y \in \\{0, 1\\}$.
* **Parameter Notation:** $W$ denotes the weights and $b$ denotes the bias for the support vector machine (as a linear classifier in the optimal margin problem) and neural network whereas $\theta$ denotes our parameter vector for our the logistic regression classifier.

## Experiments
Within the decision boundaries Jupyter Notebook, we will test our models on synthetic 2D datasets generated with scikit-learn, including make_blobs, make_circles, and make_moons. We will then visualize our decision boundaries and compare how linear vs non-linear classifiers behave.

## Limitations
* These classes are educational implementations rather than production machine learning libraries, making them intentionally limited.
* The experiments use relatively small synthetic datasets, including 1000 examples with only two features.
* These implementations intentionally prioritize an understanding of the fundamental mathematics rather than software or optimization engineering.


## Repository Structure

```text
Classical-Machine-Learning-From-Scratch
|
├──README.md
|
├── math
│   ├── logistic_regression.pdf
│   ├── svm.pdf
│   └── neural_network.pdf
│
├── models
│   ├── Logistic_Regression_Class.py
│   ├── Support_Vector_Machine_Class.py
│   └── Neural_Network_Class.py
│
├── helpers
│   └── plot_decision_boundary.py
│
└── experiments
    └── decision_boundaries.ipynb
```
