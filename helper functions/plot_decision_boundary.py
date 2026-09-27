import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, Y, figsize=(8, 6), resolution=500):
  """
  Decision boundary plot

  Args:
  model: The model we are analyzing
  X: Training data
  Y: Training labels
  figsize: the size of the graph
  resolution: how well defined we want our graph to be

  Returns:
  prints the decision boundary
  """
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()

  x_padding = 0.1 * (x_max - x_min)
  y_padding = 0.1 * (y_max - y_min)

  x_min -= x_padding
  x_max += x_padding
  y_min -= y_padding
  y_max += y_padding

  xx, yy = np.meshgrid(
      np.linspace(x_min, x_max, resolution),
      np.linspace(y_min, y_max, resolution)
  )

  grid = np.c_[xx.ravel(), yy.ravel()]

  if model.name=="Support Vector Machine":
    bool_predictions = model.predict(X, Y, grid)
  elif model.name=="Neural Network":
    bool_predictions = model.predict(grid.T)
  else:
    bool_predictions=model.predict(grid)
  
  predictions = bool_predictions.astype(int).reshape(xx.shape)

  plt.figure(figsize=figsize)

  plt.contourf(xx, yy, predictions, alpha=0.3, cmap=plt.cm.coolwarm)
  plt.contour(xx, yy, predictions, levels=[0.5], colors="black", linewidths=2)
  plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, edgecolors="k", s=70)

  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.title(f"{model.name} Decision Boundary", fontsize=16)
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.show()