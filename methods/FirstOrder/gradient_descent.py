import time
import random
import numpy as np

class GradientDescent():
  """
  An intuitive choice for descent direction d is the direction of steppest
  descent. Following the direction of steepest descent is guranteed to lead
  to improvement provided that the objective function is smooth, the step size
  is sufficiently small, and we are not already at a point where the gradient
  is zeo. The direction of steppest descent is the direction opposite the 
  gradient ∇f, hence the name gradient descent.
  """
  def __init__(self, function, learning_rate, initial_point):
    """
    The garadient descent class
    Parameters
    ----------
    function : a base function object
        Function is to be optimized on
    learning rate : float
        Learning rate for gradient descent
    initial_point : array_like
        A point is about to be optimized
    -------
    """
    self.lr = learning_rate 
    self.x = initial_point
    self.function = function
  
  def step(self):
    self.x = self.x - self.lr * self.function.gradient(self.x)
    return self.x