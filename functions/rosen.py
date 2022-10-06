import numpy as np

class Rosen():
  def evaluate(self, x):
    """
    The Rosenbrock function
    The function computed is::

        sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed
    
    Returns
    -------
    r : float
        The value of the Rosenbrock function
    """
    x = np.asarray(x)
    r = np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                axis=0)
    return r

  def gradient(self, x):
    """
    The derivative (i.e. gradient) of the Rosenbrock function.
    Parameters
    ----------
    x : array_like
        1-D array of points at which the derivative is to be computed.

    Returns
    -------
    rosen_der : (N,) ndarray
        The gradient of the Rosenbrock function at `x`. 
    """
    x = np.asarray(x)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = (200 * (xm - xm_m1**2) -
                 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
    der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2]**2)
    return der