import sys

# setting path 
sys.path.append('../')

from functions import Rosen
from methods import GradientDescent
import logging
import numpy as np

def main():
  function = Rosen()
  lr = 0.5
  x = np.array([1, 0, -1.2])
  algo = GradientDescent(function, lr, x)
  format = "%(asctime)s: %(message)s"
  logging.basicConfig(format=format, level=logging.INFO, 
                      datefmt="%H:%M:%S")
  logging.info("The next point is {}".format(algo.step()))

if __name__ == "__main__":
  # logging.getLogger().setLevel(logging.DEBUG)
  main()