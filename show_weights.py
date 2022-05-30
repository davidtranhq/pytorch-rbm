from random import random
import torch
import matplotlib.pyplot as plt
import argparse
from random import randrange

from rbm import RBM

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

fig = plt.figure(figsize=(8,8))
columns = 10
rows = 10

rbm = RBM(784, 500, 10)
rbm.load(args.model)

weights = rbm.weights
for i in range(1, rows * columns + 1):
    j = randrange(501)
    img = weights.t()[j].reshape([28, 28]).cpu()
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()