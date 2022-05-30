from random import random
import torch
import matplotlib.pyplot as plt
import argparse
from random import randrange

from rbm import RBM

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

rbm = RBM(784, 128, 10)
rbm.load(args.model)
weights = rbm.weights

fig = plt.figure(figsize=(8,8))
columns = 5
rows = 4


for i in range(1, rows * columns + 1):
    j = randrange(128)
    img = weights.t()[j].reshape([28, 28]).cpu()
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img, cmap='gray')

fig.suptitle('Weights of Randomly Selected Hidden Units')
plt.show()