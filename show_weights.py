from math import floor, ceil
from random import random
import torch
import numpy as np
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

rows = 4
cols = 5
n_imgs = rows * cols

hidden_idxs = np.random.choice(128, n_imgs)
_min, _max = torch.min(weights).item(), torch.max(weights).item()

fig, axes = plt.subplots(nrows=4, ncols=5)
for i, ax in enumerate(axes.flat):
    img = weights.t()[hidden_idxs[i]].reshape([28, 28])
    im = ax.imshow(img, cmap='gray', vmin=_min, vmax=_max)
    ax.axis('off')

fig.suptitle('Weights of Randomly Selected Hidden Units')

fig.colorbar(im, ax=axes.ravel().tolist())

plt.show()