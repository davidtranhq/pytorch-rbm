import torch
import matplotlib.pyplot as plt
import argparse

from rbm import RBM

BATCH_SIZE = 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 500
CD_K = 2

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, device=DEVICE)

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

print('Loading model...')
rbm.load(args.model)

fig = plt.figure(figsize=(8, 8))
columns = 5
rows = 4
print('Sampling...')
for i in range(1, columns*rows + 1):
    img, _ = rbm.generate_sample(None, 100)
    img = img.reshape([28, 28]).cpu()
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
fig.suptitle('MNIST Images Generated from a RBM')
plt.show()
