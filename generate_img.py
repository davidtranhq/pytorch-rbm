import torch
import matplotlib.pyplot as plt

from rbm import RBM

BATCH_SIZE = 64
VISIBLE_UNITS = 784  # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
PATIENCE = 3

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, device=DEVICE)

print('Loading model...')
rbm.load('models/FashionMNIST_params.pt')

fig = plt.figure(figsize=(8,8))
columns = 5
rows = 4
print('Sampling...')
for i in range(1, columns*rows + 1):
    img = rbm.generate_sample().reshape([28, 28]).cpu()
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    fig.suptitle('FashionMNIST Images Generated from RBM')
    plt.imshow(img, cmap='gray')
plt.show()