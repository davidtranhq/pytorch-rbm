import matplotlib.pyplot as plt
import argparse
import torchvision.datasets
import torchvision.transforms
from random import randrange

from rbm import RBM

parser = argparse.ArgumentParser()
parser.add_argument('model')
# args = parser.parse_args()

test_dataset = torchvision.datasets.MNIST(
    root='data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

rbm = RBM(784, 128, 10)
rbm.load('models/0_params.pt')

fig = plt.figure(constrained_layout=True)
subfigs = fig.subfigures(3, 2)

fig.suptitle('RBM Reconstructions from Original Examples')

for _, subfig in enumerate(subfigs.flat):
    axs = subfig.subplots(1, 2)
    i = randrange(len(test_dataset))

    original = test_dataset[i][0]
    reconstruction, _ = rbm.generate_sample(original.reshape(784), 1)
    axs[0].imshow(original.reshape([28, 28]), cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(reconstruction.reshape([28, 28]), cmap='gray')
    axs[1].set_title('Reconstruction')
    axs[1].axis('off')

plt.show()
