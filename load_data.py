import torchvision.datasets
import torchvision.transforms

from to_binary import ToBinary

DATA_FOLDER = 'data'

def load_binary_mnist():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ToBinary()
    ])
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_FOLDER,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=DATA_FOLDER,
        train=False,
        transform=transform,
        download=True
    )

    return (train_dataset, test_dataset)