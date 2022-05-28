import torchvision.datasets
import torchvision.transforms

# Folder to store MNIST data
DATA_FOLDER = 'data'

class ToBinary:
    """Convert [0, 255] tensors to binary tensors."""

    def __call__(self, image):
        image[image < 0.5] = 0.0
        image[image >= 0.5] = 1.0
        return image

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