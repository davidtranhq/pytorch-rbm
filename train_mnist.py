import numpy as np
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from rbm import RBM

# Configuration
BATCH_SIZE = 64
VISIBLE_UNITS = 784 # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 1
PATIENCE = 3

DATA_FOLDER = 'data'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load MNIST data
print('Loading dataset...')

train_dataset = torchvision.datasets.MNIST(
    root=DATA_FOLDER,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# split into training and validation data
train_subset, validation_subset = torch.utils.data.random_split(
    train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42) # fix generator for reproducible results
)

train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=BATCH_SIZE)
validation_loader = torch.utils.data.DataLoader(validation_subset, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = torchvision.datasets.MNIST(
    root=DATA_FOLDER,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Train the RBM
print('Training the RBM...')
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, device=DEVICE)

train_history, validate_history = rbm.train(PATIENCE, train_loader, validation_loader)

# Save the model parameters
print('Saving model parameters...')
rbm.save('rbm_params.pt')

# Save the error history
print('Saving error history...')
with open('MNIST_loss.csv', 'w') as f:
    epoch = 0
    f.write('Epoch,Average Training Error,Average Validation Error')
    for train, validate in zip(train_history, validate_history):
        f.write(f'{epoch},{train},{validate}\n')

print('Done.')