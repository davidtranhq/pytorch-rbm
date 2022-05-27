import torch
import torch.utils.data
from sklearn.model_selection import KFold

from load_data import load_binary_mnist
from rbm import RBM

# Configuration
BATCH_SIZE = 64
VISIBLE_UNITS = 784 # 28 x 28 images
HIDDEN_UNITS = 128
CD_K = 2
LEARNING_RATE = 1e-3
MOMENTUM_COEFFICIENT = 0.5
WEIGHT_DECAY = 1e-3
EPOCHS = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load MNIST data
print('Loading dataset...')

train_dataset, test_dataset = load_binary_mnist()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Train the RBM
print('Training the RBM...')

rbm = RBM(
    VISIBLE_UNITS,
    HIDDEN_UNITS,
    CD_K,
    weight_decay=WEIGHT_DECAY,
    momentum_coefficient=MOMENTUM_COEFFICIENT,
    device=DEVICE
)

train_history, test_history = [], []
for epoch in range(EPOCHS):
    print(f'Starting epoch {epoch}...')
    train_err = rbm.train(train_loader, epochs=1)
    test_err = rbm.test(test_loader)
    train_history += train_err
    test_history.append(test_err)
    print(f'Finished epoch {epoch}.'
        + f' Avg error (train|test): {train_err[0]:.4f}|{test_err:.4f}')

# Save the model parameters
print('Saving model parameters...')
rbm.save('MNIST_params.pt')

# Save the error history
print('Saving error history...')
with open('MNIST_loss.csv', 'w') as f:
    epoch = 0
    f.write('Epoch,Average Training Error,Average Validation Error\n')
    for train, validate in zip(train_history, test_history):
        f.write(f'{epoch},{train},{validate}\n')

print('Done.')