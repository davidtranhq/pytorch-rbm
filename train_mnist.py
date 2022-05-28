import torch
import torch.utils.data
import argparse
from sklearn.model_selection import KFold

from load_data import load_binary_mnist
from rbm import RBM

# Get name to be used for the model from CLI
parser = argparse.ArgumentParser()
parser.add_argument('model_name', nargs='?', default='model', help='the name of the model')
parser.add_argument('--hidden', type=int, default=128, help='number of hidden units')
parser.add_argument('--cdk', type=int, default=2, help='number of CD iterations')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, help='momentum coefficient')
parser.add_argument('--decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
args = parser.parse_args()

MODEL_NAME = args.model_name

# Configuration
BATCH_SIZE = 64
VISIBLE_UNITS = 784 # 28 x 28 images
HIDDEN_UNITS = args.hidden
CD_K = args.cdk
LEARNING_RATE = args.lr
MOMENTUM_COEFFICIENT = args.momentum
WEIGHT_DECAY = args.decay
EPOCHS = args.epochs

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hyperparams = f'''
Using {torch.cuda.get_device_name(DEVICE)}
BATCH_SIZE = {BATCH_SIZE}
VISIBLE_UNITS = {VISIBLE_UNITS}
HIDDEN_UNITS = {HIDDEN_UNITS}
CD_K = {CD_K}
LEARNING_RATE = {LEARNING_RATE}
MOMENTUM_COEFFICIENT = {MOMENTUM_COEFFICIENT}
WEIGHT_DECAY = {WEIGHT_DECAY}
EPOCHS = {EPOCHS}
'''

print(hyperparams)

# Load MNIST data
print('Loading dataset...')

train_dataset, test_dataset = load_binary_mnist()

train_dataset, validate_dataset = torch.utils.data.random_split(
    train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(74)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Train the RBM
print(f'Training \'{MODEL_NAME}\'')

rbm = RBM(
    VISIBLE_UNITS,
    HIDDEN_UNITS,
    CD_K,
    weight_decay=WEIGHT_DECAY,
    momentum_coefficient=MOMENTUM_COEFFICIENT,
    device=DEVICE
)

train_history, validate_history = [], []
for epoch in range(EPOCHS):
    print(f'Starting epoch {epoch}...')
    train_err = rbm.train(train_loader, epochs=1)
    validate_err = rbm.test(validate_loader)
    train_history += train_err
    validate_history.append(validate_err)
    print(f'Finished epoch {epoch}.'
        + f' Avg error (train|validate): {train_err[0]:.4f}|{validate_err:.4f}')

# Save the model parameters
param_file = f'models/{MODEL_NAME}_params.pt'
print(f'Saving model parameters to {param_file}...')
rbm.save(param_file)

# Save the hyperparameters
hyperparam_file = f'models/{MODEL_NAME}_hyperparams.txt'
print(f'Saving hyperparameters to {hyperparam_file}...')
with open(hyperparam_file, 'w') as f:
    f.write(hyperparams)

# Save the error history
perform_file = f'models/{MODEL_NAME}_loss.csv'
print(f'Saving model performance to {perform_file}...')
with open(perform_file, 'w') as f:
    epoch = 0
    f.write('Epoch,Average Training Error,Average Validation Error\n')
    for train, validate in zip(train_history, validate_history):
        f.write(f'{epoch},{train},{validate}\n')
        epoch += 1

print('Done.')