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
EPOCHS = 10
K_FOLDS = 5

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load MNIST data
print('Loading dataset...')

train_dataset, test_dataset = load_binary_mnist()

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Train the RBM
print('Training the RBM...')
rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, device=DEVICE)
kfold = KFold(n_splits=K_FOLDS, shuffle=True)

# error histories for each cross-validation fold
train_err_histories, validate_err_histories = [], []

# iterate through each cross-validation fold
for fold, (train_ids, validate_ids) in enumerate(kfold.split(train_dataset)):
    print(f'Starting fold {fold + 1} of {K_FOLDS}')
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    validate_subsampler = torch.utils.data.SubsetRandomSampler(validate_ids)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_subsampler,
    )
    validation_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=validate_subsampler,
    )

    # error history for the current fold
    train_err_history, validate_err_history = [], []
    for epoch in range(EPOCHS):
        print(f'Starting epoch {epoch}...')
        train_err = rbm.train(train_loader, epochs=1)
        validate_err = rbm.test(validation_loader)
        train_err_history += train_err
        validate_err_history.append(validate_err)
        print(f'Finished epoch {epoch}.'
            + f' Avg error (train|test): {train_err[0]:.4f}|{validate_err:.4f}')
    train_err_histories.append(train_err_history)
    validate_err_histories.append(validate_err_history)
    if fold + 1 < K_FOLDS:
        rbm.reset_parameters()

def avg_across_folds(histories):
    return [sum(errs) / len(errs) for errs in zip(*histories)]

avg_train_err_history = avg_across_folds(train_err_histories)
avg_validate_err_history = avg_across_folds(validate_err_histories)

# Save the model parameters
print('Saving model parameters...')
rbm.save('MNIST_params.pt')

# Save the error history
print('Saving error history...')
with open('MNIST_loss.csv', 'w') as f:
    epoch = 0
    f.write('Epoch,Average Training Error,Average Validation Error\n')
    for train, validate in zip(avg_train_err_history, avg_validate_err_history):
        f.write(f'{epoch},{train},{validate}\n')

print('Done.')