import torch

class RBM:
    """A restricted Boltzmann machine trainable with contrastive divergence, momentum,
    and L2 regularization.
    """

    def __init__(self, 
        num_visible,
        num_hidden,
        gibbs_steps,
        learning_rate=1e-3,
        momentum_coefficient=0.5,
        weight_decay=1e-4,
        device=None
    ):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.gibbs_steps = gibbs_steps
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.device = device

        # initialize parameters
        self.reset_parameters()

        # initialize tensors to 0 that will store the momenta of parameters
        self.weight_momenta = torch.zeros(num_visible, num_hidden, device=device)
        self.visible_bias_momenta = torch.zeros(num_visible, device=device)
        self.hidden_bias_momenta = torch.zeros(num_hidden, device=device)
    
    def train(self, train_loader, epochs=None, print_progress=False):
        """Train the RBM with contrastive divergence, momentum, and weight decay for the specified
        amount of epochs.

        Args:
            train_loader (torch.DataLoader): The DataLoader from which to load the training data.

            print_progress (bool): Flag indicating whether or not to print epoch start and epoch
            error after each epoch.

        Returns:
            list[float]: The history of training errors.
        """
        epoch = 0
        error_history = []
        while (epoch < epochs):
            if print_progress:
                print(f'Starting epoch {epoch}...')
            # train
            total_error = 0
            num_examples = 0
            for batch, _ in train_loader:
                # flatten input data
                batch = self._flatten_input_batch(batch)
                total_error += self._contrastive_divergence(batch, epoch, epochs)
                num_examples += batch.size(0)
            avg_error = total_error / num_examples
            # convert from tensor to single number
            error_history.append(avg_error.item())
            if print_progress:
                print(f'Finished epoch {epoch}. Avg error: {avg_error:.4f}')
            epoch += 1
        return error_history

    def test(self, test_loader):
        """Test the RBM on the given test data.

        Args:
            test_loader (torch.DataLoader): The DataLoader from which to load the test data.

        Returns:
            float: The average training error on the entire test set.
        """
        total_error = 0
        num_examples = 0
        for batch, _ in test_loader:
            batch = self._flatten_input_batch(batch)
            hidden_values = self.sample_hidden(batch)
            visible_values = self.sample_visible(hidden_values)
            total_error += torch.sum(torch.abs(visible_values - batch))
            num_examples += batch.size(0)
        return total_error / num_examples
        
    
    def reset_parameters(self):
        self.weights = torch.randn(self.num_visible, self.num_hidden, device=self.device) * 0.1
        self.visible_biases = torch.ones(self.num_visible, device=self.device) * 0.5
        self.hidden_biases = torch.ones(self.num_hidden, device=self.device) * 0.5

        # previous samples from Markov chain; used to initialize the Markov chain for PCD
        self.previous_visible_values = None

    def generate_sample(self, visible_values, mixing_time=10):
        """Generate a sample of the visible and hidden units using Gibbs sampling.

        Args:
            mixing_time (int, optional): The number of Gibbs steps to take before return a sample.
            Defaults to 10.

            visible_values (torch.Tensor): Visible unit values with which to initialize
            the Markov chain.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of tensors of size (visible_units, hidden_units) 
            containing the values of the visible and hidden units.
        """
        hidden_values = None
        for _ in range(mixing_time):
            hidden_values = self.sample_hidden(visible_values)
            visible_values = self.sample_visible(hidden_values)
        return (visible_values, hidden_values)
    
    def save(self, file_path):
        """Save the model's current parameters to file_path.

        Args:
            file_path (string): The file path to save the parameters to.
        """
        torch.save({
            'weights': self.weights,
            'visible_biases': self.visible_biases,
            'hidden_biases': self.hidden_biases
        }, file_path)

    def load(self, file_path):
        """Load parameters saved to file_path into the model.

        Args:
            file_path (string): The file path to load the parameters from.
        """
        parameters = torch.load(file_path, map_location=self.device)
        self.weights = parameters['weights']
        self.visible_biases = parameters['visible_biases']
        self.hidden_biases = parameters['hidden_biases']

    def _contrastive_divergence(self, input_data, epoch, total_epochs):
        """Perform one CD-k step using the input. Implements momentum and weight decay.

        Args:
            input_data (torch.Tensor): A design matrix of size (batch_size, num_visible).

            epoch (int): The current training epoch.

            total_epochs (int): The total number of training epochs to be performed.

        Returns:
            torch.Tensor: The scalar error, computed as the total squared L2 norm between
            the input and the reconstruction.
        """
        batch_size = input_data.size(0)

        # Calculate the postive phase gradients
        hidden_values = self.sample_hidden(input_data)
        weight_grads = torch.matmul(input_data.t(), hidden_values)
        visible_bias_grads = input_data
        hidden_bias_grads = hidden_values

        if (self.previous_visible_values == None):
            self.previous_visible_values = torch.randn_like(input_data, device=self.device)
        # Sample from the model for the negative phase
        visible_values, hidden_values = self.generate_sample(self.previous_visible_values, self.gibbs_steps)
        # Store samples to initialize the next Markov chain with (PCD)
        self.previous_visible_values = visible_values

        # Calculate the negative phase gradients
        weight_grads -= torch.matmul(visible_values.t(), hidden_values)
        visible_bias_grads -= visible_values
        hidden_bias_grads -= hidden_values

        # Average across the batch
        weight_grads /= batch_size
        visible_bias_grads /= batch_size
        hidden_bias_grads /= batch_size

        # Apply linear learning rate decay
        decayed_learning_rate = self.learning_rate - (self.learning_rate / total_epochs * epoch)

        # Calculate parameter momenta
        self.weight_momenta *= self.momentum_coefficient
        self.weight_momenta += self.learning_rate * weight_grads

        self.visible_bias_momenta *= self.momentum_coefficient
        self.visible_bias_momenta += self.learning_rate * torch.sum(visible_bias_grads, dim=0)

        self.hidden_bias_momenta *= self.momentum_coefficient
        self.hidden_bias_momenta += self.learning_rate * torch.sum(hidden_bias_grads, dim=0)

        # Update parameters
        self.weights += self.weight_momenta
        self.visible_biases += self.visible_bias_momenta
        self.hidden_biases += self.hidden_bias_momenta

        # Apply weight decay
        self.weights -= self.weights * self.weight_decay

        # Compute reconstruction error (L1 norm since values are binary)
        reconstruction = self.sample_visible(self.sample_hidden(input_data))
        error = torch.sum(torch.abs(input_data - reconstruction))

        return error


    def sample_hidden(self, visible_values):
        """Generate a sample from the hidden units, conditioned on the visible units.

        Args:
            visible_values (torch.Tensor): A tensor of size (batch_size, num_visible),
            where the i-th row is the state of the visible units for the i-th example.

        Returns:
            torch.Tensor: A tensor of size (batch_size, num_hidden), where the i-th row is the
            state of the hidden units for the i-th example.
        """
        hidden_probabilities = torch.sigmoid(
            self.hidden_biases 
            + torch.matmul(visible_values, self.weights)
        )
        return torch.bernoulli(hidden_probabilities)

    def sample_visible(self, hidden_values):
        """Generate a sample from the visible units, conditioned on the hidden units.

        Args:
            hidden_values (torch.Tensor): A tensor of size (batch_size, num_hidden),
            where the i-th row is the state of the hidden units for the i-th example.

        Returns:
            torch.Tensor: A tensor of size (batch_size, num_visible), where the ith row is the
            state of the visible units for the i-th example.
        """
        visible_probabilities = torch.sigmoid(
            self.visible_biases
            + torch.matmul(hidden_values, self.weights.t())
        )
        return torch.bernoulli(visible_probabilities)

    def _flatten_input_batch(self, input_batch):
        """Flatten a batch of inputs into a design matrix.

        Args:
            input_batch (torch.Tensor): A tensor of batched inputs, where the first dimension
            is the batch dimension.

        Returns:
            torch.Tensor: A (batch_size, num_visible) design matrix.
        """
        return input_batch.reshape(input_batch.size(0), self.num_visible).to(self.device)
        

         


