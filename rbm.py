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
        self.weights = torch.randn(num_visible, num_hidden, device=device) * 0.1
        self.visible_biases = torch.ones(num_visible, device=device) * 0.5
        self.hidden_biases = torch.ones(num_hidden, device=device) * 0.5

        # initialize tensors to 0 that will store the momenta of parameters
        self.weights_momentum = torch.zeros(num_visible, num_hidden, device=device)
        self.visible_biases_momentum = torch.zeros(num_visible, device=device)
        self.hidden_biases_momentum=torch.zeros(num_hidden, device=device)
    
    def infer_hidden(self, visible_probabilities):
        """Calculate the distribution of the hidden units, conditioned on the visible units.

        Args:
            visible_probabilities (torch.Tensor): A tensor of size (batch_size, num_visible),
            where the ith row is the distribution of the visible units for the ith example.

        Returns:
            torch.Tensor: A tensor of size (batch_size, num_hidden), where the ith row is
            the distribution of the hidden units for the ith batch.
        """
        hidden_activations = self.hidden_biases + torch.matmul(visible_probabilities, self.weights)
        hidden_probabilities = torch.sigmoid(hidden_activations)
        return hidden_probabilities

    def infer_visible(self, hidden_probabilities):
        """Calculate the distribution of the visible units, conditioned on the hidden units.

        Args:
            hidden_probabilities (torch.Tensor): A tensor of size (batch_size, num_hidden),
            where the ith row is the distribution of the hidden units for the ith example.

        Returns:
            torch.Tensor: A tensor of size (batch_size, num_visible), where the ith row is the
            distribution of the hidden units for the ith batch.
        """
        visible_activations = self.visible_biases + torch.matmul(hidden_probabilities, self.weights.t())
        visible_probabilities = torch.sigmoid(visible_activations)
        return visible_probabilities
    
    def contrastive_divergence(self, input_data):
        """Perform one CD-k step using the input. Implements momentum and weight decay.

        Args:
            input (torch.Tensor): A design matrix of size (batch_size, num_visible).

        Returns:
            torch.Tensor: The scalar error, computed as the total squared L2 norm between
            the input and the reconstruction.
        """
        batch_size = input_data.size(0)
        # Calculate the postive phase
        positive_hidden_probabilities = self.infer_hidden(input_data) # dim (batch_size, num_hidden)
        positive_hidden_activations = self._sample_bernoulli(positive_hidden_probabilities)
        positive_weights_gradient = torch.matmul(
            input_data.t(),
            positive_hidden_activations
        ) / batch_size

        # Calculate the negative phase
        # k steps of Gibbs sampling
        hidden_activations = positive_hidden_activations
        for step in range(self.gibbs_steps):
            # inferring the visible units with the activations (instead of the probabilities)
            # has a regularizing effect
            visible_probabilities = self.infer_visible(hidden_activations)
            hidden_probabilities = self.infer_hidden(visible_probabilities)
            hidden_activations = self._sample_bernoulli(hidden_probabilities)

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities
        negative_weights_gradient = torch.matmul(
            negative_visible_probabilities.t(),
            negative_hidden_probabilities
        ) / batch_size

        # Calculate momentum
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (
            self.learning_rate * (positive_weights_gradient - negative_weights_gradient)
        )

        self.visible_biases_momentum *= self.momentum_coefficient
        self.visible_biases_momentum += self.learning_rate * torch.sum(
            input_data - negative_visible_probabilities,
            dim=0
        )

        self.hidden_biases_momentum *= self.momentum_coefficient
        self.hidden_biases_momentum += self.learning_rate * torch.sum(
            positive_hidden_probabilities - negative_hidden_probabilities,
            dim=0
        )

        # Update parameters
        self.weights += self.weights_momentum
        self.visible_biases += self.visible_biases_momentum
        self.hidden_biases += self.hidden_biases_momentum

        # Apply weight decay
        self.weights -= self.weights * self.weight_decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities) ** 2)

        return error
        
    def train(self, patience, train_loader, validation_loader):
        """Train the RBM with contrastive divergence, momentum, weight decay, and early stopping.

        Args:
            patience (int): Maximum number of epochs to run without improvement before stopping
            train_loader (torch.DataLoader): DataLoader for the training set
            validation_loader (torch.DataLoader): DataLoader for the validation set

        Returns:
            tuple[list[float], list[float]]: The history of training and validation errors.
        """
        best_weights = self.weights.clone()
        best_visible_biases = self.visible_biases.clone()
        best_hidden_biases = self.hidden_biases.clone()
        best_validation_error = None
        epoch = 0
        epochs_since_improvement = 0
        validation_error = 0
        training_error_history = []
        validation_error_history = []
        while (epochs_since_improvement < patience):
            print(f'Starting epoch {epoch}...')
            # train
            training_error = 0
            for train_batch, _ in train_loader:
                # flatten input data
                train_batch = self._flatten_input_batch(train_batch)
                training_error += self.contrastive_divergence(train_batch)
            average_training_error = training_error / len(train_loader.dataset)
            training_error_history.append(average_training_error.item())
            # validate
            for validate_batch, _ in validation_loader:
                validate_batch = self._flatten_input_batch(validate_batch)
                hidden = self.infer_hidden(validate_batch)
                visible = self.infer_visible(hidden)
                # squared L2 norm
                validation_error += torch.sum((validate_batch - visible) ** 2)
            if (best_validation_error == None or validation_error < best_validation_error):
                # the model improved
                best_weights = self.weights.clone()
                best_visible_biases = self.visible_biases.clone()
                best_hidden_biases = self.hidden_biases.clone()
                best_validation_error = validation_error
                epochs_since_improvement = 0
            else:
                # the model did not improve
                epochs_since_improvement += 1
            average_validation_error = validation_error / len(validation_loader.dataset)
            validation_error_history.append(average_validation_error.item())
            print(f'Finished epoch. Average Train|Validate Error: '
                f'{average_training_error:.2f}|{average_validation_error:.2f}')
            epoch += 1
            validation_error = 0
        self.weights = best_weights
        self.visible_biases = best_visible_biases
        self.hidden_biases = best_hidden_biases
        return (training_error_history, validation_error_history)
    
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
        parameters = torch.load(file_path)
        self.weights = parameters['weights']
        self.visible_biases = parameters['visible_biases']
        self.hidden_biases = parameters['hidden_biases']

    def generate_sample(self, mixing_time=10):
        """Generate a sample from the visible units of the model using Gibbs sampling.

        Args:
            mixing_time (int, optional): The number of Gibbs steps to take before sampling.
            Defaults to 10.

        Returns:
            torch.Tensor: A tensor of size (visible_units) containing the probabilities
            of the visible units.
        """
        visible_probabilities = torch.randn(self.num_visible, device=self.device)
        for t in range(mixing_time):
            hidden_probabilities = self.infer_hidden(visible_probabilities)
            visible_probabilities = self.infer_visible(hidden_probabilities)
        return visible_probabilities

    def _flatten_input_batch(self, input_batch):
        """Flatten a batch of inputs into a design matrix.

        Args:
            input_batch (torch.Tensor): A tensor of batched inputs, where the first dimension
            is the batch dimension.

        Returns:
            torch.Tensor: A (batch_size, num_visible) design matrix.
        """
        return input_batch.reshape(input_batch.size(0), self.num_visible).to(self.device)

    def _sample_bernoulli(self, distribution):
        """Sample from a Bernoulli distribution.

        Args:
            distribution (torch.Tensor): A Bernoulli distribution, where distribution_i indicates
            the probability of the ith variable being 1.
        
        Returns:
            torch.Tensor: A sample from the distribution as a binary tensor of floats.
        """
        num_vars = distribution.size()
        random_nums = torch.rand(num_vars).to(self.device)
        bernoulli_sample = (random_nums <= distribution).float()
        return bernoulli_sample
        

         


