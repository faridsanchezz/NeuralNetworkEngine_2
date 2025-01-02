import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Layer:
    """Base class for neural network layers"""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def get_params_and_grads(self):
        return [], []


class DenseLayer(Layer):
    """Fully connected layer"""
    def __init__(self, input_size, output_size, init_method='he'):
        if init_method == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif init_method == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        elif init_method == 'normal':
            self.weights = np.random.randn(input_size, output_size) * 0.01

        self.biases = np.zeros(output_size)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, delta): # delta proviene al derivar la funcion de activacion respecto al loss
        self.grad_weights = np.dot(self.input.T, delta)
        self.grad_biases = np.sum(delta, axis=0)
        return np.dot(delta, self.weights.T)

    def get_params_and_grads(self):
        return [self.weights, self.biases], [self.grad_weights, self.grad_biases]


class NeuralNetwork:
    """Neural Network class that combines layers and handles training"""

    def __init__(self, layers, loss_function, optimizer):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.training_history = []

    def forward(self, x):
        """Forward pass through the network"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        """Backward pass through the network"""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def get_params_and_grads(self):
        """Get parameters and gradients from all layers"""
        params, grads = [], []
        for layer in self.layers:
            layer_params, layer_grads = layer.get_params_and_grads()
            params.extend(layer_params)
            grads.extend(layer_grads)
        return params, grads

    def train_step(self, x, y):
        """Perform one training step"""
        predictions = self.forward(x) # prediccion
        loss = self.loss_function.forward(predictions, y) # perdida -> compara prediccion con la etiqueta real
        grad_output = self.loss_function.backward(predictions, y) # gradiente inical
        self.backward(grad_output) # retropropagacion del gradiente

        params, grads = self.get_params_and_grads()
        self.optimizer.update(params, grads)
        return loss

    def create_mini_batches(self, X, y, batch_size):
        """Create mini-batches from the data"""
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        mini_batches = []
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            mini_batches.append((X_batch, y_batch))
        return mini_batches

    def predict(self, X, batch_size=None):
        """
        Make predictions for new data without modifying the model's weights.

        Parameters:
        -----------
        X : array-like
            Input data
        batch_size : int or None, default=None
            If specified, predictions are made in batches

        Returns:
        --------
        predictions : array-like
            Model predictions
        """
        if batch_size is None:
            return self.forward(X)

        # Batch prediction for large datasets
        predictions = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_predictions = self.forward(batch_X)
            predictions.append(batch_predictions)

        return np.vstack(predictions)


def train_network(network, X, y, epochs=100, batch_size=32, early_stopping_patience=None,
                  validation_split=0.2, verbose=1, val_split_random_state=None):
    """Train neural network using validation data for early stopping"""

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=val_split_random_state
    )

    history = {
        'val_loss': [],
        'val_acc': []
    }
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        mini_batches = network.create_mini_batches(X_train, y_train, batch_size)
        for X_batch, y_batch in mini_batches:
            network.train_step(X_batch, y_batch)

        # Validation phase
        val_predictions = network.predict(X_val)
        val_loss = network.loss_function.forward(val_predictions, y_val)

        # Convert predictions and targets to class labels for accuracy calculation
        val_pred_classes = np.argmax(val_predictions, axis=1)
        val_true_classes = np.argmax(y_val, axis=1)
        val_acc = accuracy_score(val_true_classes, val_pred_classes)

        # Store metrics
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping check
        if early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if verbose == 1 and epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        elif verbose == 2:
            print(f"Epoch {epoch}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history


