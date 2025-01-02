import numpy as np

class Activation:
    """Base class for activation functions"""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class ReLU(Activation):
    """Rectified Linear Unit activation function"""
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

    def get_params_and_grads(self):
        """ReLU has no parameters to return"""
        return [], []


class Sigmoid(Activation):
    """Sigmoid activation function"""
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to avoid overflow
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

    def get_params_and_grads(self):
        """Sigmoid has no parameters to return"""
        return [], []

class Tanh(Activation):
    """Hyperbolic tangent activation function"""

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

    def get_params_and_grads(self):
        """Tanh has no parameters to return"""
        return [], []


class Softmax(Activation):
    """Softmax activation function"""

    def forward(self, x):
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        # Each output depends on all inputs, so we need to account for this
        n_samples = self.output.shape[0]
        jacobian = np.zeros((n_samples, self.output.shape[1], self.output.shape[1]))

        for i in range(n_samples):
            softmax = self.output[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(softmax) - np.dot(softmax, softmax.T)

        return np.array([np.dot(grad_output[i], jacobian[i]) for i in range(n_samples)])

    def get_params_and_grads(self):
        """Softmax has no parameters to return"""
        return [], []
