import numpy as np

class Optimizer:
    """Base class for optimizers"""

    def update(self, params, grads):
        raise NotImplementedError


class SGD_momentum(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, learning_rate=0.01, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = None

    def update(self, params, grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(param) for param in params]

        for param, grad, velocity in zip(params, grads, self.velocities):
            velocity *= self.momentum
            velocity -= self.learning_rate * grad
            param += velocity


class RMSprop(Optimizer):
    """RMSprop optimizer"""

    def __init__(self, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None

    def update(self, params, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(param) for param in params]

        for param, grad, cache in zip(params, grads, self.cache):
            cache *= self.decay_rate
            cache += (1 - self.decay_rate) * grad ** 2
            param -= self.learning_rate * grad / (np.sqrt(cache) + self.epsilon)


class Adam(Optimizer):
    """Adam optimizer"""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
        self.t = 0  # Timestep

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1

        for param, grad, m, v in zip(params, grads, self.m, self.v):
            # Update biased first moment estimate
            m *= self.beta1
            m += (1 - self.beta1) * grad

            # Update biased second moment estimate
            v *= self.beta2
            v += (1 - self.beta2) * grad ** 2

            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)