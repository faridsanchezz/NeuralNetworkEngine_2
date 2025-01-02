import numpy as np

class Loss:
    """Base class for loss functions"""
    def forward(self, predicted, target):
        raise NotImplementedError

    def backward(self, predicted, target):
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error loss function"""
    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        return np.mean((predicted - target) ** 2)

    def backward(self, predicted, target):
        return 2 * (predicted - target) / target.size


class CrossEntropy(Loss):
    """Cross Entropy loss function"""

    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.sum(target * np.log(predicted)) / target.shape[0]

    def backward(self, predicted, target):
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -(target / predicted) / target.shape[0]

