import numpy as np


# enforces sparsity in weights
class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha # regularization strength
    
    def calculate_gradient(self, weights):
        return np.sign(weights) * self.alpha
    
    def norm(self, weights):
        return np.sum(np.abs(weights)) * self.alpha

# penalizes large weights
class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha # regularization strength
    
    def calculate_gradient(self, weights):
        return weights * self.alpha
    
    def norm(self, weights):
        return np.sum(weights ** 2) * self.alpha
