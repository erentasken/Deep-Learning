import numpy as np

class Optimizer:
    def __init__(self):
        self.regularizer = None
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Adam(Optimizer):
    def __init__(self, learning_rate, beta1, beta2, epsilon=1e-8):
        super().__init__()
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def calculate_update(self, weights, gradients):
        # Apply weight decay if regularizer is set
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weights)
            weights = weights - self.lr * regularizer_gradient
        
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        # first moment ( momentum )
        # if the gradients keeps growing in the same direction , m grows, step is faster 
        # if it changes the direction, m smooths the oscillation
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients 

        #second moment ( uncentered variance )
        #gradient magnitude → adaptive step size
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)  

        # bias-corrected moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # update weights
        weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum):
        super().__init__()
        self.lr = learning_rate
        self.momentum = momentum # it is momentum factor, corresponds to mass in physics ( mv = p)
        self.velocity = None

    def calculate_update(self, weights, gradients):
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weights)
            weights = weights - self.lr * regularizer_gradient
        
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        # update velocity, old velocity is multiplied by momentum factor ( like friction ) and current gradient is added
        self.velocity = self.momentum * self.velocity + self.lr * gradients

        # update weights
        weights -= self.velocity
        return weights
    
class Sgd(Optimizer):
    learning_rate = None
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # Apply weight decay if regularizer is set
        if self.regularizer is not None:
            regularizer_gradient = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * regularizer_gradient
        
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor