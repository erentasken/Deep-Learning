import numpy as np
from Layers import Base, Helpers

class BatchNormalization(Base.Base):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        
        # Learnable parameters
        self.weights = np.ones(channels)  # gamma (scale)
        self.bias = np.zeros(channels)    # beta (shift)
        
        # Moving averages for inference
        self.moving_mean = None
        self.moving_var = None
        self.alpha = 0.8
        
        # Cache
        self.input_tensor = None
        self.normalized_tensor = None
        self.input_shape = None

    def reformat(self, tensor):
        if len(tensor.shape) == 4:  # 4D → 2D
            C = tensor.shape[1]
            tensor = np.transpose(tensor, (0, 2, 3, 1))
            return tensor.reshape(-1, C)
        else:  # 2D → 4D
            reshaped = tensor.reshape(self.input_shape[0], self.input_shape[2], self.input_shape[3], self.input_shape[1])
            return np.transpose(reshaped, (0, 3, 1, 2))

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        if len(input_tensor.shape) == 4:
            input_tensor = self.reformat(input_tensor)
        self.input_tensor = input_tensor

        if self.testing_phase:
            if self.moving_mean is None:
                self.moving_mean = np.zeros(self.channels)
                self.moving_var = np.ones(self.channels)
            normalized = (input_tensor - self.moving_mean) / np.sqrt(self.moving_var + 1e-10)
        else:
            mean = np.mean(input_tensor, axis=0)
            var = np.var(input_tensor, axis=0)
            if self.moving_mean is None:
                self.moving_mean, self.moving_var = mean.copy(), var.copy()
            else:
                self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * mean
                self.moving_var  = self.alpha * self.moving_var + (1 - self.alpha) * var
            self.normalized_tensor = (input_tensor - mean) / np.sqrt(var + 1e-10)
            normalized = self.normalized_tensor

        output = self.weights * normalized + self.bias
        if len(self.input_shape) == 4:
            output = self.reformat(output)
        return output

    def backward(self, error_tensor):
        if len(self.input_shape) == 4:
            error_tensor = self.reformat(error_tensor)

        # Gradients for scale and shift
        self.gradient_weights = np.sum(error_tensor * self.normalized_tensor, axis=0)
        self.gradient_bias    = np.sum(error_tensor, axis=0)

        # Update params if optimizer exists
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias    = self.optimizer.calculate_update(self.bias, self.gradient_bias)

        # Gradient w.r.t input
        input_grad = Helpers.compute_bn_gradients(
            error_tensor, self.input_tensor, self.weights, np.mean(self.input_tensor, axis=0),
            np.var(self.input_tensor, axis=0), eps=1e-10
        )

        if len(self.input_shape) == 4:
            input_grad = self.reformat(input_grad)
        return input_grad
