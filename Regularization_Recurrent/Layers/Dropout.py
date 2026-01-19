import numpy as np
from Layers import Base


# dropout layer implementing inverted dropout
# during training, randomly drops units with probability (1-p) and scales by 1/p
# during testing, passes input unchanged ( no dropout applied )
# it's called inverted dropout because scaling is done during training, so no scaling is needed during testing
class Dropout(Base.Base):
    def __init__(self, probability):
        super().__init__()
        self.trainable = False
        self.probability = probability
        self.mask = None
    
    def forward(self, input_tensor):
        if self.testing_phase:
            # During testing, pass through unchanged
            return input_tensor
        else:
            # During training: inverted dropout
            # Create mask: 1 with probability p, 0 with probability (1-p)
            self.mask = np.random.binomial(1, self.probability, input_tensor.shape)
            
            # Apply mask and scale by 1/p, so expected value remains the same ( compensation for dropped units )
            output = input_tensor * self.mask / self.probability
            return output
    
    def backward(self, error_tensor):
        return error_tensor * self.mask / self.probability
