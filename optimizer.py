import numpy as np


"""
avg_grad_model = [avg_grad_part_1 , avg_grad_part_2 , avg_grad_part_3 , ... ,avg_grad_part_i]
avg_grad_part_i = [θ_1 , θ_2 , θ_3 , ... , θ_i] => Shape might be anything! (i,j)
"""


class Optimizer:
    def __init__(self , avg_grad_model):    # Hyperparameters of the optimizer (Ex. τ , β_1 , β_2 , ε ,etc.)
        self.avg_grad_model = avg_grad_model   # avg_grad_model can contain any parameters, θ (weight, bias ,etc.)
        # self.memory could be varied based on the optimizer.
        # Ex. Adam has Ɣ_t and μ_t
        # Ex. MGD has v_t


    def calculate(self):
        pass


    def initialize(self):
        pass


    
class Gradient_Descent(Optimizer):
    def __init__(self, avg_grad_model, learning_rate: float , momentum: float , acceleration: bool = False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.acceleration = acceleration
        self.velocity = self.initialize()
        
    def calculate(self):
        for v , p in zip(self.velocity , self.avg_grad_model):
            grad = p
            if acceleration == True:
                grad += (self.momentum * v)
            elif acceleration == False:
                pass
            
            v = (v*self.momentum)  - (self.learning_rate * grad)
         
        
        return self.velocity
            
        
        
        
    def initialize(self):
        velocity = []
        for p in self.avg_grad_model:
            v_x_p = np.zeros(p.shape)
            velocity.append(v_x_p)
        
        return np.array(velocity) 
            
        
        
        
