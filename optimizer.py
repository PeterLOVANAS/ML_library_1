import numpy
import numpy as np


"""
avg_grad_model = [avg_grad_part_1 , avg_grad_part_2 , avg_grad_part_3 , ... ,avg_grad_part_i]
avg_grad_part_i = [θ_1 , θ_2 , θ_3 , ... , θ_i] => Shape might be anything! (i,j)
"""

# Polymorphism [OOP]
class Optimizer:
    def __init__(self):    # Hyperparameters of the optimizer (Ex. τ , β_1 , β_2 , ε ,etc.)
        self.avg_grad_model = None  # avg_grad_model can contain any parameters, θ (weight, bias ,etc.)
        # self.memory could be varied based on the optimizer.
        # Ex. Adam has Ɣ_t and μ_t
        # Ex. MGD has v_t


    def compute(self):
        pass


    def initialize(self):
        pass



class Gradient_Descent(Optimizer):
    def __init__(self, avg_grad_model, learning_rate: float , momentum: float , acceleration: bool = False):
        super().__init__()
        self.avg_grad_model = avg_grad_model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.acceleration = acceleration
        self.velocity = self.initialize()

    def compute(self):
        velocity = []
        for v , p in zip(self.velocity , self.avg_grad_model):
            grad = p
            if self.acceleration == True:
                grad += (self.momentum * v)
            elif self.acceleration == False:
                pass

            velocity.append((v*self.momentum)  - (self.learning_rate * grad))

        self.velocity = np.array(velocity)

        return self.velocity

    def initialize(self):
        velocity = []
        for p in self.avg_grad_model:
            v_x_p = np.zeros(p.shape)
            velocity.append(v_x_p)

        return np.array(velocity)



class RMSprop(Optimizer):
    def __init__(self, avg_grad_model , learning_rate: float , momentum: float , beta:float = 0.9 , epsilon: float = 1e-7):
        super().__init__()
        self.avg_grad_model = avg_grad_model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.epsilon = epsilon
        self.mu, self.velocity = self.initialize()

    def compute(self):
        velocity = []
        mu = []
        for v ,m, p in zip(self.velocity , self.mu,self.avg_grad_model):
            m_p  = (self.beta * m) + ((1-self.beta)*(p**2))
            mu.append(m_p)
            velocity.append((v*self.momentum) - ((self.learning_rate / (np.sqrt(m_p) + self.epsilon)) * p))

        self.velocity = np.array(velocity)
        self.mu = np.array(mu)

        return self.velocity

    def initialize(self):
        velocity = []
        mu = []
        for p in self.avg_grad_model:
            v_x_p = np.zeros(p.shape)
            velocity.append(v_x_p)
            mu.append(v_x_p) # v_x_p is the same as what we want in mu

        return np.array(mu) , np.array(velocity)



class Adam(Optimizer):
    def __inti__(self , avg_grad_model , learning_rate: float, beta_1: float ,beta_2: float ,epsilon: float):
        super().__init__()
        self.avg_grad_model = avg_grad_model
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.t = 1
        self.epsilon = epsilon
        self.gamma ,self.mu =  self.initialize()


    def compute(self):
        gamma = []
        mu = []
        for  g, m ,p in zip(self.gamma , self.mu ,self.avg_grad_model):
            current_g = (self.beta_1 * g) + ((1- self.beta_1) * p)
            current_m = (self.beta_2 * m) + ((1- self.beta_2) * (p ** 2))

            corrected_g = current_g / (1 - (self.beta_1 ** self.t ))
            corrected_m = current_m / (1 - (self.beta_2 ** self.t ))

            grad_term = (self.learning_rate / (np.sqrt(corrected_m) + self.epsilon)) * corrected_g

            self.t += 1
            gamma.append(current_g)
            mu.append(current_m)

        self.gamma = np.array(gamma)
        self.mu = np.array(mu)

        return grad_term



    def initialize(self):
        gamma = []
        mu = []
        for p in self.avg_grad_model:
            v_x_p = np.zeros(p.shape)
            gamma.append(v_x_p)
            mu.append(v_x_p)


        return np.array(gamma) , np.array(mu)

