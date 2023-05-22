import numpy as np


"""
avg_grad_model = [avg_grad_part_1 , avg_grad_part_2 , avg_grad_part_3 , ... ,avg_grad_part_i]
avg_grad_part_i = [θ_1 , θ_2 , θ_3 , ... , θ_i] => Shape might be anything! (i,j)
"""


class Optimizer:
    def __init__(self):    # Hyperparameters of the optimizer (Ex. τ , β_1 , β_2 , ε ,etc.)
        self.avg_grad_model = None   # avg_grad_model can contain any parameters, θ (weight, bias ,etc.)
        # self.memory could be varied based on the optimizer.
        # Ex. Adam has Ɣ_t and μ_t
        # Ex. MGD has v_t


    def calcualate(self):
        pass


    def initialize(self):
        pass
