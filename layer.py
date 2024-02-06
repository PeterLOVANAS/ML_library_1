class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass
    def backward(self, output_gradient):
        pass

    def parameters(self):
        pass

    def get_gradients(self):
        pass

    def update(self , new_param): # new param : np.array
        pass
# This is base class for layers


