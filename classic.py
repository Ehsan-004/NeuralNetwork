import numpy as np
# from core.derivatives import linear as lnd
from core.activations import linear, ReLU
from core.loss import sqe


class Neuron:
    def __init__(self):
        self.weight = abs(np.random.randn()) ; print(f"weight = {self.weight}")
        self.bias = np.random.randn() ;        print(f"bias = {self.bias}")
        
    def set(self, x: int, y: int, activation):
        self.activation = activation
        self.x = x
        self.y = y
        
    def forward(self):
        print(f"w = {self.weight} | x = {self.x} | b = {self.bias}")
        output = self.activation(self.weight*self.x + self.bias)
        print(f"output = {output}")
        return output
          
          
def initialize_weights():
    weight = abs(np.random.randn()) ; print(f"weight = {weight}")
    bias = np.random.randn() ;        print(f"bias = {bias}")
    return weight, bias
          
def forward(weight, x, bias, activation):
    # print(f"w = {self.weight} | x = {self.x} | b = {self.bias}")
    output = activation(weight*x + bias)
    print(f"output = {output}")
    return output
        
            
    
def main():    
    x = [1,2,3,4,5]
    y = [5,8,11,14,17]
    
    # n = Neuron()
    # n.forward()
    w, b = initialize_weights()
    
    for i in range(len(x)):
        o = forward(w, x[i], b, ReLU)
        
        


if __name__ == "__main__":
    main()
