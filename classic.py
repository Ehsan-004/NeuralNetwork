import numpy as np
from core.activations import linear, ReLU
from core.loss import sqe, msq
from core.derivatives import d_sqe_w, d_sqe_b


# class Neuron:
#     def __init__(self):
#         self.weight = abs(np.random.randn()) ; print(f"weight = {self.weight}")
#         self.bias = np.random.randn() ;        print(f"bias = {self.bias}")
        
#     def set(self, x: int, y: int, activation):
#         self.activation = activation
#         self.x = x
#         self.y = y
        
#     def forward(self):
#         print(f"w = {self.weight} | x = {self.x} | b = {self.bias}")
#         output = self.activation(self.weight*self.x + self.bias)
#         print(f"output = {output}")
#         return output
          
          
def initialize_weights():
    weight = abs(np.random.randn()) ; print(f"weight = {weight}")
    bias = np.random.randn() ;        print(f"bias = {bias}")
    return weight, bias
          
          
def forward(weight, x, bias, activation):
    output = activation(weight*x + bias)
    return output
        
            
    
def main():    
    # batches = [[1,2],[3,4],[5,6],[7,8],[9,10]]
    # y_batches = [[3,5],[7,9],[11,13],[15,17],[19,21]]
    
    batches = np.array(range(1,20001)).reshape(-1, 2)
    y_batches = np.array(range(3, 40003)).reshape(-1, 2)
    
    batches = (batches - np.mean(batches)) / np.std(batches)
    y_batches = (y_batches - np.mean(y_batches)) / np.std(y_batches)
    
    learn_rate = 0.00001
    
    # n = Neuron()
    # n.forward()
    
    w, b = initialize_weights()
    
    for i in range(len(batches)):
        
        outputs = []
        x = batches[i]
        y = y_batches[i]
        
        for j in range(len(x)):
            o = forward(w, x[j], b, linear)
            outputs.append(o)
        
        
        msq_error = msq(outputs, y)
        sqe_error = sqe(outputs, y)
        
        dl_dw = d_sqe_w(x, outputs, y) 
        dl_db = d_sqe_b(x, outputs, y)
        w -= learn_rate*dl_dw
        b -= learn_rate*dl_db
        # print(f"weight updated | new weight = {w}")
        # print(f"bias updated | new bias = {b}")
        
        
        # print(f"sqe = {msq_error}")
        # print(f"sqe = {sqe_error}")
        
        if i % 10 == 0:
            print(f"[{i}] loss = {msq_error:.4f} | w = {w:.4f} | b = {b:.4f}")
            
        if msq_error < 0.05:
            return
        


if __name__ == "__main__":
    main()
