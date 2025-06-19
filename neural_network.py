# TODO: 
#   dataloader class to feed the model with x and y
#   calculate deltas

import numpy as np
from core.tools import initialize_weights
from core.activations import ReLU, softmax
from core.tools import linear_classified_data_generator
from core.loss import mse


class Neuron:
    def __init__(self, input_num):
        self.input_weigts = initialize_weights(input_num, w=0.01)
        self.bias = initialize_weights(1, w=0.01)[0]
        
    
    def forward(self, x):
        y = sum([(x_value*weight) for x_value, weight in zip(x, self.input_weigts)]) + self.bias
        return y
    
    
    def get_weights(self):
        return self.input_weigts
    
    
    def get_bias(self):
        return self.bias
    
    
    def __call__(self, x):
        return self.forward(x)



class Layer:
    def __init__(self, input_neurons,  neuron_num):
        self.previous_neuron_num = input_neurons
        self.neurons = [Neuron(input_neurons) for i in range(neuron_num)]
        
        
    def forward(self, x):
        if len(x) != self.previous_neuron_num:
            raise ValueError(f"x must have len {self.previous_neuron_num} but got len {len(x)}")       
        out = [nr.forward(x) for nr in self.neurons]
        return out
    
    
    def get_weights(self):
        # each line of this list shows input weight to a neuron
        out = [nr.get_weights() for nr in self.neurons]
        return out

    
    def __call__(self, x):
        return self.forward(x)
    
   

class NeuralNetwork:
    def __init__(self, layers_activation_pair: list[Layer]):
        self.layers = layers_activation_pair  # leyer, activation function
        self.depth = len(self.layers)
        self.activations = []
         
        
    def forward(self, x):
        self.activations = []
        if len(x) != self.layers[0][0].previous_neuron_num:
            raise ValueError(f"x must have len {self.layers[0][0].previous_neuron_num} but got len {len(x)}")
        
        for layer, activation in self.layers:
            x = activation(layer(x))
            self.activations.append(x)
        return x


    def calculate_deltas(self):
        deltas = []
        # for the output layer:
        # deltas.append([])
        
        # for hidden layers:
        for i in range(self.depth-1, 0, -1):
            print(f"now calculating deltas for layer {i}")
            # deltas.append()
        return


    def get_activations(self):
        return self.activations


    def parameters(self):
        weights = [w[0].get_weights() for w in self.layers]
        biases = []
        return {"weights": weights, "biases": biases}
    
    
    def __call__(self, x):
        return self.forward(x)
    
    
    def __str__(self):
        return f"I'm a model!"


def test_y(y):
    if y == 0:
        return [1, 0]
    return [0, 1]


def train():
    df = linear_classified_data_generator(2, 1, 100)
    y = list(df['class'].values)
    y = [test_y(a) for a in y]   
        
    # print(y)
    X = list(df.drop(columns="class").values)
    
    layers = [[Layer(2, 2), ReLU], [Layer(2, 3), ReLU], [Layer(3, 2), softmax]]
    nn = NeuralNetwork(layers)
    print(50*"-=")
    
    # pprint(nn.parameters())
    
    epoch_num = 10
    
    # for epoch in range(epoch_num):
    out = np.array([nn(x) for x in X[:3]])
    # pprint(out)
    loss = mse(y, out)
    pprint(f"loss for epoch {0} : {loss}")
    print()
    pprint(nn.get_activations())
    
    
    # pprint(out)
    


if __name__ == "__main__":
    from pprint import pprint
    train()
    
    
    # ==============================================================================================
    
    
    # ==============================================================================================
    
