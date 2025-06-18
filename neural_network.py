from core.tools import initialize_weights
from core.activations import ReLU, softmax


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
    
    
    
class NeuralNetwork:
    def __init__(self, layers_activation_pair: list[Layer]):
        self.layers = layers_activation_pair  # leyer, activation function
        self.depth = len(self.layers)
         
        
    def forward(self, x):
        if len(x) != self.layers[0][0].previous_neuron_num:
            raise ValueError(f"x must have len {self.layers[0].previous_neuron_num} but got len {len(x)}")
        
        self.activations = []
        
        for layer, activation in self.layers:
            x = activation(layer.forward(x))
            self.activations.append(x)
        return x


    def calculate_deltas(self):
        deltas = []
        for i in range(self.depth, 0, -1):
            deltas.append()
        return


    def get_activations(self):
        return self.activations

if __name__ == "__main__":
    from pprint import pprint
    # ne = Neuron(1)
    # print(f"output weights: {ne.input_weigts}")
    # print(f"bias: {ne.bias}")
    
    # y = ne.forward([1,1,1])
    # print(f"y: {y}")
    
    # ==============================================================================================
    
    # le = Layer(2, 3)
    # a = le.get_weights()
    # pprint(a)
    
    # ==============================================================================================
    
    i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    layers = [[Layer(10, 10), ReLU], [Layer(10, 5), ReLU], [Layer(5, 2), softmax]]
    nn = NeuralNetwork(layers)
    print(nn.forward(i))
    print(50*"-=")
    
    # print(i)
    # x = layers[0][0].forward(i)
    # xa = ReLU(x)
    # print(f"x before active: {x}")
    # print(f"x after active: {xa}")
    # y = layers[1][0].forward(x)
    # ya = ReLU(y)
    # print(f"y before active: {y}")
    # print(f"y after active: {ya}")
    # z = layers[2][0].forward(y)
    # za = softmax(z)
    # print(f"z before active: {z}")
    # print(f"z after active: {za}")
    # print(sum(za))
