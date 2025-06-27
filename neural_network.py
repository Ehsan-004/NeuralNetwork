from core.tools import initialize_weights
from core.activations import sigmoid, ReLU
from core.derivatives import sigmoid as d_sigmoid
from core.derivatives import ReLU as d_ReLU
from pprint import pprint


class Neuron:
    def __init__(self, input_num):
        self.input_weigts = initialize_weights(input_num, w=0.1)
        self.bias = initialize_weights(1, w=0.1)[0]
        
    
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
    def __init__(self, input_neurons,  neuron_num, activation, activation_differ):
        self.neurons = [Neuron(input_neurons) for i in range(neuron_num)]
        self.input_neurons = input_neurons
        self.activation = activation
        self.activation_differ = activation_differ
        self.landas = [0 for _ in range(len(self.neurons))]
        self.activations = []
    
    
    def forward(self, x):
        if len(x) != self.input_neurons:
            raise ValueError(f"input to this layer must have len {self.input_neurons} but got len {len(x)}")
        
        out = [nr.forward(x) for nr in self.neurons]
        self.activations = self.activation(out)
        return self.activations
    
    
    def compute_pre_landas(self, previous_activations, previous_activation_diff):
        """
        calculates previous layer landas
        """
        
        weights = self.get_weights()
        landas = self.landas
        self.input_neurons
        
        previous_landas = []
        
        for i in range(len(self.neurons)):  # j for current layer neurons
            temp_landa = 0
            for j in range(len(self.input_neurons)):  # i for previous layer neurons
                temp_landa += landas[i] * weights[i][j] * previous_activation_diff(previous_activations[j])
            previous_landas.append(temp_landa)
            
        # ---------- important ---------- #
        return previous_landas  # must be set for previous layer
    
    
    def compute_grades(self, next_layer_landas):
        """
        calculates gradients for weights between current layer and next layer
        """
        next_l_neuron_num = len(next_layer_landas)
        activations = self.activations
        
        gradients = [[0 for __ in range(len(self.neurons))] for _ in range(next_l_neuron_num)]
        
        for i in range(len(self.neurons)):
            for j in range(next_l_neuron_num):
                gradients[j][i] = next_layer_landas[j] * activations[i]
                
        return gradients
    
    
    def update_weights(self, gradients, lr=0.01):
        for i in range(len(self.neurons)):
            for j in range(self.input_neurons):
                self.neurons[i].input_weigts[j] -= gradients[i][j] * lr
            self.neurons[i].bias -= self.landas[i] * lr
    
    
    def get_weights(self):
        # each line of this list shows input weights to a neuron
        out = [nr.get_weights() for nr in self.neurons]
        return out



class NeuralNetwork:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
         
        
    def forward(self, x):
        if len(x) != self.layers[0].input_neurons:
            raise ValueError(f"x must have len {self.layers[0].input_neurons} but got len {len(x)}")
        
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def backward(self):
        # backward for output layer:
        # TODO ...
        
        for i in range(len(self.layers)-2, -1, -1):  # -2 because the last layer is output and is different
            l = self.layers[i]
            l_next = self.layers[i+1]
            l.landas = self.layers[i+1].compute_pre_landas(l.activation, l.activation_differ)
            gradients = l.compute_grades(l_next.landas)
            l_next.update_weights(gradients)
            
        pass
    
    
    def get_weights(self):
        return [l.get_weights() for l in self.layers]
    
    
    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":
    # l = Layer(3, 2, sigmoid, d_sigmoid)
    # pprint(l.get_weights())
    # print()
    # l.forward([1,2,3])
    # print()
    # pprint(l.activations)
    
    # /=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
    
    # layers = [Layer(2, 2, ReLU, d_ReLU),  Layer(2, 3, ReLU, d_ReLU), Layer(3, 2, sigmoid, d_sigmoid)]
    # nn = NeuralNetwork(layers)
    # print(50*"-=")
    # X = [[10,20], [30,40], [50,60], [70,80]]
    # out = [nn(x) for x in X]
    # pprint(out)
    # print()
    # ws = nn.get_weights()
    # for i in range(len(layers)):
    #     print(f"layer number {i+1}")
    #     pprint(ws[i])
    #     print()
    
    # /=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
    
    l = Layer(2,3, ReLU, d_ReLU)  # next layer has two nuerons
    l.forward([10, 11])
    w = [[0.005007011765916471, -0.06692413065941487, -0.02683456361965499],[-0.03470931602341219, 0.0027481619278391808, 0.1626835551191007]]
    pprint(l.get_weights())
    l.backward_pass([2,3])
    






def backward_pass(self, next_layer_landas, weights):  # weights from this layer to next layer
    der = self.activation_differ(self.activations)
    
    for i in range(len(self.neurons)):  # for each noruon in this layer 
        d_activation = der[i]
        
        for j in range(len(next_layer_landas)):  # for each neuron in the next layer | next layer landas = next layer neurons
            self.landas[i] += weights[j][i] * d_activation * next_layer_landas[j]
            
    d_l_w = [landa * activation for landa, activation in zip(self.landas, self.activations)]  # gradients 
    d_l_b = self.landas
    
    
    # weights_gradients = [[] for _ in range(len(next_layer_landas))]
    # for landa in self.landas:
    #     for activation in next_layer_activations:
    #         pass
    
    
    pprint(f"got the landas")
    pprint(self.landas)
    print()
    print("this is dl dw")
    pprint(d_l_w)
    print()
    print("this is dl db")
    pprint(d_l_b)