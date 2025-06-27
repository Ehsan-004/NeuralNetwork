from matplotlib import pyplot as plt
import numpy as np
from core.tools import initialize_weights, linear_classified_data_generator
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
        self.lr = 0.1
    
    
    def forward(self, x):
        if len(x) != self.input_neurons:
            raise ValueError(f"input to this layer must have len {self.input_neurons} but got len {len(x)}")
        
        out = [nr.forward(x) for nr in self.neurons]
        self.activations = self.activation(out)
        return self.activations
    
    
    def compute_pre_landas(self, previous_activations, previous_activation_diff):
        # pre act diff is a functions
        # pre acts is a list
        """
        calculates previous layer landas
        """
        
        weights = self.get_weights()
        landas = self.landas
        self.input_neurons
        
        previous_landas = []
        
        for i in range(len(self.neurons)):  # j for current layer neurons
            temp_landa = 0
            for j in range(self.input_neurons):  # i for previous layer neurons
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
    
    
    def update_weights(self, gradients):
        lr = self.lr
        for i in range(len(self.neurons)):
            for j in range(self.input_neurons):
                self.neurons[i].input_weigts[j] -= gradients[i][j] * lr
            self.neurons[i].bias -= self.landas[i] * lr
    
    
    def get_weights(self):
        # each line of this list shows input weights to a neuron
        out = [nr.get_weights() for nr in self.neurons]
        return out



class NeuralNetwork:
    def __init__(self, layers: list[Layer], lr=1):
        self.layers = layers
        self.lr = lr
        for l in self.layers:
            l.lr = lr
        
    def forward(self, x):
        if len(x) != self.layers[0].input_neurons:
            raise ValueError(f"x must have len {self.layers[0].input_neurons} but got len {len(x)}")
        
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def backward(self, y_true):
        # backward for output layer:
        # TODO ...
        # for each output neuron:
        #   landa = -2 * error * activation_diff(neuron activation)
        
        output_layer = self.layers[-1]
        y_pred = output_layer.activations
        landas = []
        
        for i in range(len(y_true)):
            error = y_true[i] - y_pred[i]
            d_act = output_layer.activation_differ(y_pred[i])
            landa = -2 * error * d_act
            landas.append(landa)
        
        output_layer.landas = landas
        
        for i in range(len(self.layers)-2, -1, -1):  # -2 because the last layer is output and is different
            l = self.layers[i]
            l_next = self.layers[i+1]
            l.landas = l_next.compute_pre_landas(l.activations, l.activation_differ)
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
    
    # l = Layer(2,3, ReLU, d_ReLU)  # next layer has two nuerons
    # l.forward([10, 11])
    # w = [[0.005007011765916471, -0.06692413065941487, -0.02683456361965499],[-0.03470931602341219, 0.0027481619278391808, 0.1626835551191007]]
    # pprint(l.get_weights())
    # l.backward_pass([2,3])
    
    # /=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
    
    # test part written by Chat GPT:
    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    Y = [
        [0],
        [1],
        [1],
        [1],
    ]
    
    hidden_layer = Layer(input_neurons=2, neuron_num=4, activation=sigmoid, activation_differ=d_sigmoid)
    output_layer = Layer(input_neurons=4, neuron_num=1, activation=sigmoid, activation_differ=d_sigmoid)

    net = NeuralNetwork([hidden_layer, output_layer])

    for epoch in range(60000):
        total_loss = 0
        for x, y_true in zip(X, Y):
            y_pred = net(x)
            net.backward(y_true)
            loss = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
            total_loss += loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    for x in X:
        pred = net(x)
        print(f"Input: {x}, Predicted: {pred}")
        
    # /=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
        
