from matplotlib import pyplot as plt
import numpy as np
from core.tools import initialize_weights, linear_classified_data_generator
from core.activations import sigmoid, ReLU
from core.derivatives import sigmoid as d_sigmoid
from core.derivatives import ReLU as d_ReLU
from pprint import pprint
from tqdm import tqdm


class Neuron:
    """
    This class simulates a NEURON. a neuron has some weights and a bias value
    """
    def __init__(self, input_num: int, w: float = 0.01):
        """
        initializes a neuron with specified input numbers and one bias
        
        args:
            input_num: number of input neurons to this neuron. (previous layer neurons for FC layers)
            w: weight for initialize the weighs and bias
        """
        self.input_weigts = initialize_weights(input_num, w)
        self.bias = initialize_weights(1, w)[0]
        
    
    def forward(self, x: list) -> float:
        """
        passes the list of actived outputs from previous layer nodes through the neuron and returns the output
        actually its the WEIGHTED SUM OF INPUTS
        
        args:
            x: list of previous layer actived outputs
        """
        return sum([(x_value*weight) for x_value, weight in zip(x, self.input_weigts)]) + self.bias
    
    
    def get_weights(self) -> list[float]:
        """
        returns a list of neuron's weights
        """
        return self.input_weigts
    
    
    def get_bias(self) -> float:
        """
        returns neurons bias
        """
        return self.bias
    
    
    def __call__(self, x):  # you can call the forward method by object's name
        return self.forward(x)


class Layer:
    def __init__(self, input_neurons: int,  neuron_num: int, activation, activation_differ):
        self.neurons = [Neuron(input_neurons) for i in range(neuron_num)]
        self.input_neurons = input_neurons
        self.activation = activation
        self.activation_differ = activation_differ  # activation function differntion to compute lambdas
        self.lambdas = [0 for _ in range(len(self.neurons))]
        # print(f"initiated self lambdas: {self.lambdas}")
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
        landas = self.lambdas
        self.input_neurons
        
        previous_lambdas = []
        
        # current layer has len(self.neurons) neurons.  corresponded by c
        # previous layer has self.input_neurons nuerons.  corresponded by i
        # print(f"\n this layer has {len(self.neurons)} | range of C | neurons and {self.input_neurons} input neurons | range of I")
        # print(f"length of this layer's lambda = {len(self.lambdas)} | range of ")
    
    
        for c in range(len(self.neurons)):  # c for current layer neurons
            temp_landa = 0
            # print(f"I'm on neuron number = {c+1}")
            for i in range(self.input_neurons):  # i for previous layer neurons
                temp_landa += landas[c] * weights[c][i] * previous_activation_diff(previous_activations[i])
                # print(f"now calculate lambda for neuron {i} in my prevous layer ")
                
                previous_lambdas.append(temp_landa)  # This line should be here not 4 spaces back!!!!!!!!!!!!!!!!!!!!!!
            
        # ---------- important ---------- #
        # print(f"I have {len(self.neurons)} neurons and I calculated my previous layer lambdas as: {previous_landas}")
        # print("=========================================")
        return previous_lambdas  # must be set for previous layer
    
    
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
            self.neurons[i].bias -= self.lambdas[i] * lr
    
    
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
        # for each output neuron:
        #   landa = -2 * error * activation_diff(neuron activation)
        
        output_layer = self.layers[-1]
        y_pred = output_layer.activations
        landas = []  # will be used as output layer landas
        
        for i in range(len(y_true)):  # for each output node (just for one input)
            # for output layer nodes the formula is:
            #   lambda = -2 * error[i] * df(actived[i])
            error = y_true[i] - y_pred[i]  # error
            d_act = output_layer.activation_differ(y_pred[i])  # df(actived[i])
            landa = -2 * error * d_act
            landas.append(landa)
        
        output_layer.lambdas = landas
        
        for i in range(len(self.layers)-2, -1, -1):  # for each layer in nn (from end to start)
            # -2 because the last layer is output and has a different formula
            # for hidde layer nodes the formula is:
            #   lambda[i] = sum(lambda[n] * df(actived[i]) * weights[i -> n])
            #   n is in next layer | i is in current layer
            l = self.layers[i]
            l_next = self.layers[i+1]  # next layer nodes (the n above coresponds to each node in this layer)
            l.lambdas = l_next.compute_pre_landas(l.activations, l.activation_differ)  # current or previous layer nodes ragarding to our logic
            # (the i above coresponds to each node in this layer)
            gradients = l.compute_grades(l_next.lambdas)
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
    # X = [
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1],
    # ]

    # Y = [
    #     [0],
    #     [1],
    #     [1],
    #     [1],
    # ]
    
    # hidden_layer = Layer(input_neurons=2, neuron_num=4, activation=sigmoid, activation_differ=d_sigmoid)
    # output_layer = Layer(input_neurons=4, neuron_num=1, activation=sigmoid, activation_differ=d_sigmoid)

    # net = NeuralNetwork([hidden_layer, output_layer])

    # for epoch in range(60000):
    #     total_loss = 0
    #     for x, y_true in zip(X, Y):
    #         y_pred = net(x)
    #         net.backward(y_true)
    #         loss = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
    #         total_loss += loss
    #     if epoch % 100 == 0:
    #         print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # for x in X:
    #     pred = net(x)
    #     print(f"Input: {x}, Predicted: {pred}")
        
    # /=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=
    print()
    print("=== preprocessing ===")
    print()
    
    df_data = linear_classified_data_generator(slope=2, intercept=5, n_samples=600, plot=False)
    X = df_data[['x1', 'x2']].values.tolist()
    Y = df_data[['class']].values.tolist()
    
    train_percent = 0.9
    valid_percent = 0.1
    
    X_train = X[:int(train_percent*len(X))]
    X_test = X[int(train_percent*len(X)):]
    
    Y_train = Y[:int(train_percent*len(X))]
    Y_test = Y[int(train_percent*len(X)):]
    
    # # pprint(X_train[:10])
    # # pprint(Y_train[:10])

    la = [
        Layer(input_neurons=2, neuron_num=8, activation=sigmoid, activation_differ=d_sigmoid),
        Layer(input_neurons=8, neuron_num=4, activation=sigmoid, activation_differ=d_sigmoid),
        Layer(input_neurons=4, neuron_num=1, activation=sigmoid, activation_differ=d_sigmoid),
        ]

    net = NeuralNetwork(la, lr=0.01)
    
    total_loss = 0
    print()
    print("=== training ===")
    print()
    # for epoch in tqdm(range(5000), desc="training"):
    for epoch in range(600):
        for x, y_true in zip(X_train, Y_train):
            y_pred = net(x)
            net.backward(y_true)
            loss = sum([(y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))])
            total_loss += loss
        
        print(f"Epoch {epoch}, Training Loss: {total_loss/len(X_train):.4f}")
        total_loss = 0

    correct = 0
    
    for x, y_true in zip(X_test, Y_test):
        pred = net(x)
        # print(f"Input: {x}, Predicted: {1 if pred[0] > 0.5 else 0} | real output: {y_true}")
        print(f"Predicted: {1 if pred[0] > 0.5 else 0} | target: {y_true[0]}")
        p = 1 if pred[0] > 0.5 else 0
        if p == y_true[0]:
            correct += 1
    print()
    print("=== validating ===")
    print()
    print(f"number of test samples: {len(X_test)}")
    print(f"accuracy of model: {correct/len(X_test)}")
        
