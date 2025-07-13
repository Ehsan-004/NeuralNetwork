from pprint import pprint
from core.tools import initialize_weights
from core.loss import CCE
from typing import Callable
from core.tools import to_one_hot


class Neuron:
    """
    This class simulates a NEURON. a neuron has some weights and a bias value
    """
    def __init__(self, input_num: int, w: float):
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
            
        returns:
            a not activated float value
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
    """
    This class simulates a LAYER in a neural network that consists of some Neurons.
    """
    def __init__(self, input_neurons: int,  neuron_num: int, activation: Callable, activation_differ: Callable, w=0.01, lr=0.1):
        """
        initializes a layer of neurons with specified input neuron numbers which actually is previous layer neurons number
        
        args:
            input_neurons: number of input neurons to this neuron. (previous layer neurons for FC layers)
            neuron_num: number of neurons for the layer
            activation: the activation function for this layer
            activation_differ: differantiate of current layer's activation function
            w: weight for initialize the weighs and bias for each neuron. change it according to layer position
            lr: learning rate for this layer
        """
        self.neurons = [Neuron(input_neurons, w=w) for _ in range(neuron_num)]  # initialize neurons
        self.input_neurons_num = input_neurons
        self.activation = activation
        self.activation_differ = activation_differ  # activation function differntion to compute lambdas
        self.lambdas = [0 for _ in range(len(self.neurons))]  # lambdas will be used later to update weights
        self.activations = []  # this is actually outputs of this layer after passing from activation function. will be used to update weights.
        self.lr = lr
        
    
    
    def forward(self, x: list) -> list:
        """
        passes the list of inputs from previous layer through the neuron and returns the outputs as a list
        
        args:
            x: list of inputs from previous layer, must have the same length as previous layer neurons
        
        returns:
            list[float]: a list of activated outputs
        """
        if len(x) != self.input_neurons_num:
            raise ValueError(f"input to this layer must have len {self.input_neurons_num} but got len {len(x)}")
        
        out = [nr.forward(x) for nr in self.neurons]  # not activated outputs
        self.activations = self.activation(out)  # activating outputs
        return self.activations
    
    
    def compute_pre_lambdas(self, previous_activations: list[float], previous_activation_diff: Callable) -> list[float]:
        """
        calculates previous layer lambdas. read docs for NeuralNetwork.backward for more details ...
        
        args:
            previous_activation: a list of previous layer's activations
            previous_activation_diff: this is actually the differantion of the previous layer activation function. 
                example: f(x)=3x    --->   df(x) = 3
                
        returns:
            list[float]: a list of previous layer lambdas (of course it has len of previous layer nodes!)
        """
        
        weights = self.get_weights()
        landas = self.lambdas
        self.input_neurons_num
        
        previous_lambdas = []
        
        # current layer has len(self.neurons) neurons.  corresponded by c
        # previous layer has self.input_neurons nuerons.  corresponded by i
    
        for i in range(self.input_neurons_num):  # i for previous layer neurons
            temp_landa = 0
            for c in range(len(self.neurons)):  # c for current layer neurons
                temp_landa += landas[c] * weights[c][i] * previous_activation_diff(previous_activations[i])
                
            previous_lambdas.append(temp_landa)
                # the line above didn't have the necessary indent and was a tab further back
            
        # ---------- important ---------- #
        return previous_lambdas  # must be set for previous layer to update weights
    
    
    def compute_grades(self, next_layer_lambdas: list) ->list[list[float]]:
        """
        calculates gradients for weights between current layer and next layer
        (actually the next layer weights because weights belong to layer they go throug)
        they will be used to update weights in Layet.update_weights()
        
        args:
            next_layer_lambdas: comes from Layer.compute_pre_lambdas() (read the comment on next line of that method)
        
        returns:
            list[list[float]]: read the comments below. 
        """
        
        # because we keep the weight with the layer which they go to, so this list should have the len of next layer nodes 
        # (bacause it is about weights from current layer to next layer)
        # obviously each list has len of current layer's nodes
        next_l_neuron_num = len(next_layer_lambdas)
        gradients = [[0 for __ in range(len(self.neurons))] for _ in range(next_l_neuron_num)]
        activations = self.activations
        
        for c in range(len(self.neurons)):  # c for current layer neurons
            for j in range(next_l_neuron_num):  # j for next layer neurons
                gradients[j][c] = next_layer_lambdas[j] * activations[c]  # g[j][c] means weight from node[c] in this layer to node[j] in next layer
                
        return gradients
    
    
    def update_weights(self, gradients: list[list[float]]) -> None:
        """
        gets the gradient between current layer and previous layer and updates weights
        
        args:
            gradients: a list which has length of len(self.neurons) and each list inside it has length of previous layer neurons
        """
        
        lr = self.lr
        for c in range(len(self.neurons)):  # c for current layer neurons
            for i in range(self.input_neurons_num):  # i for previous layer neurons
                self.neurons[c].input_weigts[i] -= gradients[c][i] * lr  # this is the Gradient Descent for weights
            self.neurons[c].bias -= self.lambdas[c] * lr  # Gradient Descent for bias
    
    
    def get_weights(self) -> list[list[float]]:
        """
        returns this layer's weights
        
        returns:
            list[list[float]]: a list which has length of len(self.neurons) and each list inside contains input weights to a neuron of this layer
        """
        out = [nr.get_weights() for nr in self.neurons]
        return out


class AbstractInputNeuron:
    def __init__(self):
        pass
        
    def forward(self, x: list) -> float:
        return x
           
    def __call__(self, x):
        return self.forward(x)


class AbstractInputLayer:
    def __init__(self,  neuron_num: int, lr=0.1):
        self.neurons = [AbstractInputNeuron() for _ in range(neuron_num)]
        self.activations = []
        self.lr = lr
    
    def forward(self, x):
        self.activations = x
        return x

    def compute_grades(self, next_layer_lambdas: list) ->list[list[float]]:
        next_l_neuron_num = len(next_layer_lambdas)  # first hidden layer gradients
        gradients = [[0 for __ in range(len(self.neurons))] for _ in range(next_l_neuron_num)]
        activations = self.activations
        
        for c in range(len(self.neurons)):
            for j in range(next_l_neuron_num):
                gradients[j][c] = next_layer_lambdas[j] * activations[c]
                
        return gradients

class NeuralNetwork:
    """
    This class simulates a NEURAL NETWORK consists of some Layers
    """
    def __init__(self, layers: list[Layer], loss: str = "mse"):
        """
        initializes a Neural Network with specified layers and learning rate
        
        args:
            layers: a list of Layers
            loss ("mse", "cce"): name of loss function to use
        """
        self.layers = layers
        self.loss_type = loss
        self.input_layer = AbstractInputLayer(layers[0].input_neurons_num)
        self.gradents = [None for _ in range(len(self.layers))]  # 1 for abstract input layer
        
        # print(f"g is: {self.gradents}")
        # self.gradents = []
        
        # if you want to have the same lr for all layers then uncomment this and add lr as an arg!
        # self.lr = lr
        # for l in self.layers:
        #     l.lr = lr
        
    def forward(self, x: list) -> list:
        """
        passes the input data to network
        
        args:
            x: input to the network according to first layers input numbers (always a list)
        
        returns:
            output of network according to number of neurons in the last layers (has its length)
        """
        if len(x) != self.layers[0].input_neurons_num:
            raise ValueError(f"x must have len {self.layers[0].input_neurons_num} but got len {len(x)}")
        
        self.input_layer.forward(x)
        
        for layer in self.layers:
            x = layer.forward(x)
        return x


    def backward(self, y_true: list) -> None:
        """
        implements backward pass process. after forward passing data into network, its time to backward pass
        in backward pass gradients which actually are dL/dw and dL/db will be calculated and then weights are updated
        read formula and How To Compute in comments!
        
        args:
            y_true: target output of data sample. must be a list
            
        note:
            if you want to solve a binary class problem use "mse" loss
            if you want to solve a classification problem with more than two classes, use "cce" loss. 
            y_true in "cce" loss must be a ONE-HOT list 
        """
        # for each output neuron:
        #   landa = -2 * error * activation_diff(neuron activation)
        
        output_layer = self.layers[-1]  # this is the last layer and has a different method to compute gradients
        y_pred = output_layer.activations
        lambdas = [0 for  _ in range(len(y_true))]  # will be used as output layer lambdas
        
        self.loss = 0
        
        # calculate lambda for output layer
        for i in range(len(y_true)):  # i corresponds to each output node (just for one sample of data)
            if self.loss_type == "mse":
                # for output layer nodes the formula to compute lambda is:
                #   lambda = -2 * error[i] * df(actived[i])
                error = y_true[i] - y_pred[i]  # error
                self.loss += error ** 2
                d_act = output_layer.activation_differ(y_pred[i])  # df(actived[i])
                landa = -2 * error * d_act  # this is the derivation of loss function with respect to outputs 
                # (MSELoss(predict, target) = sum([(t-p)**2) for t, p in zip(predicts, targets)])
                # lambdas.append(landa)
                lambdas[i] = landa
            elif self.loss_type == "cce":
                y_pred_one_hot = to_one_hot(y_pred)
                self.loss += CCE(y_true, y_pred_one_hot)
                # the formula to calculate dl/dw for CCE is:
                #   dCCE/dw = predicted_for_node[i] - target_for_node[i]
                lambdas[i] = y_pred[i] - y_true[i]
                
                
        output_layer.lambdas = lambdas

                
        # print(self.gradents)
        # print()
        
        # updating output layer weights
        output_gradients = self.layers[-2].compute_grades(output_layer.lambdas)
        
        
        # print(f"I'm trying to get the index {len(self.layers) - 1}")
        self.gradents[len(self.layers) - 1] = output_gradients
        
        # x = 1
        # print(f"seri {x}")
        # print(f"from {len(output_gradients[0])} -> {len(output_gradients)}")
        # print(output_gradients)
        # print(self.gradents)
        # print()
        # x += 1
        
        # output_layer.update_weights(output_gradients)  # =============================================================================
        
        # updating hidden layers' weights
        for i in range(len(self.layers)-2, -1, -1):  # for each layer in nn (from end to start except last layer)
            # print(f"now i is:{i}")
            # -1 because the last layer is output and has a different formula (and another -1 for indexing in python)
            # for hidde layer nodes the formula to compute lambda is:
            #   lambda[i] = sum(lambda[n] * df(actived[i]) * weights[i -> n])
            #   n is in next layer | i is in current layer
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]  # next layer nodes (the n above coresponds to each node in this layer)
            current_layer.lambdas = next_layer.compute_pre_lambdas(current_layer.activations, current_layer.activation_differ)  # current or previous layer nodes ragarding to our logic | read Layer.compute_pre_lambdas() comments
            # (the i above coresponds to each node in this layer)
            # be sure to read doc fot these methods
            gradients = current_layer.compute_grades(next_layer.lambdas)
            
            self.gradents[i+1] = gradients
            
            # print(f"{len(current_layer.neurons)} ===> {len(next_layer.neurons)}")
            
            # print(f"seri {x}")
            # print(f"from {len(gradients[0])} -> {len(gradients)}")
            # print(gradients)
            # print(self.gradents)
            # print()
            # x += 1
            # next_layer.update_weights(gradients)  # =============================================================================

        # updating first layer weights
        first_hidden_gradients = self.input_layer.compute_grades(self.layers[0].lambdas)
        self.gradents[0] = first_hidden_gradients
        # print(f"seri {x}")
        # print(f"from {len(first_hidden_gradients[0])} -> {len(first_hidden_gradients)}")
        # print(first_hidden_gradients)
        # print(self.gradents)
        # print()
        # x += 1
        
        # self.layers[0].update_weights(first_hidden_gradients) # =============================================================================


    def step(self):
        # print("0---------------------------0")
        # for gr in self.gradents:
        #     print(f"from {len(gr)} to {len(gr[0])}")
        #     print()
        
        # for i in range(len(self.gradents)):
        #     print(f"this is gradient for weights from layer with {len(self.gradents[i][0])} to {len(self.gradents[i])}")
        # exit()
        
        # g = list(reversed(self.gradents))
        for i in range(len(self.layers)):
            self.layers[i].update_weights(self.gradents[i])
            # self.layers[i+1].update_weights(g[i])
        self.gradents = [None for _ in range(len(self.layers) )]


    def get_weights(self) -> list[list[list[float]]]:
        """
        returns network weights
        
        returns:
            list[list[list[float]]]: same as Layer.get_weights() but has a layers dimension! (I mean a line for each layer ...)
        """
        return [l.get_weights() for l in self.layers]
    
    
    def __call__(self, x):
        return self.forward(x)

