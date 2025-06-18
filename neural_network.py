from core.tools import initialize_weights
from core.activations import ReLU, softmax


class Neuron:
    def __init__(self, input_num):
        self.input_weigts = initialize_weights(input_num)
        self.bias = initialize_weights(1)[0]
        
    
    def forward(self, x):
        y = sum([(x_value*weight) for x_value, weight in zip(x, self.input_weigts)]) + self.bias
        return y


class InputNeuron(Neuron):
    def __init__(self, output_num):
        super().__init__(output_num)


class Layer:
    def __init__(self, previous_layer_neurons_num,  neuron_num):
            
        self.previous_neuron_num = previous_layer_neurons_num
        self.neurons = [Neuron(previous_layer_neurons_num) for i in range(neuron_num)]
        
        # print(f"this layer has {len(self.neurons)} neurons!")
        # for j in range(len(self.neurons)):
        #     print(f"neuron number {j+1} | {self.neurons[j].input_weigts}")
        
    def forward(self, x):
        if len(x) != self.previous_neuron_num:
            raise ValueError(f"x must have len {self.previous_neuron_num} but got len {len(x)}")
        
        out = [nr.forward(x) for nr in self.neurons]
        return out
    
    
    
    
# class InputLayer:
    # def __init__(self,  neuron_num):
    #     self.neuron_num = neuron_num
    #     # self.neurons = [Neuron(previous_layer_neurons_num) for i in range(neuron_num)]
    #     self.activation = activation_function
        
    #     # print(f"this layer has {len(self.neurons)} neurons!")
    #     # for j in range(len(self.neurons)):
    #     #     print(f"neuron number {j+1} | {self.neurons[j].input_weigts}")
        
    # def forward(self, x):
    #     if len(x) != self.previous_neuron_num:
    #         raise ValueError(f"x must have len {self.previous_neuron_num} but got len {len(x)}")
        
    #     out = []
    #     for nr in self.neurons:
    #         out.append(self.activation(nr.forward(x)))
        
    #     return out


    


if __name__ == "__main__":
    # ne = Neuron(1)
    # print(f"output weights: {ne.input_weigts}")
    # print(f"bias: {ne.bias}")
    
    # y = ne.forward([1,1,1])
    # print(f"y: {y}")
    
    le1 = Layer(10, 10)
    le2 = Layer(10, 5)
    le3 = Layer(5, 2)
    
    i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(i)
    x = le1.forward(i)
    xa = ReLU(x)
    print(f"x before active: {x}")
    print(f"x after active: {xa}")
    y = le2.forward(x)
    ya = ReLU(y)
    print(f"y before active: {y}")
    print(f"y after active: {ya}")
    z = le3.forward(y)
    za = softmax(z)
    print(f"z before active: {z}")
    print(f"z after active: {za}")
    print(sum(za))
