from core.tools import initialize_weights


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
        
        out = []
        for nr in self.neurons:
            out.append(nr.forward(x))
        
        return out


if __name__ == "__main__":
    # ne = Neuron(1)
    # print(f"output weights: {ne.input_weigts}")
    # print(f"bias: {ne.bias}")
    
    # y = ne.forward([1,1,1])
    # print(f"y: {y}")
    
    le = Layer(2, 3)
    print(le.forward([2,2]))
