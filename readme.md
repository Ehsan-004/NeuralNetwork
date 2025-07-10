# Neural Network

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/Ehsan-004/NeuralNetwork)
[![License](https://img.shields.io/badge/License-OpenSource-green)](https://github.com/Ehsan-004/NeuralNetwork/blob/main/LICENSE)
[![Developer](https://img.shields.io/badge/Developer-Ehsan--004-purple?logo=github)](https://github.com/Ehsan-004)

## Implemention of a neural network from scratch! </br>
In this repository I implemented a neural network from scratch. I am very excited about this and it was verrrry eager to see it works and now it's done!</br>
</br>

## 📝 Description
a simple implemention of a neural network.</br>
also simple implemention of a single neuron classification and regression models.
</br>


# To use:
## Define a model:
Import class ```Layer``` from ```neural_network.py``` and then create a list and define layers for the network.</br>
To create a layer you have these options to be passed:</br>
```python
Layer(input_neurons, neuron_num, activation, activation_differ, w=0.1, lr=1)
```
</br>
Read about details on ```Layer``` [docs](neural_network.py) line 53.</br>


Then it's time to create the network. To create a network you should just pass the layers as a list to it:</br>
```python
net = NeuralNetwork(layers)
```

To train the model you should pass the loop of train. See a completed sample here:</br>
By the way I used stochastic gradient descent here but I'll add mini batches later ...</br>

```python
X_train = [[x1], [x2]]
Y_train = [[y1], [y2]]

for epoch in range(epochs):
    for x, y_true in zip(X_train, Y_train):
        y_pred = net(x)
        net.backward(y_true)
```

It will absolutely be easier for you to read and run this code if you are familier with mathmatics which is used in a neural network


## 🧑‍💻 Developer

- [Ehsan-004](https://github.com/Ehsan-004)

## 📜 License

This project is open-source and does not have a specific license. Feel free to use, modify, and distribute it as you see fit.