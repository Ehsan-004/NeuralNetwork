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

## Prepare appropriate data:
Your data must be in this template:
```python
X_train = [[x11, x12], [x21, x22]]
Y_train = [[y1], [y2]]
```
X_train is a list of data samples features. A datasample can be 1D or nD, anyway it should be in a list like I've definde above. </br>
Y_train is also the same as X_train. It should be a list containing lists of datasample outputs each one in a list like above.

</br>

## Define a model:
Import class ```Layer``` from ```neural_network.py``` and then create a list and define layers for the network.</br>
To create a layer you have these options to be passed:</br>
```python
Layer(input_neurons, neuron_num, activation, activation_differ, w=0.1, lr=1)
``` 

Read about details on ```Layer``` [docs](neural_network.py) line 53.</br>

Then it's time to create the network or model. To create a network you should just pass the layers as a list to it:</br>
```python
net = NeuralNetwork(layers)
```
</br>

## Train the model:
To train the model you should pass the loop of train. See a completed sample here:</br>
By the way I used stochastic gradient descent here but I'll add mini batches later ...</br>

```python
for epoch in range(epochs):
    for x, y_true in zip(X_train, Y_train):
        y_pred = net(x)
        net.backward(y_true)
```

You can also get the loss by ```net.loss```. This property is the loss just for one datapoint that goes into network. You can see loss for each time passing whole data batch into network:

```python
for epoch in range(150):
    total_loss = 0  # for one batch or one time passing datapoints into network
    for x, y_true in zip(X_train, Y_train):
        y_pred = net(x)
        net.backward(y_true)
        total_loss += net.loss  # net.loss is the loss for one datapoint: (x, y_true)
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Training Loss: {total_loss/len(X_train):.4f}")
```

It will absolutely be easier for you to read and run this code if you are familier with mathmatics which is used in a neural network


## 🧑‍💻 Developer

- [Ehsan-004](https://github.com/Ehsan-004)

## 📜 License

This project is open-source and does not have a specific license. Feel free to use, modify, and distribute it as you see fit.