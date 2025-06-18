import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.activations import sigmoid
from core.loss import msq
from math import ceil, floor
from core.tools import initialize_weights, linear_classified_data_generator


def forward(weight1, weight2,  x1, x2, bias, activation):
    return activation(weight1*x1 + weight2*x2 + bias)



def dl_dw_linear(xs: list, weights, class_predicted, true_class, loss_function, bias):
    """
    derivative of loss with respect to weight number weight number
    """
    dl_ws = []
    for w in range(len(weights)):
        res = 0
        for j in range(len(xs[0])):
            error = true_class[j] - class_predicted[j]
            sig = loss_function(weights[0] * xs[0][j] + weights[1] * xs[1][j] + bias)
            x = xs[w][j]
            res += error * (sig * (1-sig)) * x
        dl_ws.append(-2 * res)
    return dl_ws


def dl_db_linear(xs: list, weights, class_predicted, true_class, loss_function, bias):
    dl_bs = []
    for i in range(2):
        error = true_class[i] - class_predicted[i]
        z = weights[0] * xs[0][i] + weights[1] * xs[1][i] + bias
        sig = loss_function(z)
        res = ((sig * (1-sig)) * (error)) 
        dl_bs.append(-2 * res)
    return sum(dl_bs)


def linear_classification(dataset, epochs=500, learning_rate = 0.001):
    w1, w2, b = initialize_weights(3)
    
    print(f"Initial w1 = {w1:.4f}")
    print(f"Initial w2 = {w2:.4f}") 
    print(f"Initial b  = {b:.4f}\n")
            
    x1_batch = dataset["x1"]
    x2_batch = dataset["x2"]
    classes = dataset["class"]
    
    class_batch = list(dataset["class"])
   
    print("starting training")
   
    for epoch in range(epochs):
        output = forward(w1, w2, np.array(x1_batch), np.array(x2_batch), b, sigmoid)
        dl_dws = dl_dw_linear([x1_batch, x2_batch], [w1, w2], output, class_batch, sigmoid, b)
        dl_dbs = dl_db_linear([x1_batch, x2_batch], [w1, w2], output, class_batch, sigmoid, b)
        w1 -= learning_rate * dl_dws[0]
        w2 -= learning_rate * dl_dws[1]
        b -= learning_rate * dl_dbs
        
        loss = msq(classes, output)
        
        if (epoch + 1) % 10 == 0 or epoch == 0: 
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.8f}, w1: {w1:.6f}, w2: {w2:.6f}, Current b: {b:.6f}")
        
    
    
    return w1, w2, b



# This function is created by chatGPT
def predict_class(x1, x2, w1, w2, b, threshold=0.5):
    prob = forward(w1, w2, x1, x2, b, sigmoid)
    predicted_class = 1 if prob >= threshold else 0
    return prob, predicted_class


if __name__ == "__main__":
    dataset = linear_classified_data_generator(slope=3, intercept=5, plot=True)
    dataset["class"] = dataset["class"].astype(int)
    
    train_data_len = int(len(dataset) * 0.8)
    train_data = dataset.iloc[:train_data_len]
    
    test_data = dataset.iloc[train_data_len:]
    
    w1, w2, b = linear_classification(train_data)
    print("training finished")
    print(f"final w1 = {w1}")
    print(f"final w2 = {w2}")
    print(f"final b = {b}")
    

    print(" --- Testing the model ---")

    correct = 0
    total = len(test_data)

    for _, row in test_data.iterrows():
        x1 = row["x1"]
        x2 = row["x2"]
        true_class = row["class"]
        
        _, predicted_class = predict_class(x1, x2, w1, w2, b)
        
        if predicted_class == true_class:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    