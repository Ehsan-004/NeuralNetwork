import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.activations import sigmoid
from core.loss import msq


def random_data_generator(plot = False):

    n_samples = 10000

    x_min, x_max = -10, 10
    y_min, y_max = -10, 30

    X = np.random.uniform(low=x_min, high=x_max, size=n_samples)
    Y = np.random.uniform(low=y_min, high=y_max, size=n_samples)

    def decision_boundary(x):
        return 2 * x + 1

    classes = (Y > decision_boundary(X)).astype(int)

    data = np.column_stack((X, Y, classes))

    df = pd.DataFrame(data, columns=['x1', 'x2', 'class'])

    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:2000], Y[:2000], c=classes[:2000], cmap='bwr', s=10, alpha=0.6)
        x_vals = np.linspace(x_min, x_max, 100)
        plt.plot(x_vals, decision_boundary(x_vals), 'k--', label='y = 2x + 1')
        plt.title("Generated Data (First 2000 points)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return df








def initialize_weights():
    return np.random.randn() * 0.1, np.random.randn() * 0.1, np.random.randn() * 0.1 


def forward(weight1, weight2,  x1, x2, bias, activation):
    # a = weight1*x1
    # b = weight2*x2
    # print(f"a = {a}")
    # print(f"b = {b}")
    # print(f"a+b = {a+b}")
    # print(activation(19))
    return activation(weight1*x1 + weight2*x2 + bias)



def dl_dw_linear(xs: list, weights, class_predicted, true_class, loss_function, bias):
    """
    derivative of loss with respect to weight number weight number
    """
    dl_ws = []
    for w in range(len(weights)):
        for j in range(len(xs[0])):
            res = 0
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


def linear_classification(dataset, epochs=1000, learning_rate = 0.001):
    w1, w2, b = initialize_weights()
    
    print(f"Initial w1 = {w1:.4f}")
    print(f"Initial w2 = {w2:.4f}") 
    print(f"Initial b  = {b:.4f}")
            
    x1_batch = dataset["x1"]
    x2_batch = dataset["x2"]
    classes = dataset["class"]
    
    print(len(x1_batch))
    print(len(x2_batch))
    
    class_batch = list(dataset["class"])
   
    print("starting training")
   
    for epoch in range(epochs):
        output = forward(w1, w2, np.array(x1_batch), np.array(x2_batch), b, sigmoid)
        # print(output[:5])
        dl_dws = dl_dw_linear([x1_batch, x2_batch], [w1, w2], output, class_batch, sigmoid, b)
        dl_dbs = dl_db_linear([x1_batch, x2_batch], [w1, w2], output, class_batch, sigmoid, b)
        # print(f"derivatives in epoch {epoch}")
        # print(f"dl_dw1 = {dl_dws[0]}")
        # print(f"dl_dw2 = {dl_dws[1]}")
        # print(f"dl_db = {dl_dbs}")
        w1 -= learning_rate * dl_dws[0]
        w2 -= learning_rate * dl_dws[1]
        b -= learning_rate * dl_dbs
        # print()
        # print(f"new w1 = {w1}")
        # print(f"new w2 = {w2}")
        # print(f"new b = {b}")
        
        loss = msq(classes, output)
        
        if (epoch + 1) % 10 == 0 or epoch == 0: 
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.8f}, w1: {w1:.6f}, w2: {w2:.6f}, Current b: {b:.6f}")
        
    
    
        # print(output[:2])
    print("training finished")
    print(f"final w1 = {w1}")
    print(f"final w2 = {w2}")
    print(f"final b = {b}")


if __name__ == "__main__":
    
    # ws = [0.1, 0.4]
    # bis = 1
    # x = [[1,2,3,4], [5,6,7,8]]
    # y = [0,1,1,0]
    # f = [1,1,1,1]
    # print(dl_dw_linear(x, ws, f, y, sigmoid, bis))
        
    # print(forward(2, 3, np.array([1,2,3]), np.array([4,5,6]), 5, sigmoid))
        
    dataset = random_data_generator(plot=False)
    dataset["class"] = dataset["class"].astype(int)
    linear_classification(dataset)
