import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.activations import sigmoid


def random_data_generator(plot = False):

    n_samples = 100

    x_min, x_max = -10, 10
    y_min, y_max = -10, 30

    X = np.random.uniform(low=x_min, high=x_max, size=n_samples)
    Y = np.random.uniform(low=y_min, high=y_max, size=n_samples)

    def decision_boundary(x):
        return 2 * x + 1

    classes = (Y > decision_boundary(X)).astype(int)

    data = np.column_stack((X, Y, classes))

    df = pd.DataFrame(data, columns=['x', 'y', 'class'])

    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:2000], Y[:2000], c=classes[:2000], cmap='bwr', s=10, alpha=0.6)
        x_vals = np.linspace(x_min, x_max, 100)
        plt.plot(x_vals, decision_boundary(x_vals), 'k--', label='y = 2x + 1')
        plt.title("Generated Data (First 2000 points)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return df








def initialize_weights():
    return np.random.randn() * 0.1, np.random.randn() * 0.1, np.random.randn() * 0.1 


def forward(weight1, weight2,  x1, x2, bias, activation):
    return activation(weight1*x1 + weight2*x2 + bias)



def dl_dw_linear(xs: list, weights, class_predicted, true_class, loss_function, bias):
    """
    derivative of loss with respect to weight number weight number
    """
    ws = []
    for w in range(len(weights)):
        for j in range(len(xs[0])):
            res = 0
            error = true_class[j] - class_predicted[j]
            sig = loss_function(weights[0] * xs[0][j] + weights[1] * xs[1][j] + bias)
            x = xs[w][j]
            res += error * (sig * (1-sig)) * x
        ws.append(-2 * res)
    return ws


def linear_classification(dataset, epochs=10, learning_rate = 0.0001):
    w1, w2, b = initialize_weights()
    
    print(f"Initial w1 = {w1:.4f}")
    print(f"Initial w2 = {w2:.4f}")
    print(f"Initial b  = {b:.4f}")
            
    x1_batch = dataset["x"]
    x2_batch = dataset["y"]
    
    class_batch = list(dataset["class"])
    # print(x1_batch[:5])
    # print(x2_batch[:5])

    for epoch in range(epochs):
        output = forward(w1, w2, x1_batch, x2_batch, b, sigmoid)
        dl_dws = dl_dw_linear([x1_batch, x2_batch], [w1, w2], output, class_batch, sigmoid, b)
        print(f"derivatives in epoch {epoch}")
        print(dl_dws[:100])
        # print(output[:2])



if __name__ == "__main__":
    
    # ws = [0.1, 0.4]
    # bis = 1
    # x = [[1,2,3,4], [5,6,7,8]]
    # y = [0,1,1,0]
    # f = [1,1,1,1]
    # print(dl_dw_linear(x, ws, f, y, sigmoid, bis))
        
    dataset = random_data_generator(plot=False)
    print(dataset.head())
    # linear_classification(dataset)
