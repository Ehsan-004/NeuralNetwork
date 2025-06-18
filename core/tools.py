import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def generate_linear_sequence(w, b, s, l):
    """
    s: start
    l: length
    w: weight
    b: bias
    """
    sequence = []
    for i in range(s, s + l):
        value = w * i + b
        sequence.append(value)
    return np.array(sequence)


# This function is created by chatGPT
def linear_classified_data_generator(slope, intercept, n_samples=30000, plot=False):
    if n_samples < 1000:
        n_samples = 1000

    x_min, x_max = -10, 10
    y_min, y_max = -10, 30

    X = np.random.uniform(low=x_min, high=x_max, size=n_samples)
    Y = np.random.uniform(low=y_min, high=y_max, size=n_samples)

    def decision_boundary(x):
        return slope * x + intercept

    classes = (Y > decision_boundary(X)).astype(int)

    data = np.column_stack((X, Y, classes))

    df = pd.DataFrame(data, columns=['x1', 'x2', 'class'])
    df['class'] = df["class"].astype(int)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:2000], Y[:2000], c=classes[:2000], cmap='bwr', s=10, alpha=0.6)
        x_vals = np.linspace(x_min, x_max, 100)
        plt.plot(x_vals, decision_boundary(x_vals), 'k--', label=f'y = {slope}x + {intercept}')
        plt.title("Generated Data (First 2000 points)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return df


def initialize_weights(num):
    ws = [np.random.randn() * 0.1 for i in range(num)]
    return ws
