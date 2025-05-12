import numpy as np
from .activations import sigmoid as sg, tanh as th, ReLU as rl, linear as ln


def sigmoid(a):
    return (a)*(1-a)


def tanh(a):
    return 1 - (a**2)


def linear(c=1):
    return c


def ReLU(a):
    return 0 if a<0 else 1


def d_sqe_w(x, y_predicted, y_hat):
    result = 0
    for i in range(len(x)):
        result += (y_hat[i]-y_predicted[i]) * x[i]
    return (-2) * result/len(x)


def d_sqe_b(x, y_predicted, y_hat):
    result = 0
    for i in range(len(x)):
        result += (y_hat[i]-y_predicted[i])
    return (-2) * result/len(x)


if __name__ == "__main__":
    print(sigmoid(sg(3)))
