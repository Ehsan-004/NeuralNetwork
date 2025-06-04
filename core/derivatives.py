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


def d_l_w_linear(x, y_predicted, y_true):
    return -2 * ((sum((y_true - y_predicted) * x))/len(y_true))


def d_l_b_linear(y_predicted, y_true):
    return -2 * ((sum(y_true - y_predicted)) / len(y_true))


def dl_w1_linear_i(xs: list, class_predicted, true_class, weight_number, loss_function):
    """
    derivative of loss with respect to weight number weight number
    """
    a = loss_function()
    return -2 * sum((true_class - class_predicted))
    


if __name__ == "__main__":
    print(sigmoid(sg(3)))
