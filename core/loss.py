import numpy as np


def sqe(y: list, y_hat: list):
    s = 0
    for i, j in zip(y, y_hat):
        s += ((i-j)**2)
    return s
    


print(sqe([1,2,3], [2,4,6]))
