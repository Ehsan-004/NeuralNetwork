import numpy as np


def sqe(y: list, y_hat: list):
    s = 0
    for i, j in zip(y, y_hat):
        s += ((i-j)**2)
    return s
    

def msq(y_true , y_predicted):
    return (sum((y_true - y_predicted)**2)) / len(y_true)


def calculate_mse(predictions, targets):
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)
    return mse


if __name__ == "__main__":
    print(calculate_mse([1,2,3], [2,4,6]))
    print(msq([1,2,3], [2,4,6]))
