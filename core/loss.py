import numpy as np


def sqe(y: list, y_hat: list):
    s = 0
    for i, j in zip(y, y_hat):
        s += ((i-j)**2)
    return s
    

# This function has been used in single neuron but notin neural netword
def msq(y_true , y_predicted):
    return (sum((y_true - y_predicted)**2)) / len(y_true)


# This functions has been used in neural network
def mse(y_true , y_predicted):
    y_true, y_predicted = np.array(y_true), np.array(y_predicted)
    errors = [a-b for a, b in zip(y_true, y_predicted)]
    squared_errors = [a**2 for a in errors]
    return np.sum(np.array(squared_errors)) / len(y_true[0]), [list(e) for e in errors]


def CCE(y_true, y_predicted):
    indexes = [x.index(1) for x in y_true]
    errors = [-np.log(predict[i]) for i, predict in zip(indexes, y_predicted)]
    return sum(errors) / len(y_true)
    
    
def calculate_mse(predictions, targets):
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)
    return mse


if __name__ == "__main__":
    print(calculate_mse([1,2,3], [2,4,6]))
    print(msq([1,2,3], [2,4,6]))
