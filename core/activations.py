import numpy as np


# def sigmoid(x):
#     return (1) / (1+np.exp(-np.ndarray(x)))

def sigmoid(X):
    if isinstance(X, list):
        return [(1) / (1+np.exp(-x)) for x in X]
    else:
        return (1) / (1+np.exp(-X))


def linear(x, c=1):
    return c*x


def tanh(x):
    (1+np.exp((-2)*(x))) / (1-np.exp((-2)*(x)))


def ReLU(x: list):
    if isinstance(x, list):
        return [max(0, i) for i in x]
    else:
        return max(0, x)


def softmax(X): 
    s = sum(list(map(lambda i: np.exp(i), X)))
    
    result = [(np.exp(i))/s for i in X]
        
    return list(map(lambda i: float(i), result))
        

if __name__ == "__main__":
    print(sigmoid(3))
    print(softmax([2,2]))