import numpy as np

def sigmoid(x):
    return (1) / (1+np.exp(x))


def linear(x, c=1):
    return c*x


def tanh(x):
    (1+np.exp((-2)*(x))) / (1-np.exp((-2)*(x)))


def ReLU(x):
    return max(0, x)


def softmax(X): 
    s = sum(list(map(lambda i: np.exp(i), X)))
    result = []
    for i in X:
        result.append((np.exp(i))/s)
        
    return list(map(lambda i: float(i), result))
        

if __name__ == "__main__":
    print(sigmoid(3))
    print(softmax([2,2]))