import numpy as np

def sigmoid(x):
    return (1) / (1+np.exp(-x))


def linear(x, c=1):
    return c*x


def tanh(x):
    (1+np.exp((-2)*(x))) / (1-np.exp((-2)*(x)))


def ReLU(x: list):
    m = [max(0, i) for i in x]
    return m


def softmax(X): 
    s = sum(list(map(lambda i: np.exp(i), X)))
    
    result = [(np.exp(i))/s for i in X]
        
    return list(map(lambda i: float(i), result))
        

if __name__ == "__main__":
    print(sigmoid(3))
    print(softmax([2,2]))