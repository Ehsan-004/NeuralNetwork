import numpy as np


def sqe(y: list, y_hat: list):
    s = 0
    for i, j in zip(y, y_hat):
        s += ((i-j)**2)
    return s
    

# This function has been used in single neuron but notin neural netword
def msq(y_true , y_predicted):
    return (sum((y_true - y_predicted)**2)) / len(y_true)


class Loss:
    def __init__(self):
        pass
    
    def compute_loss(self):
        pass
    
    def __call__(self, y_true, y_pred):
        return self.compute_loss(y_true, y_pred)
    

class MSELoss(Loss):
    def __init__(self):
        pass
    
    def compute_loss(self, y_true , y_pred):
        error = y_true - y_pred
        self.error = error
        return error ** 2
    

class CCELoss(Loss):
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps
        
    def compute_loss(self, y_true, y_predicted):
        index = y_true.index(1)
        return -np.log(y_predicted[index] + self.eps)
    
    @staticmethod
    def to_one_hot(y_predicted):
        if isinstance(y_predicted, list):
            max_ind = y_predicted.index(max(y_predicted))
            return [0 if i != max_ind else 1 for i in range(len(y_predicted))]


# This functions has been used in neural network
def mse(y_true , y_predicted):
    if isinstance(y_true, (list, np.array)):
        y_true, y_predicted = np.array(y_true), np.array(y_predicted)
        errors = [a-b for a, b in zip(y_true, y_predicted)]
        squared_errors = [a**2 for a in errors]
        return np.sum(np.array(squared_errors)) / len(y_true[0]), [list(e) for e in errors]
    
    elif isinstance(y_true, (float, int)):
        error = y_true - y_predicted  # error
        return error ** 2


# def CCE(y_true, y_predicted, eps = 1e-5):
#     print(f"y_true is {y_true}")
#     print(f"y_predicted is {y_predicted}")
#     indexes = [x.index(1) for x in y_true]
#     errors = [-np.log(predict[i]+eps) for i, predict in zip(indexes, y_predicted)]
#     return sum(errors) / len(y_true)


def CCE(y_true, y_predicted, eps = 1e-5):  # these are outputs from net and true output (one sample)
    index = y_true.index(1)
    return -np.log(y_predicted[index] + eps)
    
    
def calculate_mse(predictions, targets):
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(targets)
    return mse


if __name__ == "__main__":
    # print(calculate_mse([1,2,3], [2,4,6]))
    # print(msq([1,2,3], [2,4,6]))
    print(CCE([[1,0,0]], [[0,0,1]]))
    print(CCE([[1,0,0]], [[1,0,0]]))
