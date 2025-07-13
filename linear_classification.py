from neural_network import Layer, NeuralNetwork
from core.tools import linear_classified_data_generator
from core.derivatives import sigmoid as d_sigmoid
from core.activations import sigmoid

if __name__ == "__main__":
    
    print("\n=== preprocessing ===\n")
    
    df_data = linear_classified_data_generator(slope=2, intercept=5, n_samples=1000, plot=False)
    X = df_data[['x1', 'x2']].values.tolist()
    Y = df_data[['class']].values.tolist()
    
    train_percent = 0.9
    valid_percent = 0.1
    
    X_train = X[:int(train_percent*len(X))]
    X_test = X[int(train_percent*len(X)):]
    
    Y_train = Y[:int(train_percent*len(X))]
    Y_test = Y[int(train_percent*len(X)):]
    

    la = [
        Layer(input_neurons=2, neuron_num=8, activation=sigmoid, activation_differ=d_sigmoid, w=0.1, lr = 0.1),
        Layer(input_neurons=8, neuron_num=4, activation=sigmoid, activation_differ=d_sigmoid, w=0.1, lr = 0.1),
        Layer(input_neurons=4, neuron_num=2, activation=sigmoid, activation_differ=d_sigmoid, w=0.1, lr = 0.1),
        Layer(input_neurons=2, neuron_num=1, activation=sigmoid, activation_differ=d_sigmoid, w=0.1, lr = 0.1),
        ]
    

    net = NeuralNetwork(la)
        
    print("\n=== training ===\n")
    
    for epoch in range(100):
        total_loss = 0
        total_loss = 0
        for x, y_true in zip(X_train, Y_train):  # x, y_true are datapoints
            y_pred = net(x)
            net.backward(y_true)
            net.step()
            total_loss += net.loss
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Training Loss: {total_loss/len(X_train):.4f}")
            
            
    print("\n=== validating ===\n")
            
    correct = 0
    
    for x, y_true in zip(X_test, Y_test):
        pred = net(x)
        # print(f"Predicted: {1 if pred[0] > 0.5 else 0} | target: {y_true[0]}")
        p = 1 if pred[0] > 0.5 else 0
        if p == y_true[0]:
            correct += 1
            
    print(f"number of test samples: {len(X_test)}")
    print(f"accuracy of model: {correct/len(X_test):.2f}")
        

