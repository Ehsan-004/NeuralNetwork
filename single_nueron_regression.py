import numpy as np
from core.activations import linear
from core.derivatives import d_l_b_linear, d_l_w_linear
from core.loss import msq



def initialize_weights():
    return np.random.randn() * 0.1, np.random.randn() * 0.1 


def forward(weight, x, bias, activation):
    return activation(weight * x + bias)


def generate_linear_sequence(w, b, s, l):
    sequence = []
    for i in range(s, s + l):
        value = w * i + b
        sequence.append(value)
    return np.array(sequence)


def linear_regressor(dataset, epochs=120000, learning_rate = 0.0001):
    w, b = initialize_weights()
    print(f"Initial w = {w:.4f}")
    print(f"Initial b = {b:.4f}")
            
    x_batch = dataset["x"]
    y_batch = dataset["y"]
    
    for epoch in range(epochs):
        outputs = forward(w, x_batch, b, linear)

        loss = msq(y_batch, outputs)

        grad_w = d_l_w_linear(x_batch, outputs, y_batch)
        grad_b = d_l_b_linear(outputs, y_batch)


        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if (epoch + 1) % 1000 == 0 or epoch == 0: 
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.8f}, Current w: {w:.6f}, Current b: {b:.6f}")
          
    return w, b

        

if __name__ == "__main__":
    dataset = {"x": np.array(list(range(1, 101))),
               "y": generate_linear_sequence(2, 1, 1, 100)}
    w, b = linear_regressor(dataset)
        
    print()
    print(" --- Training Finished ---")
    print(f"Final w = {w:.6f}")
    print(f"Final b = {b:.6f}")
    
    print()
    print(" --- Testing the model ---")
    

    test_x = 40
    print()
    print(f"Actual Y for x={test_x}: {2*40 + 1}")
    print(f"Model predict for x={test_x}: {forward(w, test_x, b, linear)}")
    
    
    test_x = 75
    print()
    print(f"Actual Y for x={test_x}: {2*75 + 1}")
    print(f"Model predict for x={test_x}: {forward(w, test_x, b, linear)}")
    
        
    
    