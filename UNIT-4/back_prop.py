import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Backpropagation implementation
class BackpropagationNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def train(self, X, y, epochs, lr):
        for _ in range(epochs):
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = sigmoid(hidden_layer_input)

            final_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output_layer_output = sigmoid(final_layer_input)

            # Error calculation
            output_error = y - output_layer_output
            output_delta = output_error * sigmoid_derivative(output_layer_output)

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

            # Backward pass (weights and bias updates)
            self.weights_hidden_output += np.dot(hidden_layer_output.T, output_delta) * lr
            self.bias_output += np.sum(output_delta, axis=0) * lr
            self.weights_input_hidden += np.dot(X.T, hidden_delta) * lr
            self.bias_hidden += np.sum(hidden_delta, axis=0) * lr

    def predict(self, X):
        hidden_layer = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        output_layer = sigmoid(np.dot(hidden_layer, self.weights_hidden_output) + self.bias_output)
        return output_layer

# Example usage
nn = BackpropagationNN(input_size=2, hidden_size=3, output_size=1)

# Input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
y = np.array([[0], [1], [1], [0]])              # XOR outputs

# Training
nn.train(X, y, epochs=10000, lr=0.1)

# Predictions
output = nn.predict(X)
print("Predicted Output:\n", output)
