import numpy as np

def step_function(x):
    return 1 if x>=0 else 0

class Perceptron:
    def __init__(self, learning_rate = 0.01, epochs = 10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1]) # weights for all the features in X
        self.bias  = 0

        for _ in range(self.epochs):
            for i in range(len(X)):
                linear_output = np.dot(X[i],self.weights) + self.bias
                y_pred = step_function(linear_output)
                update = self.learning_rate * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update
    
    def predict(self,X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([step_function(X) for X in linear_output])
    
X = np.array([[2, 3], [1, 1], [4, 5], [6, 8]])
y = np.array([0, 0, 1, 1])

perceptron = Perceptron(learning_rate=0.1,epochs=10)
perceptron.fit(X,y)

print(f"Predictions: {perceptron.predict(X)}")

"""
Predictions: [0 0 1 1]

"""