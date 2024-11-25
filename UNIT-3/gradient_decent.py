import numpy as np

def gradient_decent(X,y,lr = 0.01, epochs =1000):
    m,b = 0,0
    n = len(X)

    for _ in range(epochs):

        y_pred_fn = m*X + b

        #calculate the gradients
        slope_gradient = (-2/n) * (sum(X * (y - y_pred_fn)))
        intercept_gradient = (-2/n) * sum(y - y_pred_fn)

        #update the slope and intercept
        m -= lr * slope_gradient
        b -= lr * intercept_gradient

        return m,b
    
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 15])

learning_rate = 0.01
epochs = 1000
m,b = gradient_decent(X,y,lr = learning_rate,epochs=epochs)

print(f"Optimized parameters: m = {m}, b = {b}")
print(f"Expected relationship: y = {m:.2f}x + {b:.2f}")

"""
Optimized parameters: m = 0.66, b = 0.18
Expected relationship: y = 0.66x + 0.18


"""