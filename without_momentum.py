import numpy as np
import matplotlib.pyplot as plt

def f(x, A, b):
    return np.linalg.norm(A @ x + b) ** 2    # Defining the Objective function

def grad_f(x, A, b):                       
    return 2 * A.T @ (A @ x + b)            # Gradient of the objective function

def estimated_grad(x, A, b, delta=1e-4):
    d = len(x)
    u = np.random.randn(d)
    u /= np.linalg.norm(u)
    fx = f(x, A, b)
    fx_delta = f(x + delta * u, A, b)
    return ((fx_delta - fx) / delta) * u

def gradient_descent_estimate(A, b, x0, lr=0.05, iterations=300, prob_true_grad=0.05):
    x = x0.copy()
    history = [f(x, A, b)]
    for t in range(iterations):
        if np.random.rand() < prob_true_grad:
            grad = grad_f(x, A, b)
        else:
            grad = estimated_grad(x, A, b)
        x = x - lr * grad
        history.append(f(x, A, b))
    return x, history

# ---- Setup ----
np.random.seed(42)
d = 20
A = np.random.randn(d, d)
A = (A + A.T) / 2  # Make A symmetric
b = np.random.randn(d)
x0 = np.ones(d)

# Run GD
x_min, history = gradient_descent_estimate(A, b, x0)

# Plot
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("f(x)")
plt.title("Minimize ||Ax + b||² ")
plt.grid(True)
plt.show()
