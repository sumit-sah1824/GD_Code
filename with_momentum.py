import numpy as np
import matplotlib.pyplot as plt

def objective_function(x, A, b):
    return np.linalg.norm(A @ x + b)**2

def true_gradient(x, A, b):
    return 2 * A.T @ (A @ x + b)

def estimated_gradient(x, A, b, delta=1e-4, num_directions=10):
    d = len(x)
    fx = objective_function(x, A, b)
    grad_est = np.zeros(d)
    
    for _ in range(num_directions):
        u = np.random.randn(d)
        u /= np.linalg.norm(u)
        fx_delta = objective_function(x + delta * u, A, b)
        grad_est += ((fx_delta - fx) / delta) * u
        
    return grad_est / num_directions

def run_gd_with_momentum(A, b, x0, learning_rate=0.01, steps=500,
                         true_grad_prob=0.05, momentum=0.9, use_momentum=True):
    x = x0.copy()
    v = np.zeros_like(x)  # velocity
    loss_history = [objective_function(x, A, b)]
    
    for t in range(steps):
        # Use estimated or true gradient
        if np.random.rand() < true_grad_prob:
            grad = true_gradient(x, A, b)
        else:
            grad = estimated_gradient(x, A, b, num_directions=10)
        
        if use_momentum:
            v = momentum * v + grad
            x = x - learning_rate * v
        else:
            x = x - learning_rate * grad
        
        loss_history.append(objective_function(x, A, b))
    
    return x, loss_history

def main():
    np.random.seed(42)
    
    d = 30
    A = np.random.randn(d, d)
    A = (A + A.T) / 2  # symmetric matrix
    b = np.random.randn(d)
    x0 = np.zeros(d)

    # Run without momentum
    _, losses_no_mom = run_gd_with_momentum(A, b, x0, use_momentum=False)

    # Run with momentum
    _, losses_mom = run_gd_with_momentum(A, b, x0, use_momentum=True)

    # Plot comparison
    plt.plot(losses_no_mom, label='Without Momentum')
    plt.plot(losses_mom, label='With Momentum')
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title("Gradient Descent With and Without Momentum")
    plt.legend()
    plt.grid(True)
    plt.show()

main()
