import numpy as np
import matplotlib.pyplot as plt

def objective_function(x, A, b):
    """
    Computes f(x) = ||Ax + b||^2
    """
    return np.linalg.norm(A @ x + b)**2

def true_gradient(x, A, b):
    """
    Computes the true gradient: ∇f(x) = 2 A^T (Ax + b)
    """
    return 2 * A.T @ (A @ x + b)

def estimated_gradient(x, A, b, delta=1e-4, num_directions=20):
    """
    Estimate the gradient by averaging over 'num_directions' random directions
    """
    d = len(x)
    fx = objective_function(x, A, b)
    grad_est = np.zeros(d)
    
    for _ in range(num_directions):
        u = np.random.randn(d)
        u /= np.linalg.norm(u)  # make it a unit vector
        
        fx_plus = objective_function(x + delta * u, A, b)
        grad_est += ((fx_plus - fx) / delta) * u
    
    return grad_est / num_directions

def run_gradient_descent(A, b, x0, learning_rate=0.01, steps=500, true_grad_prob=0.05):
    """
    Runs gradient descent with estimated gradient and occasional true gradient
    """
    x = x0.copy()
    loss_history = [objective_function(x, A, b)]

    for t in range(steps):
        # With small probability, use true gradient
        if np.random.rand() < true_grad_prob:
            grad = true_gradient(x, A, b)
        else:
            grad = estimated_gradient(x, A, b, num_directions=20)
        
        x = x - learning_rate * grad
        loss_history.append(objective_function(x, A, b))
    
    return x, loss_history

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Problem dimensions
    d = 100  # dimension of x
    A = np.random.randn(d, d)
    A = (A + A.T) / 2  # make A symmetric
    b = np.random.randn(d)
    x0 = np.zeros(d)

    # Run optimization
    x_min, losses = run_gradient_descent(A, b, x0)

    # Plot loss over iterations
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("f(x)=$||Ax + b||^2$")
    plt.title("GD with Estimated Gradient (20 directions)")
    plt.grid(True)
    plt.show()

# Call the main function
main()
