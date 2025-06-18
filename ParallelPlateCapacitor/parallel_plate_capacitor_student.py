import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def initialize_grid(xgrid, ygrid, w, d):
    """Initialize grid with boundary conditions for parallel plates."""
    u = np.zeros((ygrid, xgrid))
    # Calculate plate positions
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # Set boundary conditions for plates
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    return u, (xL, xR, yB, yT)

def apply_boundary_conditions(u, plate_positions):
    """Apply boundary conditions to the grid."""
    xL, xR, yB, yT = plate_positions
    u[yT, xL:xR+1] = 100.0
    u[yB, xL:xR+1] = -100.0
    return u

def solve_laplace_jacobi(xgrid, ygrid, w, d, tol=1e-5):
    """
    Solve Laplace equation using Jacobi iteration method.
    
    Args:
        xgrid (int): Number of grid points in x direction
        ygrid (int): Number of grid points in y direction
        w (int): Width of parallel plates
        d (int): Distance between parallel plates
        tol (float): Convergence tolerance
    
    Returns:
        tuple: (potential_array, iterations, convergence_history)
    """
    # Initialize grid and get plate positions
    u, plate_positions = initialize_grid(xgrid, ygrid, w, d)
    xL, xR, yB, yT = plate_positions
    
    iterations = 0
    max_iter = 10000
    convergence_history = []
    
    while iterations < max_iter:
        u_old = u.copy()
        
        # Jacobi iteration
        u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + 
                             u[1:-1, 2:] + u[1:-1, :-2])
        
        # Maintain boundary conditions
        apply_boundary_conditions(u, plate_positions)
        
        # Calculate convergence metric
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)

        # Check convergence
        iterations += 1
        if max_change < tol:
            break
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, tol=1e-5, max_iter=10000):
    """
    Solve Laplace equation using Gauss-Seidel SOR iteration method.
    
    Args:
        xgrid (int): Number of grid points in x direction
        ygrid (int): Number of grid points in y direction
        w (int): Width of parallel plates
        d (int): Distance between parallel plates
        omega (float): Relaxation factor
        tol (float): Convergence tolerance
        max_iter (int): Maximum number of iterations
    
    Returns:
        tuple: (potential_array, iterations, convergence_history)
    """
    # Initialize grid and get plate positions
    u, plate_positions = initialize_grid(xgrid, ygrid, w, d)
    xL, xR, yB, yT = plate_positions
    
    convergence_history = []
    
    for iteration in range(max_iter):
        u_old = u.copy()
        max_change = 0.0
        
        # SOR iteration
        for i in range(1, ygrid-1):
            for j in range(1, xgrid-1):
                # Skip plate regions
                if (i == yT and xL <= j <= xR) or (i == yB and xL <= j <= xR):
                    continue
                
                # Calculate new value
                new_val = 0.25 * (u[i+1, j] + u[i-1, j] + 
                                  u[i, j+1] + u[i, j-1])
                
                # Calculate change and update
                delta = omega * (new_val - u[i, j])
                u[i, j] += delta
                
                # Track maximum change
                if abs(delta) > max_change:
                    max_change = abs(delta)
        
        # Maintain boundary conditions
        apply_boundary_conditions(u, plate_positions)
        
        convergence_history.append(max_change)
        
        # Check convergence
        if max_change < tol:
            break
    
    return u, iteration + 1, convergence_history

def plot_results(x, y, u, method_name):
    """
    Plot 3D potential distribution and equipotential contours.
    
    Args:
        x (array): X coordinates
        y (array): Y coordinates
        u (array): Potential distribution
        method_name (str): Name of the method used
    """
    fig = plt.figure(figsize=(12, 6))
    X, Y = np.meshgrid(x, y)
    
    # 3D wireframe plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, u, rstride=2, cstride=2, alpha=0.7)
    ax1.contour(X, Y, u, zdir='z', offset=u.min(), levels=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Potential (V)')
    ax1.set_title(f'3D Potential Distribution ({method_name})')
    
    # Equipotential contour plot and Electric field streamlines
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 20)
    contour = ax2.contour(X, Y, u, levels=levels, colors='red', linewidths=1)
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Calculate electric field components
    Ey, Ex = np.gradient(-u)
    ax2.streamplot(X, Y, Ex, Ey, density=1.5, color='blue', linewidth=1, 
                   arrowsize=1.5, arrowstyle='->')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Equipotential Lines & Electric Field ({method_name})')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the simulation and analysis."""
    # Simulation parameters
    xgrid, ygrid = 50, 50
    w, d = 20, 20  # plate width and separation
    tol = 1e-3
    
    # Create coordinate arrays
    x = np.arange(xgrid)
    y = np.arange(ygrid)
    
    print("Solving Laplace equation for parallel plate capacitor...")
    print(f"Grid size: {xgrid} x {ygrid}, Plate width: {w}, separation: {d}")
    print(f"Convergence tolerance: {tol}")
    
    # Solve using Jacobi method
    print("\n1. Jacobi iteration method:")
    start_time = time.perf_counter()
    u_jacobi, iter_jacobi, conv_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d, tol)
    time_jacobi = time.perf_counter() - start_time
    print(f"   Converged in {iter_jacobi} iterations ({time_jacobi:.4f} seconds)")
    
    # Solve using SOR method
    print("\n2. Gauss-Seidel SOR iteration method:")
    start_time = time.perf_counter()
    u_sor, iter_sor, conv_sor = solve_laplace_sor(xgrid, ygrid, w, d, tol=tol)
    time_sor = time.perf_counter() - start_time
    print(f"   Converged in {iter_sor} iterations ({time_sor:.4f} seconds)")
    
    # Performance comparison
    print("\n3. Performance comparison:")
    print(f"   Jacobi: {iter_jacobi} iterations, {time_jacobi:.4f}s")
    print(f"   SOR:    {iter_sor} iterations, {time_sor:.4f}s")
    print(f"   Speedup: {iter_jacobi/iter_sor:.1f}x iterations, {time_jacobi/time_sor:.1f}x time")
    
    # Plot results
    plot_results(x, y, u_jacobi, "Jacobi")
    plot_results(x, y, u_sor, "SOR")
    
    # Plot convergence comparison
    plt.figure(figsize=(10, 6))
    plt.semilogy(conv_jacobi, 'r-', label='Jacobi')
    plt.semilogy(conv_sor, 'b-', label='SOR')
    plt.xlabel('Iteration')
    plt.ylabel('Max Change')
    plt.title('Convergence Comparison')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
