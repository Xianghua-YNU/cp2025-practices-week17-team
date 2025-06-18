import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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
    # Initialize potential grid
    u = np.zeros((ygrid, xgrid))
    
    # Calculate plate positions
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # Set boundary conditions for plates
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    iterations = 0
    max_iter = 10000
    convergence_history = []
    
    # Pre-calculate indices for faster iteration
    inner_i = slice(1, ygrid-1)
    inner_j = slice(1, xgrid-1)
    
    while iterations < max_iter:
        u_old = u.copy()
        
        # Jacobi iteration (optimized using slicing)
        u[inner_i, inner_j] = 0.25 * (
            u[2:, inner_j] + 
            u[:-2, inner_j] + 
            u[inner_i, 2:] + 
            u[inner_i, :-2]
        )

        # Maintain boundary conditions
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
        # Calculate convergence metric
        max_change = np.max(np.abs(u - u_old))
        convergence_history.append(max_change)

        # Check convergence
        iterations += 1
        if max_change < tol:
            break
    
    return u, iterations, convergence_history

def solve_laplace_sor(xgrid, ygrid, w, d, omega=1.25, Niter=1000, tol=1e-5):
    """
    Solve Laplace equation using Gauss-Seidel SOR iteration method.
    
    Args:
        xgrid (int): Number of grid points in x direction
        ygrid (int): Number of grid points in y direction
        w (int): Width of parallel plates
        d (int): Distance between parallel plates
        omega (float): Relaxation factor
        Niter (int): Maximum number of iterations
    
    Returns:
        tuple: (potential_array, iterations, convergence_history)
    """
    # Initialize potential grid
    u = np.zeros((ygrid, xgrid))
    
    # Calculate plate positions
    xL = (xgrid - w) // 2
    xR = (xgrid + w) // 2
    yB = (ygrid - d) // 2
    yT = (ygrid + d) // 2
    
    # Set boundary conditions for plates
    u[yT, xL:xR+1] = 100.0  # Top plate: +100V
    u[yB, xL:xR+1] = -100.0  # Bottom plate: -100V
    
    convergence_history = []
    max_change = 0.0
    
    for iteration in range(Niter):
        max_change = 0.0  # Reset max_change for each iteration
        
        # SOR iteration with optimized boundary checks
        for i in range(1, ygrid-1):
            # Precompute plate row checks
            is_top_plate = (i == yT)
            is_bottom_plate = (i == yB)
            
            for j in range(1, xgrid-1):
                # Skip plate regions with optimized condition
                if (is_top_plate and xL <= j <= xR) or (is_bottom_plate and xL <= j <= xR):
                    continue
                
                # Calculate residual
                residual = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])
                
                # Apply SOR formula and track max change
                new_val = (1 - omega) * u[i, j] + omega * residual
                delta = abs(new_val - u[i, j])
                if delta > max_change:
                    max_change = delta
                u[i, j] = new_val
        
        # Maintain boundary conditions
        u[yT, xL:xR+1] = 100.0
        u[yB, xL:xR+1] = -100.0
        
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
    
    # 3D wireframe plot with enhanced visualization
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, u, rstride=2, cstride=2, alpha=0.7, linewidth=0.8)
    ax1.contour(X, Y, u, zdir='z', offset=u.min(), levels=15, cmap='coolwarm')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Electric Potential (V)')
    ax1.set_title(f'3D Potential Distribution ({method_name})')
    
    # Equipotential contour plot and Electric field streamlines
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 15)
    contour = ax2.contour(X, Y, u, levels=levels, colors='darkred', linewidths=1.2)
    ax2.clabel(contour, inline=True, fontsize=8, fmt='%1.1f')
    
    # Calculate electric field components (negative gradient of potential)
    Ey, Ex = np.gradient(-u)
    ax2.streamplot(X, Y, Ex, Ey, density=1.8, color='navy', linewidth=1.2, 
                   arrowsize=1.8, arrowstyle='->')
    
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'Equipotential Lines & Electric Field ({method_name})')
    ax2.grid(True, linestyle='--', alpha=0.7)
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
    
    print("Solving Laplace Equation for Parallel Plate Capacitor")
    print("=" * 60)
    print(f"Grid size: {xgrid} x {ygrid} | Plate width: {w} | Separation: {d}")
    print(f"Convergence tolerance: {tol}")
    print("-" * 60)
    
    # Solve using Jacobi method
    print("\n[1] Jacobi Iteration Method:")
    start_time = time.perf_counter()
    u_jacobi, iter_jacobi, conv_jacobi = solve_laplace_jacobi(xgrid, ygrid, w, d, tol)
    time_jacobi = time.perf_counter() - start_time
    print(f"  → Converged in {iter_jacobi} iterations ({time_jacobi:.4f} seconds)")
    
    # Solve using SOR method
    print("\n[2] SOR Iteration Method:")
    start_time = time.perf_counter()
    u_sor, iter_sor, conv_sor = solve_laplace_sor(xgrid, ygrid, w, d, tol=tol)
    time_sor = time.perf_counter() - start_time
    print(f"  → Converged in {iter_sor} iterations ({time_sor:.4f} seconds)")
    
    # Performance comparison
    print("\n[3] Performance Comparison:")
    print("  " + "-" * 40)
    print(f"  Method    | Iterations | Time (s)   | Speedup")
    print("  " + "-" * 40)
    print(f"  Jacobi    | {iter_jacobi:9d} | {time_jacobi:9.4f} | 1.00x")
    print(f"  SOR       | {iter_sor:9d} | {time_sor:9.4f} | {time_jacobi/time_sor:.2f}x")
    print("  " + "-" * 40)
    print(f"  Iteration speedup: {iter_jacobi/iter_sor:.1f}x")
    
    # Plot results
    plot_results(x, y, u_jacobi, "Jacobi Method")
    plot_results(x, y, u_sor, "SOR Method")
    
    # Plot convergence comparison with enhanced styling
    plt.figure(figsize=(10, 6))
    plt.semilogy(conv_jacobi, 'r-', linewidth=1.8, alpha=0.8, label='Jacobi Method')
    plt.semilogy(conv_sor, 'b-', linewidth=1.8, alpha=0.8, label='SOR Method')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Max Change (log scale)', fontsize=12)
    plt.title('Convergence Comparison', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
