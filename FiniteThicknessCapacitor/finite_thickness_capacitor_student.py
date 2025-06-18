#!/usr/bin/env python3
"""
Module: Finite Thickness Parallel Plate Capacitor (Student Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    Solve 2D Laplace equation using SOR method for finite thickness parallel plate capacitor.
    
    Args:
        nx (int): Number of grid points in x direction
        ny (int): Number of grid points in y direction
        plate_thickness (int): Thickness of conductor plates in grid points
        plate_separation (int): Separation between plates in grid points
        omega (float): Relaxation factor (1.0 < omega < 2.0)
        max_iter (int): Maximum number of iterations
        tolerance (float): Convergence tolerance
        
    Returns:
        np.ndarray: 2D electric potential distribution
    """
    # 初始化电势网格，边界设为0
    potential = np.zeros((ny, nx))
    
    # 设置极板位置和电势值
    plate_center_y = ny // 2
    plate_left_x = (nx - plate_separation) // 2 - plate_thickness
    plate_right_x = (nx + plate_separation) // 2
    
    # 设置左极板(负)和右极板(正)的电势边界条件
    potential[plate_center_y-plate_thickness//2:plate_center_y+plate_thickness//2+1, 
              plate_left_x:plate_left_x+plate_thickness] = -1.0
    potential[plate_center_y-plate_thickness//2:plate_center_y+plate_thickness//2+1, 
              plate_right_x:plate_right_x+plate_thickness] = 1.0
    
    # 标记极板位置，计算时保持固定
    is_plate = np.zeros_like(potential, dtype=bool)
    is_plate[plate_center_y-plate_thickness//2:plate_center_y+plate_thickness//2+1, 
             plate_left_x:plate_left_x+plate_thickness] = True
    is_plate[plate_center_y-plate_thickness//2:plate_center_y+plate_thickness//2+1, 
             plate_right_x:plate_right_x+plate_thickness] = True
    
    # SOR迭代求解
    residual = np.zeros_like(potential)
    for iteration in range(max_iter):
        max_residual = 0
        
        # 逐点更新电势
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                # 跳过极板区域
                if is_plate[j, i]:
                    continue
                    
                # 计算新的电势值
                old_value = potential[j, i]
                new_value = 0.25 * (potential[j, i+1] + potential[j, i-1] + 
                                   potential[j+1, i] + potential[j-1, i])
                
                # 超松弛更新
                potential[j, i] = old_value + omega * (new_value - old_value)
                
                # 计算残差
                residual[j, i] = abs(potential[j, i] - old_value)
                if residual[j, i] > max_residual:
                    max_residual = residual[j, i]
        
        # 检查收敛性
        if max_residual < tolerance:
            print(f"Converged after {iteration} iterations with residual {max_residual}")
            break
            
    if iteration == max_iter - 1:
        print(f"Warning: Did not converge within {max_iter} iterations. Final residual: {max_residual}")
    
    return potential

def calculate_charge_density(potential_grid, dx, dy):
    """
    Calculate charge density using Poisson equation.
    
    Args:
        potential_grid (np.ndarray): 2D electric potential distribution
        dx (float): Grid spacing in x direction
        dy (float): Grid spacing in y direction
        
    Returns:
        np.ndarray: 2D charge density distribution
    """
    # 使用中心差分计算二阶导数
    laplacian = np.zeros_like(potential_grid)
    
    # 内部点的拉普拉斯算子
    laplacian[1:-1, 1:-1] = (potential_grid[1:-1, 2:] - 2*potential_grid[1:-1, 1:-1] + potential_grid[1:-1, :-2]) / dx**2 + \
                           (potential_grid[2:, 1:-1] - 2*potential_grid[1:-1, 1:-1] + potential_grid[:-2, 1:-1]) / dy**2
    
    # 根据泊松方程 ρ = -ε₀∇²V，这里设ε₀=1
    charge_density = -laplacian
    
    return charge_density

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    Create visualization of potential and charge density distributions.
    
    Args:
        potential (np.ndarray): 2D electric potential distribution
        charge_density (np.ndarray): Charge density distribution
        x_coords (np.ndarray): X coordinate array
        y_coords (np.ndarray): Y coordinate array
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 电势分布的等高线图
    ax1 = fig.add_subplot(131)
    contour = ax1.contourf(x_coords, y_coords, potential, 50, cmap=cm.viridis)
    ax1.set_title('Electric Potential')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(contour, ax=ax1)
    
    # 电荷密度分布的热力图
    ax2 = fig.add_subplot(132)
    im = ax2.imshow(charge_density, cmap=cm.coolwarm, origin='lower', 
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax2.set_title('Charge Density')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im, ax=ax2)
    
    # 电场线和等势线的组合图
    ax3 = fig.add_subplot(133)
    
    # 计算电场 (E = -∇V)
    Ey, Ex = np.gradient(-potential)
    
    # 绘制等势线
    ax3.contour(x_coords, y_coords, potential, 20, colors='k', alpha=0.5)
    
    # 绘制电场线
    ax3.streamplot(x_coords, y_coords, Ex, Ey, density=1.5, color='b', linewidth=1, arrowsize=1.5)
    
    ax3.set_title('Electric Field Lines and Equipotentials')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置模拟参数
    nx = 100  # x方向网格点数
    ny = 100  # y方向网格点数
    plate_thickness = 6  # 极板厚度(网格点)
    plate_separation = 30  # 极板间距(网格点)
    omega = 1.9  # 超松弛因子
    max_iter = 10000  # 最大迭代次数
    tolerance = 1e-6  # 收敛容差
    
    # 计算网格间距
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    
    # 创建坐标数组
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    
    # 求解拉普拉斯方程
    potential = solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega, max_iter, tolerance)
    
    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # 绘制结果
    plot_results(potential, charge_density, x_coords, y_coords)
