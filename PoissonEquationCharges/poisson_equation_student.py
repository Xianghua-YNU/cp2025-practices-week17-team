#!/usr/bin/env python3
"""
Module: Poisson Equation Solution
File: poisson_equation_solution.py

Solves 2D Poisson equation with positive and negative charges using relaxation method.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def solve_poisson_equation(M: int = 100, target: float = 1e-6, max_iterations: int = 10000) -> Tuple[np.ndarray, int, bool]:
    """
    使用松弛迭代法求解二维泊松方程
    
    参数:
        M (int): 每边的网格点数，默认100
        target (float): 收敛精度，默认1e-6
        max_iterations (int): 最大迭代次数，默认10000
    
    返回:
        tuple: (phi, iterations, converged)
            phi (np.ndarray): 电势分布数组，形状为(M+1, M+1)
            iterations (int): 实际迭代次数
            converged (bool): 是否收敛
    
    物理背景:
        求解泊松方程 ∇²φ = -ρ/ε₀，其中：
        - φ 是电势
        - ρ 是电荷密度分布
        - 边界条件：四周电势为0
        - 正电荷位于 (60:80, 20:40)，密度 +1 C/m²
        - 负电荷位于 (20:40, 60:80)，密度 -1 C/m²
    
    数值方法:
        使用有限差分法离散化，迭代公式：
        φᵢⱼ = 0.25 * (φᵢ₊₁ⱼ + φᵢ₋₁ⱼ + φᵢⱼ₊₁ + φᵢⱼ₋₁ + h²ρᵢⱼ)
    
    实现步骤:
    1. 初始化电势数组和电荷密度数组
    2. 设置边界条件（四周为0）
    3. 设置电荷分布
    4. 松弛迭代直到收敛
    5. 返回结果
    """
    h = 1.0
    
    phi = np.zeros((M+1, M+1), dtype=float)
    phi_prev = np.copy(phi)
    
    rho = np.zeros((M+1, M+1), dtype=float)
    
    pos_y1, pos_y2 = int(0.6*M), int(0.8*M)
    pos_x1, pos_x2 = int(0.2*M), int(0.4*M)
    neg_y1, neg_y2 = int(0.2*M), int(0.4*M)
    neg_x1, neg_x2 = int(0.6*M), int(0.8*M)
    
    rho[pos_y1:pos_y2, pos_x1:pos_x2] = 1.0   
    rho[neg_y1:neg_y2, neg_x1:neg_x2] = -1.0  
    
    
    delta = 1.0
    iterations = 0
    converged = False
    
    while delta > target and iterations < max_iterations:

        phi[1:-1, 1:-1] = 0.25 * (phi[0:-2, 1:-1] + phi[2:, 1:-1] + 
                                   phi[1:-1, :-2] + phi[1:-1, 2:] + 
                                   h*h * rho[1:-1, 1:-1])
        

        delta = np.max(np.abs(phi - phi_prev))
        
        phi_prev = np.copy(phi)
        iterations += 1
    
    converged = bool(delta <= target)
    
    return phi, iterations, converged

def visualize_solution(phi: np.ndarray, M: int = 100) -> None:
        """
    可视化电势分布
    
    参数:
        phi (np.ndarray): 电势分布数组
        M (int): 网格大小
    
    功能:
        - 使用 plt.imshow() 显示电势分布
        - 添加颜色条和标签
        - 标注电荷位置
    """
    

    im = plt.imshow(phi, extent=[0, M, 0, M], origin='lower', 
                    cmap='RdBu_r', interpolation='bilinear')
    

    cbar = plt.colorbar(im)
    cbar.set_label('Electric Potential (V)', fontsize=12)
    

    plt.fill_between([20, 40], [60, 60], [80, 80], alpha=0.3, color='red', label='Positive Charge')
    plt.fill_between([60, 80], [20, 20], [40, 40], alpha=0.3, color='blue', label='Negative Charge')
    

    plt.xlabel('x (grid points)', fontsize=12)
    plt.ylabel('y (grid points)', fontsize=12)
    plt.title('Electric Potential Distribution\nPoisson Equation with Positive and Negative Charges', fontsize=14)
    plt.legend()
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_solution(phi: np.ndarray, iterations: int, converged: bool) -> None:
    """
    分析解的统计信息
    
    参数:
        phi (np.ndarray): 电势分布数组
        iterations (int): 迭代次数
        converged (bool): 收敛状态
    
    功能:
        打印解的基本统计信息，如最大值、最小值、迭代次数等
    """
    print(f"Solution Analysis:")
    print(f"  Iterations: {iterations}")
    print(f"  Converged: {converged}")
    print(f"  Max potential: {np.max(phi):.6f} V")
    print(f"  Min potential: {np.min(phi):.6f} V")
    print(f"  Potential range: {np.max(phi) - np.min(phi):.6f} V")
    

    max_idx = np.unravel_index(np.argmax(phi), phi.shape)
    min_idx = np.unravel_index(np.argmin(phi), phi.shape)
    print(f"  Max potential location: ({max_idx[0]}, {max_idx[1]})")
    print(f"  Min potential location: ({min_idx[0]}, {min_idx[1]})")

if __name__ == "__main__":

    print("Solving 2D Poisson equation with relaxation method...")
    

    M = 100
    target = 1e-6
    max_iter = 10000
    

    phi, iterations, converged = solve_poisson_equation(M, target, max_iter)
    

    analyze_solution(phi, iterations, converged)
    
    visualize_solution(phi, M)
    

    plt.figure(figsize=(12, 5))
    

    plt.subplot(1, 2, 1)
    center_y = M // 2
    plt.plot(phi[center_y, :], 'b-', linewidth=2)
    plt.xlabel('x (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along y = {center_y}')
    plt.grid(True, alpha=0.3)
    

    plt.subplot(1, 2, 2)
    center_x = M // 2
    plt.plot(phi[:, center_x], 'r-', linewidth=2)
    plt.xlabel('y (grid points)')
    plt.ylabel('Potential (V)')
    plt.title(f'Potential along x = {center_x}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
