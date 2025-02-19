import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# 计算椭球体积的行列式目标函数
def objective(A, points):
    # A是一个9维的向量，我们需要将它转化为3x3矩阵
    A = A.reshape((3, 3))
    # 计算行列式
    det_A = np.linalg.det(A)
    return -det_A  # 我们最大化行列式，因此这里要取负

# 约束条件：所有点都在椭球内
def constraint(A, points):
    A = A.reshape((3, 3))
    constraints = []
    for p in points:
        constraints.append(np.dot(p.T, np.dot(A, p)) - 1)  # 应该小于等于1
    return constraints

# 绘制点云和椭球
def plot_points_and_ellipsoid(points, A):
    # 用特征值分解椭球矩阵A来获得椭球的半轴
    eigvals, eigvecs = np.linalg.eigh(A)
    radii = np.sqrt(1 / eigvals)
    
    # 绘制点云
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', s=10, label="Points")
    
    # 绘制椭球
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x = radii[0] * np.cos(u) * np.sin(v)
    y = radii[1] * np.sin(u) * np.sin(v)
    z = radii[2] * np.cos(v)
    
    # 使用椭球的特征向量来旋转椭球
    ellipsoid_points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
    ellipsoid_points = np.dot(ellipsoid_points, eigvecs.T)  # 旋转椭球
    x_ellipsoid = ellipsoid_points[:, 0]
    y_ellipsoid = ellipsoid_points[:, 1]
    z_ellipsoid = ellipsoid_points[:, 2]
    
    ax.plot_surface(x_ellipsoid.reshape(x.shape), y_ellipsoid.reshape(y.shape), z_ellipsoid.reshape(z.shape), 
                    color='r', alpha=0.3, rstride=5, cstride=5, label="Enclosing Ellipsoid")
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

# 创建一个随机点云数据（100个点）
np.random.seed(42)
points = np.random.randn(100, 3)

# 初始化A矩阵为单位矩阵
A0 = np.eye(3).flatten()

# 优化问题：最大化行列式，约束条件为所有点都在椭球内
constraints = [{'type': 'ineq', 'fun': lambda A: constraint(A, points)}]

# 执行优化
result = minimize(objective, A0, args=(points,), constraints=constraints)

# 获取最终的最优A矩阵
A_optimal = result.x.reshape((3, 3))

# 可视化结果
plot_points_and_ellipsoid(points, A_optimal)
