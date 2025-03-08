import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

def mvee(points, tol=1e-6):
    """
    计算3D点云的最小包围椭球（MVEE）
    :param points: (n, 3) 的 numpy 数组，表示3D点云
    :param tol: 收敛容差
    :return: 椭球的中心 c 和矩阵 A
    """
    n, d = points.shape  # n 是点的数量，d 是维度（这里是3）
    Q = np.column_stack((points, np.ones(n)))  # 构造增广矩阵
    u = np.ones(n) / n  # 初始化权重

    while True:
        X = Q * u[:, np.newaxis]  # 加权点集
        M = np.dot(Q.T, X)  # 构造矩阵 M
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            break  # 如果 M 不可逆，退出

        # 计算新的权重
        dist = np.sum(np.dot(Q, inv_M) * Q, axis=1)
        new_u = u * dist / d

        # 检查收敛
        if np.linalg.norm(new_u - u) < tol:
            break
        u = new_u

    # 提取椭球参数
    c = np.dot(u, points)  # 椭球中心
    A = np.dot(points.T, np.dot(np.diag(u), points)) - np.outer(c, c)
    A = np.linalg.inv(A) / d  # 椭球矩阵

    return c, A

def plot_ellipsoid(center, matrix, ax, n_points=100):
    """
    在3D图中绘制椭球
    :param center: 椭球中心 (3,)
    :param matrix: 椭球矩阵 (3, 3)
    :param ax: matplotlib的3D坐标轴对象
    :param n_points: 椭球的采样点数
    """
    # 计算椭球的半轴和旋转矩阵
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    radii = 1.0 / np.sqrt(eigenvalues)  # 半轴长度

    # 生成椭球的点
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # 应用旋转和平移
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eigenvectors.T) + center

    # 绘制椭球
    ax.plot_surface(x, y, z, color='r', alpha=0.2)

# 示例点云
np.random.seed(42)
points = np.random.rand(30, 3)  # 30个3D点

# 计算最小包围椭球
center, matrix = mvee(points)

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点云
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='Points')

# 绘制椭球
plot_ellipsoid(center, matrix, ax)

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D Point Cloud with Minimum Volume Enclosing Ellipsoid')
plt.show()