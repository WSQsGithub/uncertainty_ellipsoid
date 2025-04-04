import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import cProfile

def mvee(points, tol=1e-6):
    """
    计算3D点云的最小包围椭球（MVEE）
    :param points: (n, 3) 的 numpy 数组，表示3D点云
    :param tol: 收敛容差
    :return: 椭球的中心 c 和矩阵 A
    """
    n, d = points.shape
    Q = np.column_stack((points, np.ones(n)))  # Shape: (n, d+1)
    u = np.ones(n) / n

    while True:
        # Compute X = Q * diag(u) * Q'
        X = np.dot(Q.T, np.dot(np.diag(u), Q))  # Shape: (d+1, d+1)
        # Compute M = diag(Q' * inv(X) * Q)
        M = np.sum(np.dot(Q, np.dot(np.linalg.inv(X), Q.T)) * np.eye(n), axis=1)
        maximum = np.max(M)
        if maximum > d + 1:
            step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
            new_u = (1 - step_size) * u
            new_u[np.argmax(M)] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u
        else:
            break

        if err < tol:
            break

    c = np.dot(u, points)  # Center
    A = np.dot(points.T, np.dot(np.diag(u), points)) - np.outer(c, c)
    A = np.linalg.inv(A) / d  # Shape matrix

    # Ensure A is positive definite
    eigenvalues, _ = np.linalg.eigh(A)
    if np.any(eigenvalues <= 0):
        print("Warning: Matrix A has non-positive eigenvalues, adjusting.")
        A = A + np.eye(3) * 1e-6

    return c, A

def plot_ellipsoid(center, matrix, ax, n_points=100, scale=1.0):
    """
    在3D图中绘制椭球
    :param center: 椭球中心 (3,)
    :param matrix: 椭球矩阵 (3, 3)
    :param ax: matplotlib的3D坐标轴对象
    :param n_points: 椭球的采样点数
    :param scale: 缩放因子，调整椭球大小
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    radii = scale / np.sqrt(np.abs(eigenvalues))

    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    u, v = np.meshgrid(u, v)
    x = radii[0] * np.cos(u) * np.sin(v)
    y = radii[1] * np.sin(u) * np.sin(v)
    z = radii[2] * np.cos(v)

    points_3d = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    rotated = np.dot(points_3d, eigenvectors)
    x = rotated[:, 0].reshape(x.shape) + center[0]
    y = rotated[:, 1].reshape(y.shape) + center[1]
    z = rotated[:, 2].reshape(z.shape) + center[2]

    ax.plot_surface(x, y, z, color='g', alpha=0.5)

# 示例点云
# np.random.seed(42)
points = np.random.rand(1000, 3)

# 计算凸包
hull = ConvexHull(points)
hull_points = points[hull.vertices]  # 提取凸包顶点

# 计算最小包围椭球
profiler = cProfile.Profile()
profiler.enable()
center, matrix = mvee(hull_points)
profiler.disable()
profiler.print_stats(sort='cumulative')

# 可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制所有点云
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='All Points', alpha=0.5)
# 绘制凸包点
ax.scatter(hull_points[:, 0], hull_points[:, 1], hull_points[:, 2], c='r', label='Hull Points', alpha=0.8)

# 绘制椭球
plot_ellipsoid(center, matrix, ax, scale=1.0)

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('3D Point Cloud with Minimum Volume Enclosing Ellipsoid (Convex Hull)')

# 设置轴限
ax.set_xlim([min(points[:, 0].min(), center[0]) - 0.5, max(points[:, 0].max(), center[0]) + 0.5])
ax.set_ylim([min(points[:, 1].min(), center[1]) - 0.5, max(points[:, 1].max(), center[1]) + 0.5])
ax.set_zlim([min(points[:, 2].min(), center[2]) - 0.5, max(points[:, 2].max(), center[2]) + 0.5])

# Calculate the percentage of points outside the ellipsoid
def is_point_inside_ellipsoid(point, center, matrix):
    p = point - center
    return np.dot(p, np.dot(matrix, p)) <= 1

outside_points = 0
for point in points:
    if not is_point_inside_ellipsoid(point, center, matrix):
        outside_points += 1

percentage_outside = (outside_points / len(points)) * 100
print(f"Percentage of points outside the ellipsoid: {percentage_outside:.2f}%")

plt.show()