import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

num_frames = 20

# 读取数据
xps = []
for frame in range(num_frames):
    xp = np.loadtxt(f"results/xp_{frame}.txt")
    xps.append(xp)

# 创建一个图像窗口
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")


# 更新函数
def update(frame):
    ax.clear()  # 清除旧图像
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.2, 0.8)
    ax.set_zlim(0, 0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([2, 2, 1])
    ax.text2D(0.05, 0.95, f"frame: {frame}", transform=ax.transAxes)

    xp = xps[frame]
    ax.scatter(xp[0, :], xp[1, :], xp[2, :], c="b")
    return (ax,)


# 创建一个动画对象
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

# 展示动画
plt.show()
