"""绘制相邻把手几何关系示意图（类似图2）。

该脚本绘制阿基米德螺线的局部段，并标注：
- r_n, r_{n+1}（极径）
- 之间的距离 l
- 对应的极角差 theta_{n+1}-theta_n

运行方式：
    python plot2.py

输出：
    plot2.png
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# 螺距参数，可调整来改变螺旋密度
PITCH = 0.55
A = PITCH / (2 * math.pi)


def spiral_r(theta: float) -> float:
    return A * theta


def spiral_xy(theta: float) -> tuple[float, float]:
    r = spiral_r(theta)
    return r * math.cos(theta), r * math.sin(theta)


def plot_adjacent_handles(theta_n: float,
                          dtheta: float,
                          theta_min: float = None,
                          theta_max: float = None,
                          save_path: str = "plot2.png"):
    """绘制螺线与相邻把手几何关系的示意图。"""

    if theta_min is None:
        theta_min = theta_n - 0.5
    if theta_max is None:
        theta_max = theta_n + dtheta + 0.5

    thetas = np.linspace(theta_min, theta_max, 1000)
    xy = np.array([spiral_xy(th) for th in thetas])

    # 相邻把手
    theta_np1 = theta_n + dtheta
    p_n = np.array(spiral_xy(theta_n))
    p_np1 = np.array(spiral_xy(theta_np1))

    # 原点
    origin = np.array([0.0, 0.0])

    # 绘制
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xy[:, 0], xy[:, 1], color="#c0392b", linewidth=2, label="Spiral")

    # 点与线
    ax.scatter(*p_n, color="#2980b9", s=50, zorder=5)
    ax.scatter(*p_np1, color="#27ae60", s=50, zorder=5)

    ax.plot([origin[0], p_n[0]], [origin[1], p_n[1]], '--', color="#2980b9", label=r"$r_n$");
    ax.plot([origin[0], p_np1[0]], [origin[1], p_np1[1]], '--', color="#27ae60", label=r"$r_{n+1}$");
    ax.plot([p_n[0], p_np1[0]], [p_n[1], p_np1[1]], '-k', linewidth=2, label=r"$l$");

    # 标注
    ax.text(*(p_n + np.array([0.1, 0.1])), "$(r_n, \\theta_n)$", fontsize=12, color="#2980b9")
    ax.text(*(p_np1 + np.array([0.1, 0.1])), "$(r_{n+1}, \\theta_{n+1})$", fontsize=12, color="#27ae60")

    # 角度标注
    # 在原点附近用弧标出角度差
    angle_lines = np.linspace(theta_n, theta_np1, 30)
    angle_arc = np.array([spiral_r(theta_n) * 0.3 * np.array([math.cos(t), math.sin(t)]) for t in angle_lines])
    ax.plot(angle_arc[:, 0], angle_arc[:, 1], color="gray", linestyle='--')
    mid_angle = 0.5 * (theta_n + theta_np1)
    # 使用双斜杠以避免在字符串中出现转义字符（例如 "\t"）导致的显示问题
    arc_label = "$\\theta_{n+1}-\\theta_n$"
    ax.text(0.35 * A * math.cos(mid_angle), 0.35 * A * math.sin(mid_angle), arc_label, fontsize=12)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Adjacent Handle Geometry")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # 可修改的参数
    theta_n = 6.0  # 任选一个 theta_n
    dtheta = 0.8   # 选择一个相邻把手的角度差（可与实际间距对应）

    plot_adjacent_handles(theta_n=theta_n, dtheta=dtheta, save_path="plot2.png")
    print("已生成 plot2.png")
