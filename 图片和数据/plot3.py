"""绘制“龙头前把手位置”图（类似截图所示）。

基于问题一生成的 CSV：
- task1_head_trajectory_full.csv（高分辨率轨迹）
- task3_positions_1s.csv（每秒位置点）

输出：plot3.png

运行：
    python plot3.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_head_positions(csv_full: str, csv_1s: str, save_path: str = "plot3.png"):
    # 读取龙头轨迹 (高分辨率)
    df_full = pd.read_csv(csv_full)
    # 读取整数秒采样点
    df_1s = pd.read_csv(csv_1s)

    x_full = df_full["x_0"].values
    y_full = df_full["y_0"].values

    x_dots = df_1s["head_front_x"].values
    y_dots = df_1s["head_front_y"].values

    # 画完整阿基米德螺线（从 theta 最小到最大）
    if "theta_0" in df_full.columns:
        theta = df_full["theta_0"].values
        theta_min = float(np.min(theta))
        theta_max = float(np.max(theta))
        A = 0.55 / (2 * np.pi)
        thetas = np.linspace(theta_min, theta_max, 3000)
        x_spiral = A * thetas * np.cos(thetas)
        y_spiral = A * thetas * np.sin(thetas)
    else:
        x_spiral = x_full
        y_spiral = y_full

    plt.figure(figsize=(8, 8))
    plt.plot(x_spiral, y_spiral, color="#bbbbbb", linewidth=1.0, alpha=0.6, label="Spiral")
    plt.plot(x_full, y_full, color="#999999", linewidth=1.0, alpha=0.7)
    plt.scatter(x_dots, y_dots, color="#d62728", s=15, label="1s samples")

    # 可选：标记起点/终点
    if len(x_dots) > 0:
        plt.scatter(x_dots[0], y_dots[0], color="#2ca02c", s=60, label="t=0s")
        plt.scatter(x_dots[-1], y_dots[-1], color="#1f77b4", s=60, label=f"t={int(df_1s['t'].iloc[-1])} s")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Head-front Handle Positions (t=0s–300s)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    cwd = os.getcwd()
    csv_full = os.path.join(cwd, "task1_head_trajectory_full.csv")
    csv_1s = os.path.join(cwd, "task3_positions_1s.csv")

    plot_head_positions(csv_full, csv_1s, save_path="plot3.png")
    print("已生成 plot3.png")


if __name__ == "__main__":
    main()
