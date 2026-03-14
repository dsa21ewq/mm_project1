"""绘制极角随时间变化曲线。

基于问题一输出的 CSV：
- task1_head_trajectory_full.csv

输出：plot_theta.png

命令：
    python plot_theta.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_polar_angle(csv_file: str, save_path: str = "plot_theta.png"):
    # 按 problem1 计算办法，直接获取 theta_n（极角）矩阵
    from problem1 import solve_head_trajectory, solve_all_thetas_over_time

    df_head = solve_head_trajectory(0.0, 300.0, 1.0, 32.0 * np.pi)
    theta_mat = solve_all_thetas_over_time(df_head, 224)

    t = df_head["t"].values
    selected_indices = [0, 51, 101, 151, 201, 223]
    selected_labels = ["head front", "handle 51", "handle 101", "handle 151", "handle 201", "tail back"]

    theta_rad = theta_mat[:, selected_indices]
    colors = ["#9400D3", "#4B0082", "#0000FF", "#00FF00", "#FFFF00", "#FF0000"]  # violet, indigo, blue, green, yellow, red

    plt.figure(figsize=(10, 6))
    for label, col, c in zip(selected_labels, range(theta_rad.shape[1]), colors):
        plt.plot(t, theta_rad[:, col], linewidth=1.8, label=label, color=c)

    plt.xlabel("Time t (s)")
    plt.ylabel("Polar angle θ (rad)")
    plt.title("Polar angle of selected handles vs time (0s-300s)")
    plt.ylim(50, 150)
    plt.yticks(np.arange(60, 151, 10))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    cwd = os.getcwd()
    csv_file = os.path.join(cwd, "task3_positions_1s.csv")
    plot_polar_angle(csv_file, save_path="plot5.png")
    print("已生成 plot5.png")


if __name__ == "__main__":
    main()
