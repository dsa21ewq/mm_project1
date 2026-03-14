"""绘制不同时间下各把手速度变化图（类似题图4）。

基于问题一输出的 CSV：
- task4_velocities_1s.csv（每秒速度数据）

输出：plot4.png

运行：
    python plot4.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_handle_speeds(csv_vel: str, times: list[int], save_path: str = "plot4.png"):
    df = pd.read_csv(csv_vel)

    # 取把手序号列表
    handle_cols = [c for c in df.columns if c.startswith("handle_") and c.endswith("_v")]
    handle_idx = np.array([int(c.split("_")[1]) for c in handle_cols])

    # 读取各个时刻的速度曲线
    speed_curves = []
    for t in times:
        row = df[df["t"] == t]
        if row.empty:
            speed_curves.append(None)
            continue
        speed_curves.append(row[handle_cols].values.flatten())

    # 计算 y 轴范围（小范围放大）
    all_speeds = np.hstack([s for s in speed_curves if s is not None])
    ymin = np.min(all_speeds)
    ymax = np.max(all_speeds)
    margin = (ymax - ymin) * 0.15
    y_min = ymin - margin
    y_max = ymax + margin

    plt.figure(figsize=(10, 6))

    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for i in range(len(times)):
        speeds = speed_curves[i]
        if speeds is None:
            continue
        plt.plot(handle_idx, speeds, color=colors[i], label=f"{times[i]} s", linewidth=1.5)

    # 画不同时间之间的填充带
    for i in range(len(times) - 1):
        s1 = speed_curves[i]
        s2 = speed_curves[i + 1]
        if s1 is None or s2 is None:
            continue
        plt.fill_between(handle_idx, s1, s2, color=colors[i], alpha=0.15)

    plt.xlabel("Handle index")
    plt.ylabel("Speed (m/s)")
    plt.title("Handle speed distribution at different times")
    plt.ylim(y_min, y_max)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Time")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    cwd = os.getcwd()
    csv_vel = os.path.join(cwd, "task4_velocities_1s.csv")

    plot_handle_speeds(csv_vel, times=[0, 100, 200, 300], save_path="plot4.png")
    print("已生成 plot4.png")


if __name__ == "__main__":
    main()
