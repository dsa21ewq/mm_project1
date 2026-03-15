import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from problem1 import (
    rk4_step_theta_pitch,
    solve_all_thetas_at_one_time_pitch,
    build_all_rectangles_at_one_time,
    min_pairwise_clearance_at_one_time,
)


PITCH = 0.55
NUM_HANDLES = 224
WIDTH = 0.30
TARGET_LINE = 0.15

T_START = 380.0
T_END = 420.0
DT_PLOT = 0.1


def integrate_theta(theta_start: float, t_start: float, t_end: float, dt: float, pitch: float) -> float:
    theta = float(theta_start)
    n = int((t_end - t_start) // dt)
    for _ in range(n):
        theta = rk4_step_theta_pitch(theta, dt, pitch)

    rem = (t_end - t_start) - n * dt
    if rem > 1e-12:
        theta = rk4_step_theta_pitch(theta, rem, pitch)
    return theta


def theta_to_xy(thetas: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    a = pitch / (2 * np.pi)
    r = a * thetas
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    return x, y


def min_clearance_metric(x: np.ndarray, y: np.ndarray) -> float:
    rects = build_all_rectangles_at_one_time(x, y, width=WIDTH)
    res = min_pairwise_clearance_at_one_time(rects, center_dist_threshold=1.5)
    return float(res["min_dist"])


def main() -> None:
    df_head = pd.read_csv("task1_head_trajectory_full.csv", encoding="utf-8-sig")
    base_t = float(df_head["t"].iloc[-1])
    base_theta = float(df_head["theta_0"].iloc[-1])

    theta_start = integrate_theta(base_theta, base_t, T_START, dt=0.001, pitch=PITCH)

    times = np.arange(T_START, T_END + 1e-12, DT_PLOT)
    theta0 = theta_start
    min_dists = []

    for k, t in enumerate(times):
        thetas = solve_all_thetas_at_one_time_pitch(theta0, NUM_HANDLES, PITCH)
        x, y = theta_to_xy(thetas, PITCH)
        dmin = min_clearance_metric(x, y)
        min_dists.append(dmin)

        if k < len(times) - 1:
            theta0 = rk4_step_theta_pitch(theta0, DT_PLOT, PITCH)

    out_df = pd.DataFrame({
        "t": times,
        "min_distance": min_dists,
    })
    out_df.to_csv("plot9_data.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8.4, 5.4))
    plt.plot(times, min_dists, color="#4C78A8", linewidth=2)
    plt.axhline(TARGET_LINE, color="#D65F5F", linestyle="--", linewidth=1.8, alpha=0.8)
    plt.xlim(T_START, T_END)
    plt.xlabel("Time t (s)")
    plt.ylabel("Min distance (m)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("plot9.png", dpi=300)
    plt.close()

    print("saved plot9.png")
    print("saved plot9_data.csv")
    print(f"global min in range = {np.min(min_dists):.6f} m")


if __name__ == "__main__":
    main()
