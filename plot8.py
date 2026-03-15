import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from problem1 import (
    rk4_step_theta_pitch,
    solve_all_thetas_at_one_time_pitch,
    spiral_xy_pitch,
    build_all_rectangles_at_one_time,
    detect_collisions_at_one_time,
    min_pairwise_clearance_at_one_time,
)


def integrate_theta(theta_start: float, t_start: float, t_end: float, dt: float, pitch: float) -> float:
    theta = float(theta_start)
    n = int((t_end - t_start) // dt)
    for _ in range(n):
        theta = rk4_step_theta_pitch(theta, dt, pitch)

    # Last fractional step for exact end time
    rem = (t_end - t_start) - n * dt
    if rem > 1e-12:
        theta = rk4_step_theta_pitch(theta, rem, pitch)
    return theta


def main() -> None:
    # Fixed by user requirement
    target_t = 412.473838
    pitch = 0.55
    num_handles = 224
    width = 0.30
    center_dist_threshold = 1.5

    # Use existing CSV as reference base (0~300s head trajectory)
    head_ref = pd.read_csv("task1_head_trajectory_full.csv", encoding="utf-8-sig")
    base_t = float(head_ref["t"].iloc[-1])
    base_theta = float(head_ref["theta_0"].iloc[-1])

    if target_t < base_t:
        # Fallback: if user chooses earlier time, use nearest from CSV
        idx = int(np.argmin(np.abs(head_ref["t"].values - target_t)))
        theta0 = float(head_ref["theta_0"].iloc[idx])
    else:
        theta0 = integrate_theta(
            theta_start=base_theta,
            t_start=base_t,
            t_end=target_t,
            dt=0.001,
            pitch=pitch,
        )

    # Solve all handle angles at this exact time
    thetas = solve_all_thetas_at_one_time_pitch(
        theta0=theta0,
        num_handles=num_handles,
        pitch=pitch,
    )

    # Convert to xy
    coords = np.array([spiral_xy_pitch(th, pitch) for th in thetas])
    x_row = coords[:, 0]
    y_row = coords[:, 1]

    # Build bench rectangles and detect collision state
    rects = build_all_rectangles_at_one_time(x_row, y_row, width=width)
    collision = detect_collisions_at_one_time(rects, center_dist_threshold)
    clearance = min_pairwise_clearance_at_one_time(rects, center_dist_threshold)

    # Plot background spiral (inward trajectory)
    theta_max = float(np.max(thetas)) + 2.0
    theta_bg = np.linspace(0.0, theta_max, 4000)
    xy_bg = np.array([spiral_xy_pitch(t, pitch) for t in theta_bg])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xy_bg[:, 0], xy_bg[:, 1], color="#c8c8c8", linewidth=2.0, alpha=0.9)

    # Draw all benches
    for r in rects:
        ax.add_patch(
            Polygon(r, closed=True, facecolor="none", edgecolor="#3f5ea9", linewidth=1.1, alpha=0.9)
        )

    # Highlight first collision pair if found
    if collision["collision"] and len(collision["pairs"]) > 0:
        i0, i1 = collision["pairs"][0]
        for i in (i0, i1):
            ax.add_patch(
                Polygon(rects[i], closed=True, facecolor="none", edgecolor="#e74c3c", linewidth=1.8)
            )
    # If no collision, still highlight the nearest pair to show near-contact status
    elif clearance["pair"] is not None:
        i0, i1 = clearance["pair"]
        for i in (i0, i1):
            ax.add_patch(
                Polygon(rects[i], closed=True, facecolor="none", edgecolor="#f39c12", linewidth=1.8)
            )

    # Handle points
    ax.scatter(x_row, y_row, s=8, c="#d9534f", zorder=3)

    ax.set_aspect("equal", "box")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"t={target_t:.6f}s, min_clearance={clearance['min_dist']:.6f} m")
    ax.grid(True, linestyle="--", alpha=0.25)

    out_img = "problem2_figure8_t412473838.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    plt.close(fig)

    # Save snapshot CSV for this time
    out_csv = "problem2_collision_snapshot_t412473838.csv"
    snap = pd.DataFrame(
        {
            "handle_id": np.arange(num_handles),
            "theta": thetas,
            "x": x_row,
            "y": y_row,
        }
    )
    snap.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"saved image: {out_img}")
    print(f"saved csv:   {out_csv}")
    print(f"collision:   {collision['collision']}")
    print(f"min_dist:    {clearance['min_dist']}")
    if collision["collision"]:
        print(f"pairs:       {collision['pairs'][:5]}")
    elif clearance["pair"] is not None:
        print(f"nearest_pair:{clearance['pair']}")


if __name__ == "__main__":
    main()
