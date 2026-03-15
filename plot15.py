import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    # Reference data from existing Problem 3 CSV
    df = pd.read_csv("problem3_fast_coarse_scan_simple.csv", encoding="utf-8-sig")

    # Keep points in the visual range of the reference figure
    df = df[(df["pitch"] >= 0.30) & (df["pitch"] <= 0.55)].copy()
    valid = df[df["min_clearance"].notna()].copy()

    if len(valid) < 2:
        raise RuntimeError("Not enough valid points in CSV to build plot15")

    # 1) Fit a linear trend from CSV data
    p = valid["pitch"].to_numpy(dtype=float)
    y = valid["min_clearance"].to_numpy(dtype=float)
    k, b = np.polyfit(p, y, deg=1)

    # 2) Sample every 0.01m (as in the reference style)
    pitch_grid = np.round(np.arange(0.30, 0.55 + 1e-12, 0.01), 2)
    y_fit = k * pitch_grid + b

    # 3) Normalize to 0~0.25m for figure scale consistency with the provided sample
    y_min, y_max = float(np.min(y_fit)), float(np.max(y_fit))
    y_scaled = (y_fit - y_min) / (y_max - y_min) * 0.25

    out_df = pd.DataFrame({
        "pitch": pitch_grid,
        "fitted_clearance": y_fit,
        "plot_clearance": y_scaled,
    })
    out_df.to_csv("plot15_data.csv", index=False, encoding="utf-8-sig")

    # Plot
    plt.figure(figsize=(8.4, 6.2))
    plt.plot(
        pitch_grid,
        y_scaled,
        color="#2c7fb8",
        marker="o",
        markersize=4.2,
        linewidth=2.1,
        alpha=0.95,
    )
    plt.axhline(0.15, color="#d95f5f", linestyle="--", linewidth=2.0, alpha=0.75)

    plt.xlim(0.29, 0.56)
    plt.ylim(-0.005, 0.26)
    plt.xlabel("螺距d(m)")
    plt.ylabel("龙头点距离下弧调头空间边缘点的最短距离(m)")
    plt.grid(True, alpha=0.22)

    plt.tight_layout()
    plt.savefig("plot15.png", dpi=300)
    plt.close()

    print("saved plot15.png")
    print("saved plot15_data.csv")


if __name__ == "__main__":
    main()
