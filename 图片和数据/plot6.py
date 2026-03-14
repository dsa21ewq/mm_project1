"""Plot 6: board geometry relationship around handle i (Figure 6 style).

Uses positions from task3_positions_1s.csv and draws:
- handle i / handle i+1 segment
- two outer points A_i and B_i at (a,b) offsets

Based on PDF formulas:
- Δx = x_{i+1}-x_i, Δy = y_{i+1}-y_i
- sin α_i = -Δy / sqrt(Δx^2 + Δy^2)
- cos α_i = -Δx / sqrt(Δx^2 + Δy^2)

Set a = 0.275 m (hole center to nearest head) and b = 0.15 m (top-bottom half width)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_handle_coord(df_row: pd.Series, i: int):
    if i == 0:
        return float(df_row['head_front_x']), float(df_row['head_front_y'])
    elif i == 223:
        return float(df_row['tail_back_x']), float(df_row['tail_back_y'])
    else:
        return float(df_row[f'handle_{i}_x']), float(df_row[f'handle_{i}_y'])


def plot_board_geometry(csv_pos: str, save_path: str = 'plot6.png'):
    df = pd.read_csv(csv_pos)
    row = df.iloc[0]  # first time snapshot

    i = 100
    a = 0.275
    b = 0.15

    P = np.array(get_handle_coord(row, i))
    Q = np.array(get_handle_coord(row, i + 1))

    d = Q - P
    L = np.linalg.norm(d)
    if L < 1e-8:
        raise ValueError('handles i and i+1 coincide')

    u = d / L
    n = np.array([-u[1], u[0]])

    topP = P + b * n
    botP = P - b * n
    topQ = Q + b * n
    botQ = Q - b * n

    P_a = P + a * u
    A_i = P_a + b * n
    B_i = P_a - b * n

    plt.figure(figsize=(12, 6))

    # board edges
    plt.plot([topP[0], topQ[0]], [topP[1], topQ[1]], color='black', linewidth=2)
    plt.plot([botP[0], botQ[0]], [botP[1], botQ[1]], color='black', linewidth=2)
    plt.plot([topP[0], botP[0]], [topP[1], botP[1]], color='black', linewidth=2)
    plt.plot([topQ[0], botQ[0]], [topQ[1], botQ[1]], color='black', linewidth=2)

    # center line
    plt.plot([P[0], Q[0]], [P[1], Q[1]], '--', color='gray', linewidth=1.5)

    # front triangle small a_i/a/b
    plt.plot([P[0], P_a[0]], [P[1], P_a[1]], color='black', linewidth=2)
    plt.plot([P_a[0], A_i[0]], [P_a[1], A_i[1]], color='black', linewidth=2)
    plt.plot([P[0], A_i[0]], [P[1], A_i[1]], color='black', linewidth=1.3)

    plt.scatter([P[0], Q[0], A_i[0], B_i[0]], [P[1], Q[1], A_i[1], B_i[1]], c='red', s=50)

    plt.text(P[0], P[1], f'(x_{i}, y_{i})', fontsize=11, ha='right', va='bottom')
    plt.text(Q[0], Q[1], f'(x_{i+1}, y_{i+1})', fontsize=11, ha='left', va='top')
    plt.text(A_i[0], A_i[1], 'A_i', fontsize=12, color='blue')
    plt.text(B_i[0], B_i[1], 'B_i', fontsize=12, color='blue')

    pa_mid = P + 0.5 * (P_a - P)
    plt.text(pa_mid[0], pa_mid[1], 'a', color='blue', fontsize=10)

    pai_mid = P + 0.5 * (A_i - P)
    plt.text(pai_mid[0], pai_mid[1], 'a_i', color='blue', fontsize=10)

    b_mid = 0.5 * (A_i + B_i)
    plt.text(b_mid[0], b_mid[1], 'b', color='blue', fontsize=10)

    # alpha arrow
    alpha_end = P + 0.25 * u
    plt.annotate('', xy=alpha_end, xytext=P, arrowprops=dict(arrowstyle='->', color='green'))
    plt.annotate('', xy=P + 0.25 * n, xytext=P, arrowprops=dict(arrowstyle='->', color='green'))
    plt.text(P[0] + 0.18 * u[0] + 0.04 * n[0], P[1] + 0.18 * u[1] + 0.04 * n[1], r'$	heta_i$', fontsize=12, color='green')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Figure 6 style geometry for i={i}')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    cwd = os.getcwd()
    csv_pos = os.path.join(cwd, 'task3_positions_1s.csv')
    plot_board_geometry(csv_pos)
    print('已生成 plot6.png')
