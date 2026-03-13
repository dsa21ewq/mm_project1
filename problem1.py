import math
import numpy as np
import pandas as pd

# =========================
# 基本参数
# =========================
PITCH = 0.55                    # 螺距 d
A = PITCH / (2 * math.pi)       # a = d / (2pi)
V_HEAD = 1.0                    # 龙头前把手速度
THETA0_INIT = 32 * math.pi      # 第16圈起点
T_START = 0.0
T_END = 300.0

# 把手间距
L_HEAD = 2.86                   # 龙头板把手间距
L_BODY = 1.65                   # 龙身/龙尾板把手间距

# 这里先按“总把手数”建模
# 你后续可按题目实际数量修改
NUM_HANDLES = 224

# 内部数值步长
DT = 0.01


# =========================
# 工具函数
# =========================
def spiral_r(theta: float) -> float:
    """阿基米德螺线极径 r=a*theta"""
    return A * theta


def spiral_xy(theta: float) -> tuple[float, float]:
    """由 theta 转直角坐标"""
    r = spiral_r(theta)
    return r * math.cos(theta), r * math.sin(theta)


def handle_distance(theta1: float, theta2: float) -> float:
    """
    同一阿基米德螺线上两个点的欧氏距离
    """
    r1 = spiral_r(theta1)
    r2 = spiral_r(theta2)
    return math.sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * math.cos(theta2 - theta1))


def distance_equation(theta_next: float, theta_curr: float, target_len: float) -> float:
    """
    递推方程 f(theta_next)=0
    """
    return handle_distance(theta_curr, theta_next) - target_len


def theta_dot(theta: float) -> float:
    """
    龙头前把手极角微分方程:
    dtheta/dt = -1 / (a * sqrt(1 + theta^2))
    """
    return -1.0 / (A * math.sqrt(1.0 + theta * theta))


def rk4_step_theta(theta: float, dt: float) -> float:
    """
    用 RK4 积分龙头 theta
    """
    k1 = theta_dot(theta)
    k2 = theta_dot(theta + 0.5 * dt * k1)
    k3 = theta_dot(theta + 0.5 * dt * k2)
    k4 = theta_dot(theta + dt * k3)
    return theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_next_theta(theta_curr: float, target_len: float,
                     step_out: float = 0.02,
                     max_expand: int = 20000,
                     tol: float = 1e-10,
                     max_iter: int = 100) -> float:
    """
    已知 theta_curr，求外侧相邻把手 theta_next
    要求 theta_next > theta_curr，且满足两点距离 = target_len

    方法：
    1. 从 theta_curr 往外逐步扩大，找到变号区间
    2. 再做二分法求根
    """
    left = theta_curr
    f_left = distance_equation(left, theta_curr, target_len)

    right = left + step_out
    f_right = distance_equation(right, theta_curr, target_len)

    expand_count = 0
    while f_left * f_right > 0 and expand_count < max_expand:
        right += step_out
        f_right = distance_equation(right, theta_curr, target_len)
        expand_count += 1

    if f_left * f_right > 0:
        raise RuntimeError(
            f"未找到根区间: theta_curr={theta_curr:.6f}, target_len={target_len}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = distance_equation(mid, theta_curr, target_len)

        if abs(f_mid) < tol or (right - left) < tol:
            return mid

        if f_left * f_mid <= 0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid

    return 0.5 * (left + right)

#problem1
#=========================任务1========================
def solve_head_trajectory(t_start: float, t_end: float, dt: float,
                          theta_init: float) -> pd.DataFrame:
    """
    数值积分龙头前把手轨迹，返回所有离散时刻的 theta, x, y
    """
    times = np.arange(t_start, t_end + dt, dt)
    thetas = np.zeros(len(times), dtype=float)
    xs = np.zeros(len(times), dtype=float)
    ys = np.zeros(len(times), dtype=float)

    theta = theta_init
    for i, t in enumerate(times):
        thetas[i] = theta
        x, y = spiral_xy(theta)
        xs[i] = x
        ys[i] = y

        if i < len(times) - 1:
            theta = rk4_step_theta(theta, dt)

    df = pd.DataFrame({
        "t": times,
        "theta_0": thetas,
        "x_0": xs,
        "y_0": ys,
    })
    return df


# df_head = solve_head_trajectory(T_START, T_END, DT, THETA0_INIT)
# print(df_head.head())
# print(df_head.tail())

#=========================任务2========================
def get_handle_spacing(handle_index: int) -> float:
    """
    handle_index 表示当前已知的是第 handle_index 个把手，
    要求下一把手时所对应的把手间距 l_n

    n=0 对应 龙头前把手 -> 龙头后把手，用 L_HEAD
    n>=1 对应其余相邻把手，用 L_BODY
    """
    if handle_index == 0:
        return L_HEAD
    return L_BODY


def solve_all_thetas_at_one_time(theta0: float, num_handles: int) -> np.ndarray:
    """
    已知某一时刻龙头前把手 theta0，
    递推求所有把手 theta
    """
    thetas = np.zeros(num_handles, dtype=float)
    thetas[0] = theta0

    for n in range(num_handles - 1):
        target_len = get_handle_spacing(n)
        thetas[n + 1] = solve_next_theta(thetas[n], target_len)

    return thetas


def solve_all_thetas_over_time(df_head: pd.DataFrame, num_handles: int) -> np.ndarray:
    """
    对所有时刻递推全部把手 theta
    返回 shape = (num_times, num_handles)
    """
    num_times = len(df_head)
    theta_mat = np.zeros((num_times, num_handles), dtype=float)

    for i in range(num_times):
        theta0 = df_head.loc[i, "theta_0"]
        theta_mat[i, :] = solve_all_thetas_at_one_time(theta0, num_handles)

        if i % 1000 == 0:
            print(f"theta递推进度: {i}/{num_times}")

    return theta_mat

# theta_mat = solve_all_thetas_over_time(df_head, NUM_HANDLES)
# print(theta_mat.shape)   # (时间步数, 把手数)

#=========================任务3========================

def theta_matrix_to_xy(theta_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    把 theta 矩阵转成 x, y 矩阵
    """
    r_mat = A * theta_mat
    x_mat = r_mat * np.cos(theta_mat)
    y_mat = r_mat * np.sin(theta_mat)
    return x_mat, y_mat


def sample_integer_seconds(df_head: pd.DataFrame, x_mat: np.ndarray, y_mat: np.ndarray,
                           dt: float) -> pd.DataFrame:
    """
    提取整数秒结果，并整理成宽表
    每一行对应一个整数秒
    """
    second_indices = []
    second_values = []
    max_sec = int(round(df_head["t"].iloc[-1]))

    for sec in range(max_sec + 1):
        idx = int(round(sec / dt))
        second_indices.append(idx)
        second_values.append(sec)

    rows = []
    num_handles = x_mat.shape[1]

    for sec, idx in zip(second_values, second_indices):
        row = {"t": sec}

        # 龙头前把手
        row["head_front_x"] = x_mat[idx, 0]
        row["head_front_y"] = y_mat[idx, 0]

        # 龙尾前把手、龙尾后把手
        # 这里默认最后两个把手分别视为龙尾前/后把手
        row["tail_front_x"] = x_mat[idx, num_handles - 2]
        row["tail_front_y"] = y_mat[idx, num_handles - 2]
        row["tail_back_x"] = x_mat[idx, num_handles - 1]
        row["tail_back_y"] = y_mat[idx, num_handles - 1]

        # 各节龙身前把手
        # 这里示意性输出 handle_1 ~ handle_(num_handles-2)
        for j in range(1, num_handles - 1):
            row[f"handle_{j}_x"] = x_mat[idx, j]
            row[f"handle_{j}_y"] = y_mat[idx, j]

        rows.append(row)

    return pd.DataFrame(rows)

# x_mat, y_mat = theta_matrix_to_xy(theta_mat)
# df_pos_1s = sample_integer_seconds(df_head, x_mat, y_mat, DT)
# print(df_pos_1s.head())

#  ========================任务4========================


def compute_velocity_matrix(x_mat: np.ndarray, y_mat: np.ndarray, dt: float) -> np.ndarray:
    """
    差分法计算各把手速度
    返回 shape = (num_times, num_handles)
    """
    num_times, num_handles = x_mat.shape
    v_mat = np.zeros((num_times, num_handles), dtype=float)

    # 前向差分
    for i in range(num_times - 1):
        dx = x_mat[i + 1, :] - x_mat[i, :]
        dy = y_mat[i + 1, :] - y_mat[i, :]
        v_mat[i, :] = np.sqrt(dx * dx + dy * dy) / dt

    # 最后一个时刻用后退近似
    v_mat[-1, :] = v_mat[-2, :]
    return v_mat


def sample_integer_second_velocity(df_head: pd.DataFrame, v_mat: np.ndarray, dt: float) -> pd.DataFrame:
    """
    提取整数秒的速度结果
    """
    rows = []
    num_handles = v_mat.shape[1]
    max_sec = int(round(df_head["t"].iloc[-1]))

    for sec in range(max_sec + 1):
        idx = int(round(sec / dt))
        row = {"t": sec}

        row["head_front_v"] = v_mat[idx, 0]
        row["tail_front_v"] = v_mat[idx, num_handles - 2]
        row["tail_back_v"] = v_mat[idx, num_handles - 1]

        for j in range(1, num_handles - 1):
            row[f"handle_{j}_v"] = v_mat[idx, j]

        rows.append(row)

    return pd.DataFrame(rows)


# v_mat = compute_velocity_matrix(x_mat, y_mat, DT)
# df_vel_1s = sample_integer_second_velocity(df_head, v_mat, DT)
# print(df_vel_1s.head())


#任务1的封装
def solve_problem1(
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    export_csv: bool = True,
) -> dict:
    """
    问题1总封装：
    1. 求龙头前把手轨迹
    2. 递推全部把手极角
    3. 转为直角坐标
    4. 计算速度
    5. 提取整数秒位置/速度结果
    6. 导出CSV（可选）

    返回：
    {
        "df_head": ...,
        "theta_mat": ...,
        "x_mat": ...,
        "y_mat": ...,
        "v_mat": ...,
        "df_pos_1s": ...,
        "df_vel_1s": ...,
        "df_result_1s": ...
    }
    """
    print("问题1 - 任务1：求龙头前把手轨迹")
    df_head = solve_head_trajectory(t_start, t_end, dt, theta0_init)

    print("问题1 - 任务2：递推全部把手极角")
    theta_mat = solve_all_thetas_over_time(df_head, num_handles)

    print("问题1 - 任务3：转成直角坐标并提取整数秒位置")
    x_mat, y_mat = theta_matrix_to_xy(theta_mat)
    df_pos_1s = sample_integer_seconds(df_head, x_mat, y_mat, dt)

    print("问题1 - 任务4：计算速度并提取整数秒结果")
    v_mat = compute_velocity_matrix(x_mat, y_mat, dt)
    df_vel_1s = sample_integer_second_velocity(df_head, v_mat, dt)

    df_result_1s = pd.merge(df_pos_1s, df_vel_1s, on="t", how="left")

    if export_csv:
        df_head.to_csv("task1_head_trajectory_full.csv", index=False, encoding="utf-8-sig")
        df_pos_1s.to_csv("task3_positions_1s.csv", index=False, encoding="utf-8-sig")
        df_vel_1s.to_csv("task4_velocities_1s.csv", index=False, encoding="utf-8-sig")
        df_result_1s.to_csv("problem1_result_1s.csv", index=False, encoding="utf-8-sig")

        print("问题1结果已导出：")
        print(" - task1_head_trajectory_full.csv")
        print(" - task3_positions_1s.csv")
        print(" - task4_velocities_1s.csv")
        print(" - problem1_result_1s.csv")

    return {
        "df_head": df_head,
        "theta_mat": theta_mat,
        "x_mat": x_mat,
        "y_mat": y_mat,
        "v_mat": v_mat,
        "df_pos_1s": df_pos_1s,
        "df_vel_1s": df_vel_1s,
        "df_result_1s": df_result_1s,
    }


#problem2
# ========================任务1========================
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 问题二参数
# =========================
BENCH_WIDTH = 0.30   # 板凳宽度，先用占位值，按题目实际改


def normalize(vec: np.ndarray) -> np.ndarray:
    """向量单位化"""
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("零向量无法单位化")
    return vec / norm


def get_bench_segments_from_handles(num_handles: int) -> list[tuple[int, int]]:
    """
    由把手编号构造每节板凳的前后把手编号对
    默认：
    第0节板凳: handle 0 -> handle 1
    第1节板凳: handle 1 -> handle 2
    ...
    第(num_handles-2)节板凳: handle (num_handles-2) -> handle (num_handles-1)
    """
    return [(i, i + 1) for i in range(num_handles - 1)]


def build_bench_rectangle(front_point: np.ndarray,
                          back_point: np.ndarray,
                          width: float) -> np.ndarray:
    """
    根据一节板凳前后把手坐标，构建矩形四顶点
    返回 shape=(4,2)，顶点顺序为 A,B,C,D
    """
    direction = back_point - front_point
    u = normalize(direction)

    # 法向量
    v = np.array([-u[1], u[0]])

    A = front_point + 0.5 * width * v
    B = front_point - 0.5 * width * v
    C = back_point  - 0.5 * width * v
    D = back_point  + 0.5 * width * v

    return np.vstack([A, B, C, D])


def build_all_rectangles_at_one_time(x_row: np.ndarray,
                                     y_row: np.ndarray,
                                     width: float) -> list[np.ndarray]:
    """
    某一时刻下重建全部板凳矩形
    返回 list，每个元素 shape=(4,2)
    """
    num_handles = len(x_row)
    bench_pairs = get_bench_segments_from_handles(num_handles)

    rectangles = []
    for i_front, i_back in bench_pairs:
        front_point = np.array([x_row[i_front], y_row[i_front]], dtype=float)
        back_point = np.array([x_row[i_back], y_row[i_back]], dtype=float)
        rect = build_bench_rectangle(front_point, back_point, width)
        rectangles.append(rect)

    return rectangles


def build_all_rectangles_over_time(x_mat: np.ndarray,
                                   y_mat: np.ndarray,
                                   width: float) -> list[list[np.ndarray]]:
    """
    对所有时刻重建全部板凳矩形
    返回:
      rectangles_over_time[time_index][bench_index] = shape(4,2)矩形顶点
    """
    all_rectangles = []
    for t_idx in range(x_mat.shape[0]):
        rects = build_all_rectangles_at_one_time(x_mat[t_idx], y_mat[t_idx], width)
        all_rectangles.append(rects)

        if t_idx % 1000 == 0:
            print(f"矩形重建进度: {t_idx}/{x_mat.shape[0]}")

    return all_rectangles
#=========================任务2========================
def rectangle_center(rect: np.ndarray) -> np.ndarray:
    """矩形中心点"""
    return rect.mean(axis=0)


def candidate_pairs_by_center_distance(rectangles: list[np.ndarray],
                                       center_dist_threshold: float) -> list[tuple[int, int]]:
    """
    候选板凳对筛选：
    1. 只保留非相邻板凳 |i-j|>1
    2. 中心点距离小于阈值
    """
    candidates = []
    n = len(rectangles)

    centers = [rectangle_center(rect) for rect in rectangles]

    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1:
                continue

            dist = np.linalg.norm(centers[i] - centers[j])
            if dist <= center_dist_threshold:
                candidates.append((i, j))

    return candidates

#=========================任务3========================
def polygon_edges(poly: np.ndarray) -> list[np.ndarray]:
    """返回多边形边向量列表"""
    edges = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        edges.append(p2 - p1)
    return edges


def perpendicular_axis(edge: np.ndarray) -> np.ndarray:
    """边的法向量轴，并单位化"""
    axis = np.array([-edge[1], edge[0]], dtype=float)
    return normalize(axis)


def project_polygon(poly: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """将多边形投影到轴上"""
    projections = poly @ axis
    return projections.min(), projections.max()


def interval_overlap(interval1: tuple[float, float],
                     interval2: tuple[float, float]) -> bool:
    """判断两个区间是否重叠"""
    a1, a2 = interval1
    b1, b2 = interval2
    return not (a2 < b1 or b2 < a1)


def sat_intersect_rectangle(rect1: np.ndarray, rect2: np.ndarray) -> bool:
    """
    SAT 判定两个凸四边形（矩形）是否相交
    """
    axes = []

    for edge in polygon_edges(rect1):
        axes.append(perpendicular_axis(edge))

    for edge in polygon_edges(rect2):
        axes.append(perpendicular_axis(edge))

    for axis in axes:
        proj1 = project_polygon(rect1, axis)
        proj2 = project_polygon(rect2, axis)

        if not interval_overlap(proj1, proj2):
            return False

    return True


def point_to_segment_distance(point: np.ndarray,
                              seg_a: np.ndarray,
                              seg_b: np.ndarray) -> float:
    """点到线段距离"""
    ab = seg_b - seg_a
    ap = point - seg_a
    denom = np.dot(ab, ab)

    if denom < 1e-12:
        return np.linalg.norm(point - seg_a)

    t = np.dot(ap, ab) / denom
    t = max(0.0, min(1.0, t))
    proj = seg_a + t * ab
    return np.linalg.norm(point - proj)


def segment_to_segment_distance(a1: np.ndarray, a2: np.ndarray,
                                b1: np.ndarray, b2: np.ndarray) -> float:
    """
    近似线段间距离：
    若已相交则返回0，否则取端点到对方线段距离最小值
    """
    # 若所在矩形相交，此函数通常不会单独决定碰撞
    d1 = point_to_segment_distance(a1, b1, b2)
    d2 = point_to_segment_distance(a2, b1, b2)
    d3 = point_to_segment_distance(b1, a1, a2)
    d4 = point_to_segment_distance(b2, a1, a2)
    return min(d1, d2, d3, d4)


def rectangle_min_distance(rect1: np.ndarray, rect2: np.ndarray) -> float:
    """
    两矩形最小距离：
    若相交返回0，否则计算边与边间最小距离
    """
    if sat_intersect_rectangle(rect1, rect2):
        return 0.0

    min_dist = float("inf")
    for i in range(4):
        a1 = rect1[i]
        a2 = rect1[(i + 1) % 4]
        for j in range(4):
            b1 = rect2[j]
            b2 = rect2[(j + 1) % 4]
            d = segment_to_segment_distance(a1, a2, b1, b2)
            min_dist = min(min_dist, d)

    return min_dist


def detect_collisions_at_one_time(rectangles: list[np.ndarray],
                                  center_dist_threshold: float,
                                  eps: float = 1e-9) -> dict:
    """
    某一时刻检测碰撞
    返回:
    {
        "collision": bool,
        "pairs": [(i,j), ...],
        "detail": [
            {"i":..., "j":..., "intersect":..., "min_dist":..., "center_dist":...},
            ...
        ]
    }
    """
    candidates = candidate_pairs_by_center_distance(rectangles, center_dist_threshold)
    detail = []
    collision_pairs = []

    for i, j in candidates:
        rect_i = rectangles[i]
        rect_j = rectangles[j]

        center_dist = np.linalg.norm(rectangle_center(rect_i) - rectangle_center(rect_j))
        intersect = sat_intersect_rectangle(rect_i, rect_j)
        min_dist = rectangle_min_distance(rect_i, rect_j)

        item = {
            "i": i,
            "j": j,
            "intersect": intersect,
            "min_dist": min_dist,
            "center_dist": center_dist,
        }
        detail.append(item)

        if intersect or min_dist <= eps:
            collision_pairs.append((i, j))

    return {
        "collision": len(collision_pairs) > 0,
        "pairs": collision_pairs,
        "detail": detail,
    }


def interpolate_handle_positions(x_mat: np.ndarray,
                                 y_mat: np.ndarray,
                                 time_index_left: int,
                                 alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    在相邻离散时刻之间线性插值
    alpha in [0,1]
    """
    x = (1 - alpha) * x_mat[time_index_left] + alpha * x_mat[time_index_left + 1]
    y = (1 - alpha) * y_mat[time_index_left] + alpha * y_mat[time_index_left + 1]
    return x, y


def collision_state_interpolated(x_mat: np.ndarray,
                                 y_mat: np.ndarray,
                                 left_idx: int,
                                 alpha: float,
                                 width: float,
                                 center_dist_threshold: float) -> dict:
    """
    插值时刻的碰撞状态
    """
    x_row, y_row = interpolate_handle_positions(x_mat, y_mat, left_idx, alpha)
    rects = build_all_rectangles_at_one_time(x_row, y_row, width)
    return detect_collisions_at_one_time(rects, center_dist_threshold)


def refine_first_collision_bisection(x_mat: np.ndarray,
                                     y_mat: np.ndarray,
                                     coarse_index: int,
                                     dt: float,
                                     width: float,
                                     center_dist_threshold: float,
                                     tol: float = 1e-5,
                                     max_iter: int = 50) -> dict:
    """
    在 [coarse_index-1, coarse_index] 区间内二分细化首次碰撞时间
    假定 coarse_index 处第一次检测到碰撞
    """
    if coarse_index == 0:
        left_idx = 0
    else:
        left_idx = coarse_index - 1

    # 左端应无碰撞，右端应有碰撞
    left_alpha = 0.0
    right_alpha = 1.0

    # 若 coarse_index=0，则只能用当前结果
    if coarse_index == 0:
        result = collision_state_interpolated(
            x_mat, y_mat, 0, 0.0, width, center_dist_threshold
        )
        return {
            "time": 0.0,
            "pairs": result["pairs"],
            "detail": result["detail"],
            "left_idx": 0,
            "alpha": 0.0,
        }

    # 二分
    final_result = None
    for _ in range(max_iter):
        mid_alpha = 0.5 * (left_alpha + right_alpha)
        result = collision_state_interpolated(
            x_mat, y_mat, left_idx, mid_alpha, width, center_dist_threshold
        )

        if result["collision"]:
            right_alpha = mid_alpha
            final_result = result
        else:
            left_alpha = mid_alpha

        if (right_alpha - left_alpha) * dt < tol:
            break

    collision_time = (left_idx + right_alpha) * dt
    return {
        "time": collision_time,
        "pairs": final_result["pairs"] if final_result else [],
        "detail": final_result["detail"] if final_result else [],
        "left_idx": left_idx,
        "alpha": right_alpha,
    }

#=========================任务4========================
def find_first_collision_coarse(rectangles_over_time: list[list[np.ndarray]],
                                dt: float,
                                center_dist_threshold: float) -> dict | None:
    """
    粗搜索首次碰撞时刻
    返回:
    {
        "time_index": k,
        "time": k*dt,
        "pairs": [...],
        "detail": [...]
    }
    若未碰撞返回 None
    """
    for t_idx, rectangles in enumerate(rectangles_over_time):
        result = detect_collisions_at_one_time(rectangles, center_dist_threshold)
        if result["collision"]:
            return {
                "time_index": t_idx,
                "time": t_idx * dt,
                "pairs": result["pairs"],
                "detail": result["detail"],
            }

        if t_idx % 1000 == 0:
            print(f"粗搜索进度: {t_idx}/{len(rectangles_over_time)}")

    return None





#====================任务5=====================
def extract_collision_details(rectangles: list[np.ndarray],
                              collision_result: dict) -> list[dict]:
    """
    提取首次碰撞的详细信息
    """
    details = []
    target_pairs = set(collision_result["pairs"])

    for item in collision_result["detail"]:
        pair = (item["i"], item["j"])
        if pair in target_pairs:
            i, j = item["i"], item["j"]
            rect_i = rectangles[i]
            rect_j = rectangles[j]

            details.append({
                "bench_i": i,
                "bench_j": j,
                "center_i": rectangle_center(rect_i),
                "center_j": rectangle_center(rect_j),
                "rect_i": rect_i,
                "rect_j": rect_j,
                "min_dist": item["min_dist"],
                "center_dist": item["center_dist"],
                "intersect": item["intersect"],
            })

    return details


def interpolate_rectangles(x_mat: np.ndarray,
                           y_mat: np.ndarray,
                           left_idx: int,
                           alpha: float,
                           width: float) -> list[np.ndarray]:
    """
    获取插值时刻全部矩形
    """
    x_row, y_row = interpolate_handle_positions(x_mat, y_mat, left_idx, alpha)
    return build_all_rectangles_at_one_time(x_row, y_row, width)


def plot_rectangles(rectangles: list[np.ndarray],
                    collision_pairs: list[tuple[int, int]] | None = None,
                    title: str = "Collision Configuration",
                    save_path: str | None = None) -> None:
    """
    绘制整条板凳龙平面形态
    """
    plt.figure(figsize=(8, 8))

    collision_set = set(collision_pairs) if collision_pairs else set()

    for idx, rect in enumerate(rectangles):
        closed_rect = np.vstack([rect, rect[0]])

        highlight = any(idx in pair for pair in collision_set)
        line_width = 2.0 if highlight else 1.0

        plt.plot(closed_rect[:, 0], closed_rect[:, 1], linewidth=line_width)
        center = rectangle_center(rect)
        plt.text(center[0], center[1], str(idx), fontsize=6)

    plt.axis("equal")
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def save_collision_report(collision_time: float,
                          collision_details: list[dict],
                          save_csv_path: str) -> pd.DataFrame:
    """
    保存首次碰撞报告
    """
    rows = []
    for item in collision_details:
        row = {
            "collision_time": collision_time,
            "bench_i": item["bench_i"],
            "bench_j": item["bench_j"],
            "center_i_x": item["center_i"][0],
            "center_i_y": item["center_i"][1],
            "center_j_x": item["center_j"][0],
            "center_j_y": item["center_j"][1],
            "min_dist": item["min_dist"],
            "center_dist": item["center_dist"],
            "intersect": item["intersect"],
        }

        rect_i = item["rect_i"]
        rect_j = item["rect_j"]

        for k in range(4):
            row[f"rect_i_p{k}_x"] = rect_i[k, 0]
            row[f"rect_i_p{k}_y"] = rect_i[k, 1]
            row[f"rect_j_p{k}_x"] = rect_j[k, 0]
            row[f"rect_j_p{k}_y"] = rect_j[k, 1]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    return df

#问题2的封装
def solve_problem2(x_mat: np.ndarray,
                   y_mat: np.ndarray,
                   dt: float,
                   width: float = BENCH_WIDTH,
                   center_dist_threshold: float = 1.5) -> dict:
    """
    问题二主函数：
    1. 重建全部矩形
    2. 粗搜索首次碰撞
    3. 二分细化碰撞时刻
    4. 输出报告与图像
    """
    print("问题2 - 任务1：重建全部板凳矩形")
    rectangles_over_time = build_all_rectangles_over_time(x_mat, y_mat, width)

    print("问题2 - 任务4：粗搜索首次碰撞时刻")
    coarse = find_first_collision_coarse(
        rectangles_over_time=rectangles_over_time,
        dt=dt,
        center_dist_threshold=center_dist_threshold
    )

    if coarse is None:
        print("在给定时间区间内未检测到碰撞。")
        return {
            "collision": False,
            "coarse": None,
            "refined": None,
            "details": [],
        }

    print(f"粗搜索首次碰撞时刻约为: {coarse['time']:.6f} s")

    print("问题2 - 任务4：局部二分细化首次碰撞时刻")
    refined = refine_first_collision_bisection(
        x_mat=x_mat,
        y_mat=y_mat,
        coarse_index=coarse["time_index"],
        dt=dt,
        width=width,
        center_dist_threshold=center_dist_threshold,
        tol=1e-5,
        max_iter=60
    )

    print(f"细化后的首次碰撞时刻: {refined['time']:.8f} s")

    print("问题2 - 任务5：提取首次碰撞详细信息")
    refined_rectangles = interpolate_rectangles(
        x_mat, y_mat, refined["left_idx"], refined["alpha"], width
    )

    refined_result = detect_collisions_at_one_time(
        refined_rectangles,
        center_dist_threshold=center_dist_threshold
    )

    details = extract_collision_details(refined_rectangles, refined_result)

    print("问题2 - 任务5：绘制首次碰撞图")
    plot_rectangles(
        refined_rectangles,
        collision_pairs=refined_result["pairs"],
        title=f"First Collision at t={refined['time']:.6f}s",
        save_path="problem2_first_collision.png"
    )

    print("问题2 - 任务5：保存首次碰撞报告")
    df_collision = save_collision_report(
        collision_time=refined["time"],
        collision_details=details,
        save_csv_path="problem2_collision_report.csv"
    )

    return {
        "collision": True,
        "coarse": coarse,
        "refined": refined,
        "details": details,
        "df_collision": df_collision,
    }

# ============================================================
# 问题三：最小螺距求解
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 公共参数（问题三）
# ============================================================

TURN_RADIUS = 4.5           # 调头空间半径 R（占位，按题目修改）
PITCH_SEARCH_LEFT = 0.20    # 螺距搜索左端（占位）
PITCH_SEARCH_RIGHT = 0.80   # 螺距搜索右端（占位）
PITCH_SEARCH_TOL = 1e-3     # 二分搜索精度
PITCH_SCAN_POINTS = 13      # 粗扫描点数


# ============================================================
# 任务1：参数化问题一和问题二，使其可接收任意螺距 pitch
# ============================================================

def pitch_to_a(pitch: float) -> float:
    """a = d / (2*pi)"""
    return pitch / (2 * math.pi)


def spiral_r_pitch(theta: float, pitch: float) -> float:
    """给定 pitch 的螺线极径"""
    return pitch_to_a(pitch) * theta


def spiral_xy_pitch(theta: float, pitch: float) -> tuple[float, float]:
    """给定 pitch 的螺线坐标"""
    r = spiral_r_pitch(theta, pitch)
    return r * math.cos(theta), r * math.sin(theta)


def handle_distance_pitch(theta1: float, theta2: float, pitch: float) -> float:
    """给定 pitch 的两点欧氏距离"""
    r1 = spiral_r_pitch(theta1, pitch)
    r2 = spiral_r_pitch(theta2, pitch)
    return math.sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * math.cos(theta2 - theta1))


def distance_equation_pitch(theta_next: float,
                            theta_curr: float,
                            target_len: float,
                            pitch: float) -> float:
    """递推方程 f(theta_next)=0"""
    return handle_distance_pitch(theta_curr, theta_next, pitch) - target_len


def theta_dot_pitch(theta: float, pitch: float) -> float:
    """
    龙头前把手极角微分方程:
    dtheta/dt = -1 / (a * sqrt(1 + theta^2))
    """
    a = pitch_to_a(pitch)
    return -1.0 / (a * math.sqrt(1.0 + theta * theta))


def rk4_step_theta_pitch(theta: float, dt: float, pitch: float) -> float:
    """给定 pitch 的 RK4 积分"""
    k1 = theta_dot_pitch(theta, pitch)
    k2 = theta_dot_pitch(theta + 0.5 * dt * k1, pitch)
    k3 = theta_dot_pitch(theta + 0.5 * dt * k2, pitch)
    k4 = theta_dot_pitch(theta + dt * k3, pitch)
    return theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_next_theta_pitch(theta_curr: float,
                           target_len: float,
                           pitch: float,
                           step_out: float = 0.02,
                           max_expand: int = 20000,
                           tol: float = 1e-10,
                           max_iter: int = 100) -> float:
    """
    给定 pitch，已知 theta_curr，求外侧相邻把手 theta_next
    """
    left = theta_curr
    f_left = distance_equation_pitch(left, theta_curr, target_len, pitch)

    right = left + step_out
    f_right = distance_equation_pitch(right, theta_curr, target_len, pitch)

    expand_count = 0
    while f_left * f_right > 0 and expand_count < max_expand:
        right += step_out
        f_right = distance_equation_pitch(right, theta_curr, target_len, pitch)
        expand_count += 1

    if f_left * f_right > 0:
        raise RuntimeError(
            f"未找到根区间: theta_curr={theta_curr:.6f}, target_len={target_len}, pitch={pitch}"
        )

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = distance_equation_pitch(mid, theta_curr, target_len, pitch)

        if abs(f_mid) < tol or (right - left) < tol:
            return mid

        if f_left * f_mid <= 0:
            right = mid
        else:
            left = mid
            f_left = f_mid

    return 0.5 * (left + right)


def solve_head_trajectory_pitch(t_start: float,
                                t_end: float,
                                dt: float,
                                theta_init: float,
                                pitch: float) -> pd.DataFrame:
    """
    给定 pitch 的龙头轨迹
    """
    times = np.arange(t_start, t_end + dt, dt)
    thetas = np.zeros(len(times), dtype=float)
    xs = np.zeros(len(times), dtype=float)
    ys = np.zeros(len(times), dtype=float)
    rs = np.zeros(len(times), dtype=float)

    theta = theta_init
    for i, t in enumerate(times):
        thetas[i] = theta
        x, y = spiral_xy_pitch(theta, pitch)
        xs[i] = x
        ys[i] = y
        rs[i] = spiral_r_pitch(theta, pitch)

        if i < len(times) - 1:
            theta = rk4_step_theta_pitch(theta, dt, pitch)

    return pd.DataFrame({
        "t": times,
        "theta_0": thetas,
        "r_0": rs,
        "x_0": xs,
        "y_0": ys,
    })


def solve_all_thetas_at_one_time_pitch(theta0: float,
                                       num_handles: int,
                                       pitch: float) -> np.ndarray:
    """
    给定 pitch 和某一时刻龙头 theta0，递推全部把手极角
    """
    thetas = np.zeros(num_handles, dtype=float)
    thetas[0] = theta0

    for n in range(num_handles - 1):
        target_len = get_handle_spacing(n)
        thetas[n + 1] = solve_next_theta_pitch(thetas[n], target_len, pitch)

    return thetas


def solve_all_thetas_over_time_pitch(df_head: pd.DataFrame,
                                     num_handles: int,
                                     pitch: float) -> np.ndarray:
    """
    给定 pitch，对所有时刻递推全部把手极角
    """
    num_times = len(df_head)
    theta_mat = np.zeros((num_times, num_handles), dtype=float)

    for i in range(num_times):
        theta0 = df_head.loc[i, "theta_0"]
        theta_mat[i, :] = solve_all_thetas_at_one_time_pitch(theta0, num_handles, pitch)

    return theta_mat


def theta_matrix_to_xy_pitch(theta_mat: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """
    给定 pitch，把 theta 矩阵转成 x,y
    """
    a = pitch_to_a(pitch)
    r_mat = a * theta_mat
    x_mat = r_mat * np.cos(theta_mat)
    y_mat = r_mat * np.sin(theta_mat)
    return x_mat, y_mat


def solve_problem1_with_pitch(
    pitch: float,
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
) -> dict:
    """
    给定螺距，完成问题1的核心仿真
    """
    df_head = solve_head_trajectory_pitch(
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta_init=theta0_init,
        pitch=pitch,
    )

    theta_mat = solve_all_thetas_over_time_pitch(
        df_head=df_head,
        num_handles=num_handles,
        pitch=pitch,
    )

    x_mat, y_mat = theta_matrix_to_xy_pitch(theta_mat, pitch)
    v_mat = compute_velocity_matrix(x_mat, y_mat, dt)

    return {
        "pitch": pitch,
        "df_head": df_head,
        "theta_mat": theta_mat,
        "x_mat": x_mat,
        "y_mat": y_mat,
        "v_mat": v_mat,
    }


# ============================================================
# 任务2：做“螺距—碰撞风险”粗扫描
# ============================================================

def first_enter_turn_radius(df_head: pd.DataFrame, turn_radius: float) -> int | None:
    """
    找到龙头第一次进入调头空间半径 R 的时间索引
    """
    mask = df_head["r_0"].values <= turn_radius
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    return int(indices[0])


def min_pairwise_clearance_at_one_time(rectangles: list[np.ndarray],
                                       center_dist_threshold: float) -> dict:
    """
    某一时刻下，所有非相邻候选板凳对的最小距离
    """
    candidates = candidate_pairs_by_center_distance(rectangles, center_dist_threshold)

    if len(candidates) == 0:
        return {
            "min_dist": float("inf"),
            "pair": None,
            "collision": False,
        }

    min_dist = float("inf")
    min_pair = None

    for i, j in candidates:
        rect_i = rectangles[i]
        rect_j = rectangles[j]

        if sat_intersect_rectangle(rect_i, rect_j):
            return {
                "min_dist": 0.0,
                "pair": (i, j),
                "collision": True,
            }

        d = rectangle_min_distance(rect_i, rect_j)
        if d < min_dist:
            min_dist = d
            min_pair = (i, j)

    return {
        "min_dist": min_dist,
        "pair": min_pair,
        "collision": False,
    }


def evaluate_pitch_feasibility(
    pitch: float,
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    width: float,
    turn_radius: float,
    center_dist_threshold: float,
) -> dict:
    """
    给定螺距，判断：
    1. 是否进入调头空间
    2. 在进入调头空间之前是否无碰撞
    3. 全过程最小安全距离
    """
    sim = solve_problem1_with_pitch(
        pitch=pitch,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
    )

    df_head = sim["df_head"]
    x_mat = sim["x_mat"]
    y_mat = sim["y_mat"]

    enter_idx = first_enter_turn_radius(df_head, turn_radius)
    if enter_idx is None:
        return {
            "pitch": pitch,
            "reachable": False,
            "feasible": False,
            "enter_index": None,
            "enter_time": None,
            "collision": None,
            "collision_time": None,
            "collision_pair": None,
            "min_clearance": None,
            "min_clearance_time": None,
            "min_clearance_pair": None,
            "sim": sim,
        }

    min_clearance = float("inf")
    min_clearance_time = None
    min_clearance_pair = None

    collision = False
    collision_time = None
    collision_pair = None

    for t_idx in range(enter_idx + 1):
        rects = build_all_rectangles_at_one_time(x_mat[t_idx], y_mat[t_idx], width)
        clearance_info = min_pairwise_clearance_at_one_time(rects, center_dist_threshold)

        if clearance_info["min_dist"] < min_clearance:
            min_clearance = clearance_info["min_dist"]
            min_clearance_time = t_idx * dt
            min_clearance_pair = clearance_info["pair"]

        if clearance_info["collision"]:
            collision = True
            collision_time = t_idx * dt
            collision_pair = clearance_info["pair"]
            break

    feasible = (not collision) and (enter_idx is not None)

    return {
        "pitch": pitch,
        "reachable": True,
        "feasible": feasible,
        "enter_index": enter_idx,
        "enter_time": enter_idx * dt,
        "collision": collision,
        "collision_time": collision_time,
        "collision_pair": collision_pair,
        "min_clearance": min_clearance,
        "min_clearance_time": min_clearance_time,
        "min_clearance_pair": min_clearance_pair,
        "sim": sim,
    }


def coarse_scan_pitch_values(
    pitch_values: np.ndarray,
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    width: float,
    turn_radius: float,
    center_dist_threshold: float,
) -> pd.DataFrame:
    """
    对多个螺距做粗扫描，输出螺距—可行性/最小安全距离表
    """
    rows = []

    for pitch in pitch_values:
        print(f"[粗扫描] 正在评估 pitch = {pitch:.6f}")
        result = evaluate_pitch_feasibility(
            pitch=pitch,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            theta0_init=theta0_init,
            num_handles=num_handles,
            width=width,
            turn_radius=turn_radius,
            center_dist_threshold=center_dist_threshold,
        )

        rows.append({
            "pitch": result["pitch"],
            "reachable": result["reachable"],
            "feasible": result["feasible"],
            "enter_time": result["enter_time"],
            "collision": result["collision"],
            "collision_time": result["collision_time"],
            "min_clearance": result["min_clearance"],
            "min_clearance_time": result["min_clearance_time"],
            "min_clearance_pair": result["min_clearance_pair"],
            "collision_pair": result["collision_pair"],
        })

    return pd.DataFrame(rows)


# ============================================================
# 任务3：计算 D_min(d)，并据此判定螺距是否可行
# ============================================================

def compute_D_min_for_pitch(
    pitch: float,
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    width: float,
    turn_radius: float,
    center_dist_threshold: float,
) -> dict:
    """
    计算 D_min(d) = 进入调头空间之前全过程中的最小安全距离
    """
    result = evaluate_pitch_feasibility(
        pitch=pitch,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        width=width,
        turn_radius=turn_radius,
        center_dist_threshold=center_dist_threshold,
    )

    return {
        "pitch": result["pitch"],
        "D_min": result["min_clearance"],
        "reachable": result["reachable"],
        "feasible": result["feasible"],
        "min_clearance_time": result["min_clearance_time"],
        "min_clearance_pair": result["min_clearance_pair"],
        "collision_time": result["collision_time"],
        "collision_pair": result["collision_pair"],
    }


# ============================================================
# 任务4：粗扫描 + 二分搜索，求最小可行螺距
# ============================================================

def find_initial_feasible_interval(
    left: float,
    right: float,
    num_points: int,
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    width: float,
    turn_radius: float,
    center_dist_threshold: float,
) -> tuple[float, float, pd.DataFrame]:
    """
    粗扫描寻找“左不可行、右可行”的初始区间
    """
    pitch_values = np.linspace(left, right, num_points)
    df_scan = coarse_scan_pitch_values(
        pitch_values=pitch_values,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        width=width,
        turn_radius=turn_radius,
        center_dist_threshold=center_dist_threshold,
    )

    feasible_flags = df_scan["feasible"].values.astype(bool)
    pitch_arr = df_scan["pitch"].values

    left_bad = None
    right_good = None

    for i in range(1, len(df_scan)):
        if (not feasible_flags[i - 1]) and feasible_flags[i]:
            left_bad = float(pitch_arr[i - 1])
            right_good = float(pitch_arr[i])
            break

    if left_bad is None or right_good is None:
        raise RuntimeError("粗扫描未找到有效初始区间，请扩大搜索范围或调整参数。")

    return left_bad, right_good, df_scan


def binary_search_min_feasible_pitch(
    left_bad: float,
    right_good: float,
    tol: float,
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    width: float,
    turn_radius: float,
    center_dist_threshold: float,
) -> dict:
    """
    二分搜索最小可行螺距
    假定：
      left_bad  不可行
      right_good 可行
    """
    history = []

    while right_good - left_bad > tol:
        mid = 0.5 * (left_bad + right_good)
        print(f"[二分] 检查 pitch = {mid:.6f}")

        result = evaluate_pitch_feasibility(
            pitch=mid,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            theta0_init=theta0_init,
            num_handles=num_handles,
            width=width,
            turn_radius=turn_radius,
            center_dist_threshold=center_dist_threshold,
        )

        history.append({
            "pitch": result["pitch"],
            "feasible": result["feasible"],
            "reachable": result["reachable"],
            "min_clearance": result["min_clearance"],
            "collision_time": result["collision_time"],
        })

        if result["feasible"]:
            right_good = mid
        else:
            left_bad = mid

    final_result = evaluate_pitch_feasibility(
        pitch=right_good,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        width=width,
        turn_radius=turn_radius,
        center_dist_threshold=center_dist_threshold,
    )

    return {
        "pitch_star": right_good,
        "left_bad": left_bad,
        "right_good": right_good,
        "history": pd.DataFrame(history),
        "final_result": final_result,
    }


# ============================================================
# 任务5：在最优螺距附近做扰动分析，并输出表格和图
# ============================================================

def analyze_pitch_sensitivity(
    pitch_star: float,
    delta_list: list[float],
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    width: float,
    turn_radius: float,
    center_dist_threshold: float,
) -> pd.DataFrame:
    """
    在 pitch_star 附近做扰动分析
    """
    rows = []

    for delta in delta_list:
        pitch = pitch_star + delta
        if pitch <= 0:
            continue

        print(f"[扰动分析] 正在评估 pitch = {pitch:.6f}")
        result = evaluate_pitch_feasibility(
            pitch=pitch,
            t_start=t_start,
            t_end=t_end,
            dt=dt,
            theta0_init=theta0_init,
            num_handles=num_handles,
            width=width,
            turn_radius=turn_radius,
            center_dist_threshold=center_dist_threshold,
        )

        rows.append({
            "pitch": result["pitch"],
            "delta": delta,
            "reachable": result["reachable"],
            "feasible": result["feasible"],
            "enter_time": result["enter_time"],
            "collision": result["collision"],
            "collision_time": result["collision_time"],
            "collision_pair": result["collision_pair"],
            "min_clearance": result["min_clearance"],
            "min_clearance_time": result["min_clearance_time"],
            "min_clearance_pair": result["min_clearance_pair"],
        })

    return pd.DataFrame(rows)


def plot_pitch_clearance_curve(df_scan: pd.DataFrame,
                               save_path: str = "problem3_pitch_clearance_curve.png") -> None:
    """
    绘制 螺距 - 最小安全距离 曲线
    """
    plt.figure(figsize=(8, 5))
    plt.plot(df_scan["pitch"], df_scan["min_clearance"], marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Pitch d (m)")
    plt.ylabel("Minimum clearance D_min(d)")
    plt.title("Pitch vs Minimum Clearance")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pitch_feasibility_bar(df_scan: pd.DataFrame,
                               save_path: str = "problem3_pitch_feasibility.png") -> None:
    """
    绘制螺距可行性示意图
    """
    plt.figure(figsize=(8, 4))
    y = df_scan["feasible"].astype(int)
    plt.plot(df_scan["pitch"], y, marker="o")
    plt.yticks([0, 1], ["Infeasible", "Feasible"])
    plt.xlabel("Pitch d (m)")
    plt.ylabel("Feasibility")
    plt.title("Pitch Feasibility")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 问题三总封装：solve_problem3(...)
# ============================================================

def solve_problem3(
    t_start: float,
    t_end: float,
    dt: float,
    theta0_init: float,
    num_handles: int,
    width: float,
    turn_radius: float = TURN_RADIUS,
    center_dist_threshold: float = 1.5,
    pitch_left: float = PITCH_SEARCH_LEFT,
    pitch_right: float = PITCH_SEARCH_RIGHT,
    pitch_tol: float = PITCH_SEARCH_TOL,
    pitch_scan_points: int = PITCH_SCAN_POINTS,
    export_csv: bool = True,
) -> dict:
    """
    问题三总封装：
    1. 粗扫描
    2. 找初始区间
    3. 二分搜索最小可行螺距
    4. 做最优附近扰动分析
    5. 导出表格和图
    """
    print("问题3 - 任务2：粗扫描螺距")
    left_bad, right_good, df_scan = find_initial_feasible_interval(
        left=pitch_left,
        right=pitch_right,
        num_points=pitch_scan_points,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        width=width,
        turn_radius=turn_radius,
        center_dist_threshold=center_dist_threshold,
    )

    print(f"问题3 - 初始区间找到: left_bad={left_bad:.6f}, right_good={right_good:.6f}")

    print("问题3 - 任务4：二分搜索最小可行螺距")
    search_result = binary_search_min_feasible_pitch(
        left_bad=left_bad,
        right_good=right_good,
        tol=pitch_tol,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        width=width,
        turn_radius=turn_radius,
        center_dist_threshold=center_dist_threshold,
    )

    pitch_star = search_result["pitch_star"]
    final_result = search_result["final_result"]

    print(f"问题3 - 最小可行螺距近似为: {pitch_star:.6f}")

    print("问题3 - 任务5：最优附近扰动分析")
    delta_list = [-2 * pitch_tol, -pitch_tol, 0.0, pitch_tol, 2 * pitch_tol]
    df_sensitivity = analyze_pitch_sensitivity(
        pitch_star=pitch_star,
        delta_list=delta_list,
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        width=width,
        turn_radius=turn_radius,
        center_dist_threshold=center_dist_threshold,
    )

    print("问题3 - 绘图")
    plot_pitch_clearance_curve(df_scan)
    plot_pitch_feasibility_bar(df_scan)

    if export_csv:
        df_scan.to_csv("problem3_pitch_scan.csv", index=False, encoding="utf-8-sig")
        search_result["history"].to_csv("problem3_binary_search_history.csv", index=False, encoding="utf-8-sig")
        df_sensitivity.to_csv("problem3_pitch_sensitivity.csv", index=False, encoding="utf-8-sig")

    return {
        "pitch_star": pitch_star,
        "scan_table": df_scan,
        "search_history": search_result["history"],
        "final_result": final_result,
        "sensitivity_table": df_sensitivity,
    }


# ============================================================
# 问题四：调头路径设计与运动仿真
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 公共参数（问题四）
# ============================================================

TURN_RADIUS_P4 = 4.5          # 调头空间半径，占位
PATH_DS = 0.01                # 路径离散步长（弧长步长近似）
TURN_ARC_R1 = 1.8             # 第一段圆弧半径，占位
TURN_ARC_R2 = 1.2             # 第二段圆弧半径，占位
P4_CENTER_DIST_THRESHOLD = 1.5
P4_EXPORT_PREFIX = "problem4"


# ============================================================
# 任务1：构造调头路径几何对象
# ============================================================

def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("零向量无法单位化")
    return vec / norm


def rotate90(vec: np.ndarray) -> np.ndarray:
    """逆时针旋转90度"""
    return np.array([-vec[1], vec[0]], dtype=float)


def rotate_minus_90(vec: np.ndarray) -> np.ndarray:
    """顺时针旋转90度"""
    return np.array([vec[1], -vec[0]], dtype=float)


def polyline_arclength(points: np.ndarray) -> np.ndarray:
    """
    输入 shape=(N,2)
    返回累计弧长数组 shape=(N,)
    """
    n = len(points)
    s = np.zeros(n, dtype=float)
    for i in range(1, n):
        s[i] = s[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    return s


def resample_polyline_by_step(points: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """
    将折线按近似等弧长重采样
    返回：
      new_points: shape=(M,2)
      new_s: shape=(M,)
    """
    old_s = polyline_arclength(points)
    total_len = old_s[-1]

    if total_len < 1e-12:
        return points.copy(), np.array([0.0])

    new_s = np.arange(0.0, total_len + ds, ds)
    if new_s[-1] > total_len:
        new_s[-1] = total_len

    x = np.interp(new_s, old_s, points[:, 0])
    y = np.interp(new_s, old_s, points[:, 1])
    new_points = np.column_stack([x, y])
    return new_points, new_s


def arc_points(center: np.ndarray,
               radius: float,
               angle_start: float,
               angle_end: float,
               num: int = 200) -> np.ndarray:
    """
    生成圆弧离散点
    """
    angles = np.linspace(angle_start, angle_end, num)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])


# ============================================================
# 任务2：确定盘入段终点 P1、盘出段起点 P2 及切向方向
# ============================================================

def spiral_point(theta: float, pitch: float) -> np.ndarray:
    a = pitch / (2 * math.pi)
    r = a * theta
    return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)


def spiral_tangent(theta: float, pitch: float) -> np.ndarray:
    """
    阿基米德螺线 r=a*theta 的参数导数方向
    x = a theta cos(theta), y = a theta sin(theta)
    """
    a = pitch / (2 * math.pi)
    dx = a * (math.cos(theta) - theta * math.sin(theta))
    dy = a * (math.sin(theta) + theta * math.cos(theta))
    return unit_vector(np.array([dx, dy], dtype=float))


def solve_theta_for_radius(radius: float, pitch: float) -> float:
    """
    由 r = pitch/(2pi) * theta 反解 theta
    """
    a = pitch / (2 * math.pi)
    return radius / a


def build_turn_boundary_points(pitch: float,
                               turn_radius: float) -> dict:
    """
    取盘入段与圆形调头空间边界交点 P1
    以及中心对称的盘出连接点 P2
    """
    theta1 = solve_theta_for_radius(turn_radius, pitch)
    P1 = spiral_point(theta1, pitch)
    T1 = spiral_tangent(theta1, pitch)

    # 盘出点用中心对称点
    P2 = -P1
    T2 = -T1

    return {
        "theta1": theta1,
        "P1": P1,
        "T1": T1,
        "P2": P2,
        "T2": T2,
    }


# ============================================================
# 任务3：构造两段圆弧调头路径，并检查连续性
# ============================================================

def build_two_arc_turn_path(P1: np.ndarray,
                            T1: np.ndarray,
                            P2: np.ndarray,
                            T2: np.ndarray,
                            R1: float,
                            R2: float,
                            arc_samples: int = 200) -> dict:
    """
    构造双圆弧调头路径（简化版）
    """
    N1 = rotate90(T1)
    N2 = rotate90(T2)

    O1 = P1 + R1 * N1
    O2 = P2 + R2 * N2

    OO = O2 - O1
    dist_OO = np.linalg.norm(OO)
    if dist_OO < 1e-10:
        raise ValueError("两圆心重合，无法构造双圆弧")

    u = OO / dist_OO

    Pm1 = O1 + R1 * u
    Pm2 = O2 - R2 * u
    Pm = 0.5 * (Pm1 + Pm2)

    a1_start = math.atan2(P1[1] - O1[1], P1[0] - O1[0])
    a1_mid = math.atan2(Pm[1] - O1[1], Pm[0] - O1[0])

    a2_mid = math.atan2(Pm[1] - O2[1], Pm[0] - O2[0])
    a2_end = math.atan2(P2[1] - O2[1], P2[0] - O2[0])

    def unwrap_short(a, b):
        while b - a > math.pi:
            b -= 2 * math.pi
        while b - a < -math.pi:
            b += 2 * math.pi
        return a, b

    a1_start, a1_mid = unwrap_short(a1_start, a1_mid)
    a2_mid, a2_end = unwrap_short(a2_mid, a2_end)

    arc1 = arc_points(O1, R1, a1_start, a1_mid, num=arc_samples)
    arc2 = arc_points(O2, R2, a2_mid, a2_end, num=arc_samples)

    bridge = np.vstack([arc1[-1], arc2[0]])
    turn_points = np.vstack([arc1, bridge, arc2])

    return {
        "O1": O1,
        "O2": O2,
        "R1": R1,
        "R2": R2,
        "Pm": Pm,
        "arc1": arc1,
        "arc2": arc2,
        "turn_points": turn_points,
        "continuity_gap": np.linalg.norm(arc1[-1] - arc2[0]),
    }


# ============================================================
# 任务4：复合路径离散化与弧长参数化
# ============================================================

def build_inbound_spiral_segment(pitch: float,
                                 theta_start: float,
                                 theta_end: float,
                                 num: int = 1200) -> np.ndarray:
    """
    从 theta_start 到 theta_end 的盘入螺线段
    默认 theta_start > theta_end
    """
    thetas = np.linspace(theta_start, theta_end, num)
    pts = np.array([spiral_point(th, pitch) for th in thetas], dtype=float)
    return pts


def build_outbound_spiral_segment(pitch: float,
                                  theta_start: float,
                                  theta_end: float,
                                  num: int = 1200) -> np.ndarray:
    """
    盘出段：这里采用盘入段中心对称后的路径
    """
    thetas = np.linspace(theta_start, theta_end, num)
    pts = np.array([-spiral_point(th, pitch) for th in thetas], dtype=float)
    return pts


def build_composite_turning_path(pitch: float,
                                 theta_start: float,
                                 turn_radius: float,
                                 R1: float,
                                 R2: float,
                                 ds: float = PATH_DS) -> dict:
    """
    构造完整复合路径：
    盘入段 + 调头双圆弧 + 盘出段
    """
    boundary = build_turn_boundary_points(pitch, turn_radius)

    P1 = boundary["P1"]
    T1 = boundary["T1"]
    P2 = boundary["P2"]
    T2 = boundary["T2"]
    theta1 = boundary["theta1"]

    inbound = build_inbound_spiral_segment(
        pitch=pitch,
        theta_start=theta_start,
        theta_end=theta1,
        num=1500,
    )

    turn_obj = build_two_arc_turn_path(
        P1=P1,
        T1=T1,
        P2=P2,
        T2=T2,
        R1=R1,
        R2=R2,
        arc_samples=300,
    )

    outbound = build_outbound_spiral_segment(
        pitch=pitch,
        theta_start=theta1,
        theta_end=theta_start,
        num=1500,
    )

    raw_path = np.vstack([inbound, turn_obj["turn_points"], outbound])
    path_points, path_s = resample_polyline_by_step(raw_path, ds)

    return {
        "boundary": boundary,
        "turn_obj": turn_obj,
        "raw_path": raw_path,
        "path_points": path_points,
        "path_s": path_s,
        "total_length": path_s[-1],
    }


def path_position(path_points: np.ndarray,
                  path_s: np.ndarray,
                  s_query: float) -> np.ndarray:
    """
    由弧长参数查询路径坐标
    """
    s_query = max(0.0, min(float(s_query), float(path_s[-1])))
    x = np.interp(s_query, path_s, path_points[:, 0])
    y = np.interp(s_query, path_s, path_points[:, 1])
    return np.array([x, y], dtype=float)


def path_positions_batch(path_points: np.ndarray,
                         path_s: np.ndarray,
                         s_queries: np.ndarray) -> np.ndarray:
    s_queries = np.clip(s_queries, 0.0, path_s[-1])
    x = np.interp(s_queries, path_s, path_points[:, 0])
    y = np.interp(s_queries, path_s, path_points[:, 1])
    return np.column_stack([x, y])


def build_head_motion_on_path(path_points: np.ndarray,
                              path_s: np.ndarray,
                              v_head: float,
                              dt: float) -> dict:
    """
    龙头沿复合路径恒速运动
    """
    total_length = path_s[-1]
    total_time = total_length / v_head
    times = np.arange(0.0, total_time + dt, dt)
    s_head = np.clip(v_head * times, 0.0, total_length)
    head_xy = path_positions_batch(path_points, path_s, s_head)

    return {
        "times": times,
        "s_head": s_head,
        "head_xy": head_xy,
        "total_time": total_time,
    }


def build_head_motion_on_turn_window(path_obj: dict,
                                     v_head: float,
                                     dt: float,
                                     num_handles: int,
                                     margin: float = 2.0) -> dict:
    """
    只在调头段附近生成龙头轨迹，避免把问题4错误地扩展到整条路径后半段。
    对当前参数，整龙链长通常大于“盘入+调头”长度，因此这里采用局部调头窗口近似，
    不再要求整龙先完全铺到路径上再开始取样。
    """
    total_length = float(path_obj["total_length"])
    if total_length <= 0.0:
        raise ValueError("复合路径长度必须为正。")

    s_turn_start = float(path_obj["s_stage_in_end"])
    s_turn_end = float(path_obj["s_stage_turn_end"])
    s0 = max(0.0, s_turn_start - margin)
    s1 = min(total_length, s_turn_end + margin)

    if s1 <= s0:
        raise ValueError(
            f"调头窗口非法: start={s0:.6f}, end={s1:.6f}。请检查路径长度或 margin 设置。"
        )

    times = np.arange(0.0, (s1 - s0) / v_head + dt, dt)
    s_head = np.clip(s0 + v_head * times, s0, s1)
    if s_head[-1] < s1:
        s_head = np.append(s_head, s1)
        times = np.append(times, (s1 - s0) / v_head)

    head_xy = path_positions_batch(path_obj["path_points"], path_obj["path_s"], s_head)
    return {
        "times": times,
        "s_head": s_head,
        "head_xy": head_xy,
        "total_time": (s1 - s0) / v_head,
        "start_s_head": s0,
        "end_s_head": s1,
    }


# ============================================================
# 任务5：全体把手在复合路径上的位置递推与速度计算
# ============================================================

def solve_next_handle_s_along_path(s_prev: float,
                                   prev_xy: np.ndarray,
                                   target_len: float,
                                   path_points: np.ndarray,
                                   path_s: np.ndarray,
                                   ds_search: float = 0.002,
                                   max_back: float = 20.0) -> float:
    """
    已知前一个把手位于 s_prev，求后一个把手弧长 s_next < s_prev
    """
    left = max(0.0, s_prev - max_back)
    right = s_prev

    def f(s):
        p = path_position(path_points, path_s, s)
        return np.linalg.norm(prev_xy - p) - target_len

    s_hi = right
    found = False
    s_lo = s_hi

    s = s_hi
    while s > left:
        s = max(left, s - ds_search)
        f_cur = f(s)
        if f_cur >= 0.0:
            s_lo = s
            found = True
            break

    if not found:
        return left

    a, b = s_lo, s_hi

    for _ in range(60):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < 1e-8 or (b - a) < 1e-8:
            return mid
        if fm >= 0:
            a = mid
        else:
            b = mid

    return 0.5 * (a + b)


def solve_all_handles_on_composite_path(head_motion: dict,
                                        path_points: np.ndarray,
                                        path_s: np.ndarray,
                                        num_handles: int) -> dict:
    """
    对每个时刻，已知龙头 s_head，递推全体把手在路径上的弧长位置与坐标
    """
    times = head_motion["times"]
    s_head = head_motion["s_head"]
    num_times = len(times)

    s_mat = np.zeros((num_times, num_handles), dtype=float)
    x_mat = np.zeros((num_times, num_handles), dtype=float)
    y_mat = np.zeros((num_times, num_handles), dtype=float)

    for t_idx in range(num_times):
        s_mat[t_idx, 0] = s_head[t_idx]
        head_xy = path_position(path_points, path_s, s_head[t_idx])
        x_mat[t_idx, 0] = head_xy[0]
        y_mat[t_idx, 0] = head_xy[1]

        for h in range(num_handles - 1):
            prev_s = s_mat[t_idx, h]
            prev_xy = np.array([x_mat[t_idx, h], y_mat[t_idx, h]], dtype=float)
            target_len = get_handle_spacing(h)

            next_s = solve_next_handle_s_along_path(
                s_prev=prev_s,
                prev_xy=prev_xy,
                target_len=target_len,
                path_points=path_points,
                path_s=path_s,
            )

            next_xy = path_position(path_points, path_s, next_s)

            s_mat[t_idx, h + 1] = next_s
            x_mat[t_idx, h + 1] = next_xy[0]
            y_mat[t_idx, h + 1] = next_xy[1]

        if t_idx % 500 == 0:
            print(f"问题4递推进度: {t_idx}/{num_times}")

    v_mat = compute_velocity_matrix(x_mat, y_mat, times[1] - times[0])

    return {
        "times": times,
        "s_mat": s_mat,
        "x_mat": x_mat,
        "y_mat": y_mat,
        "v_mat": v_mat,
    }


# ============================================================
# 任务6：调头阶段碰撞检测
# ============================================================

def detect_collisions_over_time_p4(x_mat: np.ndarray,
                                   y_mat: np.ndarray,
                                   width: float,
                                   center_dist_threshold: float) -> dict:
    """
    逐时刻检查调头过程碰撞
    """
    num_times = x_mat.shape[0]

    first_collision_index = None
    first_collision_pairs = None
    min_clearance = float("inf")
    min_clearance_index = None
    min_clearance_pair = None

    for t_idx in range(num_times):
        rects = build_all_rectangles_at_one_time(x_mat[t_idx], y_mat[t_idx], width)
        candidates = candidate_pairs_by_center_distance(rects, center_dist_threshold)

        current_min = float("inf")
        current_pair = None
        collided_pairs = []

        for i, j in candidates:
            rect_i = rects[i]
            rect_j = rects[j]

            if sat_intersect_rectangle(rect_i, rect_j):
                collided_pairs.append((i, j))
                current_min = 0.0
                current_pair = (i, j)
            else:
                d = rectangle_min_distance(rect_i, rect_j)
                if d < current_min:
                    current_min = d
                    current_pair = (i, j)

        if current_min < min_clearance:
            min_clearance = current_min
            min_clearance_index = t_idx
            min_clearance_pair = current_pair

        if first_collision_index is None and len(collided_pairs) > 0:
            first_collision_index = t_idx
            first_collision_pairs = collided_pairs

    return {
        "collision": first_collision_index is not None,
        "first_collision_index": first_collision_index,
        "first_collision_pairs": first_collision_pairs,
        "min_clearance": min_clearance,
        "min_clearance_index": min_clearance_index,
        "min_clearance_pair": min_clearance_pair,
    }


# ============================================================
# 任务7：评价指标与绘图输出
# ============================================================

def compute_problem4_metrics(path_obj: dict,
                             handle_motion: dict,
                             collision_result: dict) -> dict:
    """
    计算总路径长度、最大速度、最小安全距离等
    """
    times = handle_motion["times"]
    v_mat = handle_motion["v_mat"]

    total_length = path_obj["total_length"]
    max_speed = float(np.max(v_mat))
    max_speed_index = np.unravel_index(np.argmax(v_mat), v_mat.shape)

    return {
        "total_length": total_length,
        "total_time": times[-1],
        "max_speed": max_speed,
        "max_speed_time_index": int(max_speed_index[0]),
        "max_speed_handle_index": int(max_speed_index[1]),
        "min_clearance": collision_result["min_clearance"],
        "min_clearance_time_index": collision_result["min_clearance_index"],
        "min_clearance_pair": collision_result["min_clearance_pair"],
        "collision": collision_result["collision"],
        "first_collision_index": collision_result["first_collision_index"],
        "first_collision_pairs": collision_result["first_collision_pairs"],
    }


def plot_composite_path(path_obj: dict,
                        save_path: str = f"{P4_EXPORT_PREFIX}_path.png") -> None:
    pts = path_obj["path_points"]
    plt.figure(figsize=(7, 7))
    plt.plot(pts[:, 0], pts[:, 1])
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Problem 4 Composite Turning Path")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_handle_configuration(x_mat: np.ndarray,
                              y_mat: np.ndarray,
                              time_index: int,
                              width: float,
                              title: str,
                              save_path: str) -> None:
    rects = build_all_rectangles_at_one_time(x_mat[time_index], y_mat[time_index], width)
    plt.figure(figsize=(8, 8))

    for idx, rect in enumerate(rects):
        closed_rect = np.vstack([rect, rect[0]])
        plt.plot(closed_rect[:, 0], closed_rect[:, 1], linewidth=1.0)
        center = rect.mean(axis=0)
        plt.text(center[0], center[1], str(idx), fontsize=6)

    plt.scatter(x_mat[time_index], y_mat[time_index], s=8)
    plt.axis("equal")
    plt.grid(True)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_speed_curve(handle_motion: dict,
                     save_path: str = f"{P4_EXPORT_PREFIX}_speed_curve.png") -> None:
    times = handle_motion["times"]
    v_mat = handle_motion["v_mat"]
    vmax_t = np.max(v_mat, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(times, vmax_t)
    plt.grid(True)
    plt.xlabel("t (s)")
    plt.ylabel("max handle speed (m/s)")
    plt.title("Problem 4 Maximum Handle Speed Over Time")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def export_problem4_tables(path_obj: dict,
                           handle_motion: dict,
                           metrics: dict,
                           prefix: str = P4_EXPORT_PREFIX) -> None:
    path_df = pd.DataFrame({
        "s": path_obj["path_s"],
        "x": path_obj["path_points"][:, 0],
        "y": path_obj["path_points"][:, 1],
    })
    path_df.to_csv(f"{prefix}_path_points.csv", index=False, encoding="utf-8-sig")

    times = handle_motion["times"]
    x_mat = handle_motion["x_mat"]
    y_mat = handle_motion["y_mat"]
    v_mat = handle_motion["v_mat"]

    rows = []
    for t_idx, t in enumerate(times):
        row = {"t": t}
        for j in range(x_mat.shape[1]):
            row[f"handle_{j}_x"] = x_mat[t_idx, j]
            row[f"handle_{j}_y"] = y_mat[t_idx, j]
            row[f"handle_{j}_v"] = v_mat[t_idx, j]
        rows.append(row)

    motion_df = pd.DataFrame(rows)
    motion_df.to_csv(f"{prefix}_handle_motion.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{prefix}_metrics.csv", index=False, encoding="utf-8-sig")


# ============================================================
# solve_problem4(...) 总封装
# ============================================================

def solve_problem4(
    pitch: float,
    theta_start: float,
    num_handles: int,
    bench_width: float,
    v_head: float = 1.0,
    dt: float = 0.01,
    turn_radius: float = TURN_RADIUS_P4,
    R1: float = TURN_ARC_R1,
    R2: float = TURN_ARC_R2,
    ds: float = PATH_DS,
    center_dist_threshold: float = P4_CENTER_DIST_THRESHOLD,
    export_csv: bool = True,
    export_fig: bool = True,
    export_prefix: str = P4_EXPORT_PREFIX,
) -> dict:
    """
    问题四总封装：
    1. 构造盘入-调头-盘出复合路径
    2. 龙头沿路径恒速运动
    3. 全队沿路径递推
    4. 计算速度
    5. 碰撞检测
    6. 输出评价指标与图表
    """
    print("问题4 - 任务1~4：构造复合路径并弧长参数化")
    path_obj = build_composite_turning_path(
        pitch=pitch,
        theta_start=theta_start,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
    )

    print("问题4 - 任务4：生成调头窗口内的龙头轨迹")
    head_motion = build_head_motion_on_turn_window(
        path_obj=path_obj,
        v_head=v_head,
        dt=dt,
        num_handles=num_handles,
        margin=2.0,
    )

    print("问题4 - 任务5：全体把手位置递推与速度计算")
    handle_motion = solve_all_handles_on_composite_path(
        head_motion=head_motion,
        path_points=path_obj["path_points"],
        path_s=path_obj["path_s"],
        num_handles=num_handles,
    )

    print("问题4 - 任务6：调头阶段碰撞检测")
    collision_result = detect_collisions_over_time_p4(
        x_mat=handle_motion["x_mat"],
        y_mat=handle_motion["y_mat"],
        width=bench_width,
        center_dist_threshold=center_dist_threshold,
    )

    print("问题4 - 任务7：计算评价指标")
    metrics = compute_problem4_metrics(
        path_obj=path_obj,
        handle_motion=handle_motion,
        collision_result=collision_result,
    )

    if export_fig:
        print("问题4 - 导出图像")
        plot_composite_path(path_obj, save_path=f"{export_prefix}_path.png")
        plot_speed_curve(handle_motion, save_path=f"{export_prefix}_speed_curve.png")

        ntime = len(handle_motion["times"])
        idx_list = [0, ntime // 2, ntime - 1]
        name_list = ["start", "middle", "end"]

        for idx, name in zip(idx_list, name_list):
            plot_handle_configuration(
                x_mat=handle_motion["x_mat"],
                y_mat=handle_motion["y_mat"],
                time_index=idx,
                width=bench_width,
                title=f"Problem 4 Configuration - {name}",
                save_path=f"{export_prefix}_config_{name}.png",
            )

        if collision_result["collision"]:
            idx = collision_result["first_collision_index"]
            plot_handle_configuration(
                x_mat=handle_motion["x_mat"],
                y_mat=handle_motion["y_mat"],
                time_index=idx,
                width=bench_width,
                title="Problem 4 First Collision Configuration",
                save_path=f"{export_prefix}_first_collision.png",
            )

    if export_csv:
        print("问题4 - 导出表格")
        export_problem4_tables(
            path_obj=path_obj,
            handle_motion=handle_motion,
            metrics=metrics,
            prefix=export_prefix,
        )

    return {
        "path_obj": path_obj,
        "head_motion": head_motion,
        "handle_motion": handle_motion,
        "collision_result": collision_result,
        "metrics": metrics,
    }


# ============================================================
# 问题五：最大允许行进速度分析
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 公共参数（问题五）
# ============================================================

P5_SPEED_LIMIT = 2.0              # 全体把手速度上限 v_lim，占位
P5_VHEAD_LEFT = 0.10              # 龙头速度搜索左端，占位
P5_VHEAD_RIGHT = 5.00             # 龙头速度搜索右端，占位
P5_VHEAD_TOL = 1e-3               # 龙头速度搜索精度
P5_SCAN_POINTS = 12               # 粗扫描点数
P5_EXPORT_PREFIX = "problem5"


# ============================================================
# 任务1：给定龙头速度 v0，输出全体把手速度矩阵与全局最大速度
# ============================================================

def evaluate_head_speed_once(
    v_head: float,
    pitch: float,
    theta_start: float,
    num_handles: int,
    bench_width: float,
    dt: float = 0.01,
    turn_radius: float = TURN_RADIUS_P4,
    R1: float = TURN_ARC_R1,
    R2: float = TURN_ARC_R2,
    ds: float = PATH_DS,
    center_dist_threshold: float = P4_CENTER_DIST_THRESHOLD,
    export_csv: bool = False,
    export_fig: bool = False,
    export_prefix: str = P5_EXPORT_PREFIX,
) -> dict:
    """
    给定龙头速度，调用问题四求解，返回速度分析结果
    """
    result_p4 = solve_problem4(
        pitch=pitch,
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        v_head=v_head,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=center_dist_threshold,
        export_csv=export_csv,
        export_fig=export_fig,
        export_prefix=export_prefix,
    )

    handle_motion = result_p4["handle_motion"]
    times = handle_motion["times"]
    x_mat = handle_motion["x_mat"]
    y_mat = handle_motion["y_mat"]
    v_mat = handle_motion["v_mat"]

    vmax = float(np.max(v_mat))
    vmax_idx = np.unravel_index(np.argmax(v_mat), v_mat.shape)
    t_idx = int(vmax_idx[0])
    h_idx = int(vmax_idx[1])

    return {
        "v_head": v_head,
        "V_max": vmax,
        "critical_time_index": t_idx,
        "critical_time": float(times[t_idx]),
        "critical_handle_index": h_idx,
        "critical_x": float(x_mat[t_idx, h_idx]),
        "critical_y": float(y_mat[t_idx, h_idx]),
        "result_p4": result_p4,
    }


# ============================================================
# 任务2：绘制“龙头速度—全队最大速度”关系曲线
# ============================================================

def scan_head_speed_values(
    v_values: np.ndarray,
    pitch: float,
    theta_start: float,
    num_handles: int,
    bench_width: float,
    dt: float = 0.01,
    turn_radius: float = TURN_RADIUS_P4,
    R1: float = TURN_ARC_R1,
    R2: float = TURN_ARC_R2,
    ds: float = PATH_DS,
    center_dist_threshold: float = P4_CENTER_DIST_THRESHOLD,
) -> pd.DataFrame:
    """
    对多个龙头速度做扫描，输出 V_max(v_head) 表
    """
    rows = []

    for v_head in v_values:
        print(f"[问题5 粗扫描] 正在评估 v_head = {v_head:.6f}")
        result = evaluate_head_speed_once(
            v_head=v_head,
            pitch=pitch,
            theta_start=theta_start,
            num_handles=num_handles,
            bench_width=bench_width,
            dt=dt,
            turn_radius=turn_radius,
            R1=R1,
            R2=R2,
            ds=ds,
            center_dist_threshold=center_dist_threshold,
            export_csv=False,
            export_fig=False,
        )

        rows.append({
            "v_head": result["v_head"],
            "V_max": result["V_max"],
            "critical_time": result["critical_time"],
            "critical_handle_index": result["critical_handle_index"],
            "critical_x": result["critical_x"],
            "critical_y": result["critical_y"],
        })

    return pd.DataFrame(rows)


def plot_headspeed_vs_vmax(
    df_scan: pd.DataFrame,
    v_lim: float,
    save_path: str = f"{P5_EXPORT_PREFIX}_headspeed_vs_vmax.png",
) -> None:
    """
    绘制 龙头速度 - 全队最大速度 曲线
    """
    plt.figure(figsize=(8, 5))
    plt.plot(df_scan["v_head"], df_scan["V_max"], marker="o")
    plt.axhline(v_lim, linestyle="--")
    plt.xlabel("Head speed $v_0$ (m/s)")
    plt.ylabel("Global max handle speed $V_{max}(v_0)$ (m/s)")
    plt.title("Head Speed vs Global Max Handle Speed")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 任务3：输出全局最大速度、达到最大值的把手编号和时刻
# ============================================================

def build_speed_summary_row(result: dict, v_lim: float) -> dict:
    """
    将一次速度评估结果整理成摘要行
    """
    feasible = result["V_max"] <= v_lim
    return {
        "v_head": result["v_head"],
        "V_max": result["V_max"],
        "feasible": feasible,
        "critical_time_index": result["critical_time_index"],
        "critical_time": result["critical_time"],
        "critical_handle_index": result["critical_handle_index"],
        "critical_x": result["critical_x"],
        "critical_y": result["critical_y"],
    }


# ============================================================
# 任务4：粗扫描 + 二分搜索，求最大允许龙头速度
# ============================================================

def find_initial_speed_interval(
    v_left: float,
    v_right: float,
    num_points: int,
    pitch: float,
    theta_start: float,
    num_handles: int,
    bench_width: float,
    v_lim: float,
    dt: float = 0.01,
    turn_radius: float = TURN_RADIUS_P4,
    R1: float = TURN_ARC_R1,
    R2: float = TURN_ARC_R2,
    ds: float = PATH_DS,
    center_dist_threshold: float = P4_CENTER_DIST_THRESHOLD,
) -> tuple[float, float, pd.DataFrame]:
    """
    找到“左可行、右不可行”的初始速度区间
    """
    v_values = np.linspace(v_left, v_right, num_points)
    df_scan = scan_head_speed_values(
        v_values=v_values,
        pitch=pitch,
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=center_dist_threshold,
    )

    df_scan["feasible"] = df_scan["V_max"] <= v_lim

    left_good = None
    right_bad = None

    feasible_flags = df_scan["feasible"].values.astype(bool)
    v_arr = df_scan["v_head"].values

    for i in range(1, len(df_scan)):
        if feasible_flags[i - 1] and (not feasible_flags[i]):
            left_good = float(v_arr[i - 1])
            right_bad = float(v_arr[i])
            break

    if left_good is None or right_bad is None:
        raise RuntimeError("粗扫描未找到有效初始速度区间，请扩大范围或调整参数。")

    return left_good, right_bad, df_scan


def binary_search_max_feasible_head_speed(
    left_good: float,
    right_bad: float,
    tol: float,
    pitch: float,
    theta_start: float,
    num_handles: int,
    bench_width: float,
    v_lim: float,
    dt: float = 0.01,
    turn_radius: float = TURN_RADIUS_P4,
    R1: float = TURN_ARC_R1,
    R2: float = TURN_ARC_R2,
    ds: float = PATH_DS,
    center_dist_threshold: float = P4_CENTER_DIST_THRESHOLD,
) -> dict:
    """
    二分搜索最大允许龙头速度
    假定 left_good 可行，right_bad 不可行
    """
    history = []

    while right_bad - left_good > tol:
        mid = 0.5 * (left_good + right_bad)
        print(f"[问题5 二分] 检查 v_head = {mid:.6f}")

        result = evaluate_head_speed_once(
            v_head=mid,
            pitch=pitch,
            theta_start=theta_start,
            num_handles=num_handles,
            bench_width=bench_width,
            dt=dt,
            turn_radius=turn_radius,
            R1=R1,
            R2=R2,
            ds=ds,
            center_dist_threshold=center_dist_threshold,
            export_csv=False,
            export_fig=False,
        )

        feasible = result["V_max"] <= v_lim

        history.append({
            "v_head": result["v_head"],
            "V_max": result["V_max"],
            "feasible": feasible,
            "critical_time": result["critical_time"],
            "critical_handle_index": result["critical_handle_index"],
        })

        if feasible:
            left_good = mid
        else:
            right_bad = mid

    final_result = evaluate_head_speed_once(
        v_head=left_good,
        pitch=pitch,
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=center_dist_threshold,
        export_csv=False,
        export_fig=False,
    )

    return {
        "v_head_star": left_good,
        "left_good": left_good,
        "right_bad": right_bad,
        "history": pd.DataFrame(history),
        "final_result": final_result,
    }


# ============================================================
# 任务5：分析速度瓶颈把手、关键时刻和关键阶段
# ============================================================

def classify_path_stage_by_s(path_obj: dict, s_value: float) -> str:
    """
    依据弧长位置粗略判断属于盘入段/调头段/盘出段
    """
    boundary = path_obj["boundary"]
    turn_obj = path_obj["turn_obj"]

    # 用 raw_path 的分段长度做近似划分
    # 这里只做工程化近似：按完整路径 1/3, 1/3, 1/3 先粗分
    total_length = path_obj["total_length"]
    r1 = total_length / 3.0
    r2 = 2.0 * total_length / 3.0

    if s_value <= r1:
        return "盘入段"
    elif s_value <= r2:
        return "调头段"
    else:
        return "盘出段"


def analyze_speed_bottlenecks(
    final_result: dict,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    分析速度瓶颈点：找出速度最大的若干 (time, handle)
    """
    result_p4 = final_result["result_p4"]
    path_obj = result_p4["path_obj"]
    handle_motion = result_p4["handle_motion"]

    times = handle_motion["times"]
    s_mat = handle_motion["s_mat"]
    x_mat = handle_motion["x_mat"]
    y_mat = handle_motion["y_mat"]
    v_mat = handle_motion["v_mat"]

    flat_idx = np.argsort(v_mat.ravel())[::-1][:top_k]
    rows = []

    for idx in flat_idx:
        t_idx, h_idx = np.unravel_index(idx, v_mat.shape)
        s_value = float(s_mat[t_idx, h_idx])

        rows.append({
            "rank": len(rows) + 1,
            "speed": float(v_mat[t_idx, h_idx]),
            "time_index": int(t_idx),
            "time": float(times[t_idx]),
            "handle_index": int(h_idx),
            "x": float(x_mat[t_idx, h_idx]),
            "y": float(y_mat[t_idx, h_idx]),
            "s": s_value,
            "stage": classify_path_stage_by_s(path_obj, s_value),
        })

    return pd.DataFrame(rows)


def plot_key_handle_speed_curves(
    final_result: dict,
    handle_indices: list[int],
    save_path: str = f"{P5_EXPORT_PREFIX}_key_handle_speeds.png",
) -> None:
    """
    绘制若干关键把手的速度曲线
    """
    handle_motion = final_result["result_p4"]["handle_motion"]
    times = handle_motion["times"]
    v_mat = handle_motion["v_mat"]

    plt.figure(figsize=(8, 5))
    for h in handle_indices:
        if 0 <= h < v_mat.shape[1]:
            plt.plot(times, v_mat[:, h], label=f"handle {h}")

    plt.xlabel("t (s)")
    plt.ylabel("speed (m/s)")
    plt.title("Key Handle Speed Curves")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_critical_speed_position(
    final_result: dict,
    bench_width: float,
    save_path: str = f"{P5_EXPORT_PREFIX}_critical_speed_position.png",
) -> None:
    """
    绘制达到全局最大速度时的整队构型
    """
    result_p4 = final_result["result_p4"]
    handle_motion = result_p4["handle_motion"]

    critical_t_idx = final_result["critical_time_index"]

    plot_handle_configuration(
        x_mat=handle_motion["x_mat"],
        y_mat=handle_motion["y_mat"],
        time_index=critical_t_idx,
        width=bench_width,
        title="Configuration at Global Maximum Speed",
        save_path=save_path,
    )


# ============================================================
# 任务6：敏感性分析
# ============================================================

def sensitivity_analysis_speed_limit(
    v_lim_list: list[float],
    pitch: float,
    theta_start: float,
    num_handles: int,
    bench_width: float,
    v_left: float,
    v_right: float,
    v_tol: float,
    dt: float = 0.01,
    turn_radius: float = TURN_RADIUS_P4,
    R1: float = TURN_ARC_R1,
    R2: float = TURN_ARC_R2,
    ds: float = PATH_DS,
    center_dist_threshold: float = P4_CENTER_DIST_THRESHOLD,
    scan_points: int = P5_SCAN_POINTS,
) -> pd.DataFrame:
    """
    对速度上限 v_lim 做敏感性分析
    """
    rows = []

    for v_lim in v_lim_list:
        print(f"[问题5 敏感性] 正在分析 v_lim = {v_lim:.6f}")

        left_good, right_bad, _ = find_initial_speed_interval(
            v_left=v_left,
            v_right=v_right,
            num_points=scan_points,
            pitch=pitch,
            theta_start=theta_start,
            num_handles=num_handles,
            bench_width=bench_width,
            v_lim=v_lim,
            dt=dt,
            turn_radius=turn_radius,
            R1=R1,
            R2=R2,
            ds=ds,
            center_dist_threshold=center_dist_threshold,
        )

        search_result = binary_search_max_feasible_head_speed(
            left_good=left_good,
            right_bad=right_bad,
            tol=v_tol,
            pitch=pitch,
            theta_start=theta_start,
            num_handles=num_handles,
            bench_width=bench_width,
            v_lim=v_lim,
            dt=dt,
            turn_radius=turn_radius,
            R1=R1,
            R2=R2,
            ds=ds,
            center_dist_threshold=center_dist_threshold,
        )

        rows.append({
            "v_lim": v_lim,
            "v_head_star": search_result["v_head_star"],
            "V_max_at_star": search_result["final_result"]["V_max"],
        })

    return pd.DataFrame(rows)


def plot_sensitivity_curve(
    df_sens: pd.DataFrame,
    save_path: str = f"{P5_EXPORT_PREFIX}_sensitivity.png",
) -> None:
    """
    绘制速度上限—最大允许龙头速度关系图
    """
    plt.figure(figsize=(8, 5))
    plt.plot(df_sens["v_lim"], df_sens["v_head_star"], marker="o")
    plt.xlabel("Speed limit $v_{lim}$ (m/s)")
    plt.ylabel("Max feasible head speed $v_0^*$ (m/s)")
    plt.title("Sensitivity of Max Feasible Head Speed to Speed Limit")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# solve_problem5(...) 总封装
# ============================================================

def solve_problem5(
    pitch: float,
    theta_start: float,
    num_handles: int,
    bench_width: float,
    v_lim: float = P5_SPEED_LIMIT,
    v_left: float = P5_VHEAD_LEFT,
    v_right: float = P5_VHEAD_RIGHT,
    v_tol: float = P5_VHEAD_TOL,
    scan_points: int = P5_SCAN_POINTS,
    dt: float = 0.01,
    turn_radius: float = TURN_RADIUS_P4,
    R1: float = TURN_ARC_R1,
    R2: float = TURN_ARC_R2,
    ds: float = PATH_DS,
    center_dist_threshold: float = P4_CENTER_DIST_THRESHOLD,
    export_csv: bool = True,
    export_fig: bool = True,
    export_prefix: str = P5_EXPORT_PREFIX,
) -> dict:
    """
    问题五总封装：
    1. 粗扫描龙头速度
    2. 二分搜索最大允许龙头速度
    3. 分析关键速度瓶颈
    4. 做敏感性分析
    """
    print("问题5 - 任务2：粗扫描龙头速度")
    left_good, right_bad, df_scan = find_initial_speed_interval(
        v_left=v_left,
        v_right=v_right,
        num_points=scan_points,
        pitch=pitch,
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        v_lim=v_lim,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=center_dist_threshold,
    )

    print(f"问题5 - 初始区间找到: left_good={left_good:.6f}, right_bad={right_bad:.6f}")

    print("问题5 - 任务4：二分搜索最大允许龙头速度")
    search_result = binary_search_max_feasible_head_speed(
        left_good=left_good,
        right_bad=right_bad,
        tol=v_tol,
        pitch=pitch,
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        v_lim=v_lim,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=center_dist_threshold,
    )

    v_head_star = search_result["v_head_star"]
    final_result = search_result["final_result"]

    print(f"问题5 - 最大允许龙头速度近似为: {v_head_star:.6f}")

    print("问题5 - 任务5：速度瓶颈分析")
    df_bottleneck = analyze_speed_bottlenecks(final_result, top_k=10)

    unique_handles = df_bottleneck["handle_index"].drop_duplicates().tolist()[:5]

    print("问题5 - 任务6：敏感性分析")
    v_lim_list = [
        0.9 * v_lim,
        0.95 * v_lim,
        v_lim,
        1.05 * v_lim,
        1.10 * v_lim,
    ]
    df_sensitivity = sensitivity_analysis_speed_limit(
        v_lim_list=v_lim_list,
        pitch=pitch,
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        v_left=v_left,
        v_right=v_right,
        v_tol=v_tol,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=center_dist_threshold,
        scan_points=scan_points,
    )

    if export_fig:
        print("问题5 - 导出图像")
        plot_headspeed_vs_vmax(
            df_scan=df_scan.assign(feasible=df_scan["V_max"] <= v_lim),
            v_lim=v_lim,
            save_path=f"{export_prefix}_headspeed_vs_vmax.png",
        )
        plot_key_handle_speed_curves(
            final_result=final_result,
            handle_indices=unique_handles,
            save_path=f"{export_prefix}_key_handle_speeds.png",
        )
        plot_critical_speed_position(
            final_result=final_result,
            bench_width=bench_width,
            save_path=f"{export_prefix}_critical_speed_position.png",
        )
        plot_sensitivity_curve(
            df_sens=df_sensitivity,
            save_path=f"{export_prefix}_sensitivity.png",
        )

    if export_csv:
        print("问题5 - 导出表格")
        df_scan.assign(feasible=df_scan["V_max"] <= v_lim).to_csv(
            f"{export_prefix}_scan.csv", index=False, encoding="utf-8-sig"
        )
        search_result["history"].to_csv(
            f"{export_prefix}_binary_search_history.csv", index=False, encoding="utf-8-sig"
        )
        df_bottleneck.to_csv(
            f"{export_prefix}_bottleneck.csv", index=False, encoding="utf-8-sig"
        )
        df_sensitivity.to_csv(
            f"{export_prefix}_sensitivity.csv", index=False, encoding="utf-8-sig"
        )

        summary_row = build_speed_summary_row(final_result, v_lim)
        pd.DataFrame([summary_row]).to_csv(
            f"{export_prefix}_summary.csv", index=False, encoding="utf-8-sig"
        )

    return {
        "v_head_star": v_head_star,
        "scan_table": df_scan.assign(feasible=df_scan["V_max"] <= v_lim),
        "search_history": search_result["history"],
        "final_result": final_result,
        "bottleneck_table": df_bottleneck,
        "sensitivity_table": df_sensitivity,
    }

def main():
    print("========== 开始求解 ==========")

    # ============================================================
    # 公共参数
    # ============================================================
    theta_start = 32 * math.pi      # 第16圈起点
    num_handles = 224               # 把手总数，按题目真实值修改
    bench_width = 0.30              # 板凳宽度，按题目真实值修改

    # 问题1 / 问题2 公共参数
    t_start = 0.0
    t_end = 300.0
    dt = 0.01
    theta0_init = theta_start

    # 问题3 公共参数
    turn_radius = 4.5               # 调头空间半径，按题目真实值修改

    # 问题4 公共参数
    pitch_p4 = 0.55                 # 问题4使用的螺距
    v_head_p4 = 1.0                 # 问题4龙头速度
    R1 = 1.8                        # 调头第一段圆弧半径
    R2 = 1.2                        # 调头第二段圆弧半径
    ds = 0.01                       # 路径弧长离散步长

    # 问题5 公共参数
    v_lim = 2.0                     # 全体把手速度上限，按题目真实值修改

    # ============================================================
    # 问题1：盘入过程中的位置与速度求解
    # ============================================================
    result_p1 = solve_problem1(
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        export_csv=True,
    )

    # ============================================================
    # 问题2：盘入过程中的碰撞判定
    # ============================================================
    result_p2 = solve_problem2(
        x_mat=result_p1["x_mat"],
        y_mat=result_p1["y_mat"],
        dt=dt,
        width=bench_width,
        center_dist_threshold=1.5,
    )

    # ============================================================
    # 问题3：最小螺距求解
    # ============================================================
    result_p3 = solve_problem3(
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        theta0_init=theta0_init,
        num_handles=num_handles,
        width=bench_width,
        turn_radius=turn_radius,
        center_dist_threshold=1.5,
        pitch_left=0.20,
        pitch_right=0.80,
        pitch_tol=1e-3,
        pitch_scan_points=13,
        export_csv=True,
    )

    # 问题3求出来的最小可行螺距，后面问题4/5可以直接复用
    pitch_star = result_p3["pitch_star"]

    # ============================================================
    # 问题4：调头路径设计与运动仿真
    # 这里你可以选用 pitch_star，也可以先固定 0.55
    # ============================================================
    result_p4 = solve_problem4(
        pitch=pitch_star,            # 或写 pitch_p4
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        v_head=v_head_p4,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=1.5,
        export_csv=True,
        export_fig=True,
        export_prefix="problem4",
    )

    # ============================================================
    # 问题5：最大允许行进速度分析
    # ============================================================
    result_p5 = solve_problem5(
        pitch=pitch_star,            # 或写 pitch_p4
        theta_start=theta_start,
        num_handles=num_handles,
        bench_width=bench_width,
        v_lim=v_lim,
        v_left=0.10,
        v_right=5.00,
        v_tol=1e-3,
        scan_points=12,
        dt=dt,
        turn_radius=turn_radius,
        R1=R1,
        R2=R2,
        ds=ds,
        center_dist_threshold=1.5,
        export_csv=True,
        export_fig=True,
        export_prefix="problem5",
    )

    # ============================================================
    # 控制台摘要输出
    # ============================================================
    print("\n========== 求解结束 ==========")

    print("\n[问题2]")
    if result_p2["collision"]:
        print(f"首次碰撞时刻: {result_p2['refined']['time']:.8f} s")
    else:
        print("给定时间区间内未检测到碰撞")

    print("\n[问题3]")
    print(f"最小可行螺距: {result_p3['pitch_star']:.6f}")

    print("\n[问题4]")
    print(f"调头路径总长度: {result_p4['metrics']['total_length']:.6f}")
    print(f"调头过程最大把手速度: {result_p4['metrics']['max_speed']:.6f}")
    print(f"调头过程最小安全距离: {result_p4['metrics']['min_clearance']:.6f}")
    print(f"是否发生碰撞: {result_p4['metrics']['collision']}")

    print("\n[问题5]")
    print(f"最大允许龙头速度: {result_p5['v_head_star']:.6f}")
    print(f"对应全队最大速度: {result_p5['final_result']['V_max']:.6f}")
    print(f"关键把手编号: {result_p5['final_result']['critical_handle_index']}")
    print(f"关键时刻: {result_p5['final_result']['critical_time']:.6f} s")


if __name__ == "__main__":
    main()