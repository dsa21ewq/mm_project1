from __future__ import annotations

import argparse
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.optimize import brentq as _brentq
except Exception:  # pragma: no cover - optional dependency
    _brentq = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    def njit(*args: Any, **kwargs: Any):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator


# ============================================================
# Load the user's original implementation as a legacy module.
# The refactor keeps the mathematical building blocks, but rewires
# the orchestration so that later tasks reuse earlier results.
# ============================================================

_THIS_DIR = Path(__file__).resolve().parent
_LEGACY_PATH = _THIS_DIR / "problem1.py"

if not _LEGACY_PATH.exists():
    raise FileNotFoundError(f"Cannot find legacy file: {_LEGACY_PATH}")

_spec = importlib.util.spec_from_file_location("legacy_problem", _LEGACY_PATH)
legacy = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(legacy)


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class AppConfig:
    theta_start: float = 32 * math.pi
    num_handles: int = 224
    bench_width: float = 0.30

    # Problem 1/2/3
    t_start: float = 0.0
    t_end: float = 300.0
    dt: float = 0.01
    dt_search: float = 0.05

    # Problem 3
    turn_radius: float = 4.5
    pitch_left: float = 0.20
    pitch_right: float = 0.80
    pitch_tol: float = 1e-3
    pitch_scan_points: int = 13

    # Problem 4/5
    pitch_p4: float = 0.55
    v_head_p4: float = 1.0
    v_lim: float = 2.0
    v_left: float = 0.10
    v_right: float = 5.00
    v_tol: float = 1e-3
    scan_points: int = 12

    R1: float = 1.8
    R2: float = 1.2
    ds_path: float = 0.02
    ds_sample: float = 0.02
    center_dist_threshold: float = 1.5

    export_csv: bool = True
    export_fig: bool = True


# ============================================================
# Utilities
# ============================================================

def print_banner(msg: str) -> None:
    print("\n" + "=" * 12 + f" {msg} " + "=" * 12)


def safe_linspace(left: float, right: float, num: int) -> np.ndarray:
    if num < 2:
        return np.array([left], dtype=float)
    return np.linspace(left, right, num)



# ============================================================
# Low-level accelerators
# - Optional SciPy Brent root finder
# - Optional Numba kernels for rectangle geometry
# - Monkey-patch legacy hotspots so all downstream calls benefit
# ============================================================

_GEOM_EPS = 1e-12


def _bisect_root(
    f,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> float:
    f_left = f(left)
    f_right = f(right)
    if f_left == 0.0:
        return float(left)
    if f_right == 0.0:
        return float(right)

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = f(mid)
        if abs(f_mid) < tol or (right - left) < tol:
            return float(mid)
        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid
    return float(0.5 * (left + right))


def _solve_bracketed_root(
    f,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> float:
    if _brentq is not None:
        try:
            return float(_brentq(f, left, right, xtol=tol, maxiter=max_iter))
        except Exception:
            pass
    return _bisect_root(f, left, right, tol=tol, max_iter=max_iter)


def solve_next_theta_fast(
    theta_curr: float,
    target_len: float,
    step_out: float = 0.02,
    max_expand: int = 20000,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    left = float(theta_curr)

    def f(theta_next: float) -> float:
        return legacy.distance_equation(theta_next, theta_curr, target_len)

    f_left = f(left)
    right = left + step_out
    f_right = f(right)

    expand_count = 0
    while f_left * f_right > 0.0 and expand_count < max_expand:
        right += step_out
        f_right = f(right)
        expand_count += 1

    if f_left * f_right > 0.0:
        raise RuntimeError(
            f"未找到根区间: theta_curr={theta_curr:.6f}, target_len={target_len}"
        )

    return _solve_bracketed_root(f, left, right, tol=tol, max_iter=max_iter)


def solve_next_theta_pitch_fast(
    theta_curr: float,
    target_len: float,
    pitch: float,
    step_out: float = 0.02,
    max_expand: int = 20000,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    left = float(theta_curr)

    def f(theta_next: float) -> float:
        return legacy.distance_equation_pitch(theta_next, theta_curr, target_len, pitch)

    f_left = f(left)
    right = left + step_out
    f_right = f(right)

    expand_count = 0
    while f_left * f_right > 0.0 and expand_count < max_expand:
        right += step_out
        f_right = f(right)
        expand_count += 1

    if f_left * f_right > 0.0:
        raise RuntimeError(
            f"未找到根区间: theta_curr={theta_curr:.6f}, target_len={target_len}, pitch={pitch}"
        )

    return _solve_bracketed_root(f, left, right, tol=tol, max_iter=max_iter)


def solve_next_handle_s_along_path_fast(
    s_prev: float,
    prev_xy: np.ndarray,
    target_len: float,
    path_points: np.ndarray,
    path_s: np.ndarray,
    ds_search: float = 0.002,
    max_back: float = 20.0,
) -> float:
    left = max(0.0, float(s_prev) - max_back)
    right = float(s_prev)

    def f(s_value: float) -> float:
        p = legacy.path_position(path_points, path_s, s_value)
        return float(np.linalg.norm(prev_xy - p) - target_len)

    s_hi = right
    s_lo = s_hi
    found = False

    s = s_hi
    while s > left:
        s = max(left, s - ds_search)
        if f(s) >= 0.0:
            s_lo = s
            found = True
            break

    if not found:
        return float(left)

    return _solve_bracketed_root(f, s_lo, s_hi, tol=1e-8, max_iter=60)


def compute_velocity_matrix_fast(x_mat: np.ndarray, y_mat: np.ndarray, dt: float) -> np.ndarray:
    x = np.asarray(x_mat, dtype=float)
    y = np.asarray(y_mat, dtype=float)
    if x.shape != y.shape:
        raise ValueError('x_mat 与 y_mat 的 shape 不一致')
    num_times, num_handles = x.shape
    if num_times == 0:
        return np.zeros_like(x)
    if num_times == 1:
        return np.zeros_like(x)

    dx = np.diff(x, axis=0)
    dy = np.diff(y, axis=0)
    speed = np.sqrt(dx * dx + dy * dy) / float(dt)
    return np.vstack([speed, speed[-1:, :]])


@njit(cache=False)
def _build_all_rectangles_at_one_time_numba(
    x_row: np.ndarray,
    y_row: np.ndarray,
    width: float,
) -> np.ndarray:
    n = len(x_row) - 1
    rects = np.empty((n, 4, 2), dtype=np.float64)
    half_width = 0.5 * width

    for i in range(n):
        fx = x_row[i]
        fy = y_row[i]
        bx = x_row[i + 1]
        by = y_row[i + 1]

        dx = bx - fx
        dy = by - fy
        norm = math.sqrt(dx * dx + dy * dy)

        if norm < _GEOM_EPS:
            vx = 0.0
            vy = 1.0
        else:
            ux = dx / norm
            uy = dy / norm
            vx = -uy
            vy = ux

        rects[i, 0, 0] = fx + half_width * vx
        rects[i, 0, 1] = fy + half_width * vy
        rects[i, 1, 0] = fx - half_width * vx
        rects[i, 1, 1] = fy - half_width * vy
        rects[i, 2, 0] = bx - half_width * vx
        rects[i, 2, 1] = by - half_width * vy
        rects[i, 3, 0] = bx + half_width * vx
        rects[i, 3, 1] = by + half_width * vy

    return rects


@njit(cache=False)
def _sat_intersect_rectangle_numba(rect1: np.ndarray, rect2: np.ndarray) -> bool:
    for poly_index in range(2):
        poly = rect1 if poly_index == 0 else rect2
        for i in range(4):
            j = (i + 1) % 4
            ex = poly[j, 0] - poly[i, 0]
            ey = poly[j, 1] - poly[i, 1]

            ax = -ey
            ay = ex
            norm = math.sqrt(ax * ax + ay * ay)
            if norm < _GEOM_EPS:
                continue
            ax /= norm
            ay /= norm

            min1 = 1e308
            max1 = -1e308
            min2 = 1e308
            max2 = -1e308

            for k in range(4):
                p1 = rect1[k, 0] * ax + rect1[k, 1] * ay
                p2 = rect2[k, 0] * ax + rect2[k, 1] * ay
                if p1 < min1:
                    min1 = p1
                if p1 > max1:
                    max1 = p1
                if p2 < min2:
                    min2 = p2
                if p2 > max2:
                    max2 = p2

            if max1 < min2 or max2 < min1:
                return False

    return True


@njit(cache=False)
def _point_to_segment_distance_numba(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby

    if denom < _GEOM_EPS:
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)

    t = (apx * abx + apy * aby) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=False)
def _segment_to_segment_distance_numba(
    a1x: float,
    a1y: float,
    a2x: float,
    a2y: float,
    b1x: float,
    b1y: float,
    b2x: float,
    b2y: float,
) -> float:
    d1 = _point_to_segment_distance_numba(a1x, a1y, b1x, b1y, b2x, b2y)
    d2 = _point_to_segment_distance_numba(a2x, a2y, b1x, b1y, b2x, b2y)
    d3 = _point_to_segment_distance_numba(b1x, b1y, a1x, a1y, a2x, a2y)
    d4 = _point_to_segment_distance_numba(b2x, b2y, a1x, a1y, a2x, a2y)
    return min(d1, d2, d3, d4)


@njit(cache=False)
def _rectangle_min_distance_no_intersect_numba(rect1: np.ndarray, rect2: np.ndarray) -> float:
    min_dist = 1e308
    for i in range(4):
        j = (i + 1) % 4
        a1x = rect1[i, 0]
        a1y = rect1[i, 1]
        a2x = rect1[j, 0]
        a2y = rect1[j, 1]
        for k in range(4):
            l = (k + 1) % 4
            b1x = rect2[k, 0]
            b1y = rect2[k, 1]
            b2x = rect2[l, 0]
            b2y = rect2[l, 1]
            d = _segment_to_segment_distance_numba(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y)
            if d < min_dist:
                min_dist = d
    return min_dist


def _as_rectangles_array(rectangles: Any) -> np.ndarray:
    rects = np.asarray(rectangles, dtype=float)
    if rects.ndim != 3 or rects.shape[1:] != (4, 2):
        raise ValueError('rectangles 必须可转为 shape=(n,4,2) 的数组')
    return rects


def build_all_rectangles_at_one_time_fast(
    x_row: np.ndarray,
    y_row: np.ndarray,
    width: float,
) -> np.ndarray:
    x = np.asarray(x_row, dtype=np.float64)
    y = np.asarray(y_row, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError('x_row 与 y_row 必须是一维且长度一致')
    return _build_all_rectangles_at_one_time_numba(x, y, float(width))


def _candidate_pairs_with_centers(
    rectangles: Any,
    center_dist_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rects = _as_rectangles_array(rectangles)
    n = rects.shape[0]
    centers = rects.mean(axis=1)
    if n == 0:
        return rects, centers, np.empty((0, 2), dtype=np.int64)

    diff = centers[:, None, :] - centers[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff, optimize=True)
    mask = np.triu(np.ones((n, n), dtype=bool), k=2)
    mask &= dist2 <= float(center_dist_threshold) ** 2
    pairs = np.argwhere(mask).astype(np.int64, copy=False)
    return rects, centers, pairs


def candidate_pairs_by_center_distance_fast(
    rectangles: Any,
    center_dist_threshold: float,
) -> list[tuple[int, int]]:
    _, _, pairs = _candidate_pairs_with_centers(rectangles, center_dist_threshold)
    return [(int(i), int(j)) for i, j in pairs]


def sat_intersect_rectangle_fast(rect1: np.ndarray, rect2: np.ndarray) -> bool:
    return bool(
        _sat_intersect_rectangle_numba(
            np.asarray(rect1, dtype=np.float64),
            np.asarray(rect2, dtype=np.float64),
        )
    )


def rectangle_min_distance_fast(rect1: np.ndarray, rect2: np.ndarray) -> float:
    r1 = np.asarray(rect1, dtype=np.float64)
    r2 = np.asarray(rect2, dtype=np.float64)
    if _sat_intersect_rectangle_numba(r1, r2):
        return 0.0
    return float(_rectangle_min_distance_no_intersect_numba(r1, r2))


def has_collision_at_one_time_fast(
    rectangles: Any,
    center_dist_threshold: float,
    eps: float = 1e-9,
) -> bool:
    rects, _, pairs = _candidate_pairs_with_centers(rectangles, center_dist_threshold)
    for i, j in pairs:
        rect_i = rects[i]
        rect_j = rects[j]
        if _sat_intersect_rectangle_numba(rect_i, rect_j):
            return True
        if _rectangle_min_distance_no_intersect_numba(rect_i, rect_j) <= eps:
            return True
    return False


def min_pairwise_clearance_at_one_time_fast(
    rectangles: Any,
    center_dist_threshold: float,
) -> dict[str, Any]:
    rects, _, pairs = _candidate_pairs_with_centers(rectangles, center_dist_threshold)
    if len(pairs) == 0:
        return {
            'min_dist': float('inf'),
            'pair': None,
            'collision': False,
        }

    min_dist = float('inf')
    min_pair = None
    for i, j in pairs:
        rect_i = rects[i]
        rect_j = rects[j]
        if _sat_intersect_rectangle_numba(rect_i, rect_j):
            return {
                'min_dist': 0.0,
                'pair': (int(i), int(j)),
                'collision': True,
            }

        d = float(_rectangle_min_distance_no_intersect_numba(rect_i, rect_j))
        if d < min_dist:
            min_dist = d
            min_pair = (int(i), int(j))

    return {
        'min_dist': min_dist,
        'pair': min_pair,
        'collision': False,
    }


def detect_collisions_at_one_time_fast(
    rectangles: Any,
    center_dist_threshold: float,
    eps: float = 1e-9,
) -> dict[str, Any]:
    rects, centers, pairs = _candidate_pairs_with_centers(rectangles, center_dist_threshold)
    detail: list[dict[str, Any]] = []
    collision_pairs: list[tuple[int, int]] = []

    for i, j in pairs:
        rect_i = rects[i]
        rect_j = rects[j]
        center_dx = centers[i, 0] - centers[j, 0]
        center_dy = centers[i, 1] - centers[j, 1]
        center_dist = float(math.sqrt(center_dx * center_dx + center_dy * center_dy))

        intersect = bool(_sat_intersect_rectangle_numba(rect_i, rect_j))
        if intersect:
            min_dist = 0.0
        else:
            min_dist = float(_rectangle_min_distance_no_intersect_numba(rect_i, rect_j))

        detail.append({
            'i': int(i),
            'j': int(j),
            'intersect': intersect,
            'min_dist': min_dist,
            'center_dist': center_dist,
        })

        if intersect or min_dist <= eps:
            collision_pairs.append((int(i), int(j)))

    return {
        'collision': len(collision_pairs) > 0,
        'pairs': collision_pairs,
        'detail': detail,
    }


def detect_collisions_over_time_p4_fast(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    width: float,
    center_dist_threshold: float,
) -> dict[str, Any]:
    num_times = x_mat.shape[0]

    first_collision_index = None
    first_collision_pairs = None
    min_clearance = float('inf')
    min_clearance_index = None
    min_clearance_pair = None

    for t_idx in range(num_times):
        rects = build_all_rectangles_at_one_time_fast(x_mat[t_idx], y_mat[t_idx], width)
        rects, _, pairs = _candidate_pairs_with_centers(rects, center_dist_threshold)

        current_min = float('inf')
        current_pair = None
        collided_pairs: list[tuple[int, int]] = []

        for i, j in pairs:
            rect_i = rects[i]
            rect_j = rects[j]
            if _sat_intersect_rectangle_numba(rect_i, rect_j):
                collided_pairs.append((int(i), int(j)))
                if current_min > 0.0:
                    current_min = 0.0
                    current_pair = (int(i), int(j))
            else:
                d = float(_rectangle_min_distance_no_intersect_numba(rect_i, rect_j))
                if d < current_min:
                    current_min = d
                    current_pair = (int(i), int(j))

        if current_min < min_clearance:
            min_clearance = current_min
            min_clearance_index = t_idx
            min_clearance_pair = current_pair

        if first_collision_index is None and collided_pairs:
            first_collision_index = t_idx
            first_collision_pairs = collided_pairs

    return {
        'collision': first_collision_index is not None,
        'first_collision_index': first_collision_index,
        'first_collision_pairs': first_collision_pairs,
        'min_clearance': min_clearance,
        'min_clearance_index': min_clearance_index,
        'min_clearance_pair': min_clearance_pair,
    }


def install_legacy_fastpath() -> None:
    legacy.solve_next_theta = solve_next_theta_fast
    legacy.solve_next_theta_pitch = solve_next_theta_pitch_fast
    legacy.solve_next_handle_s_along_path = solve_next_handle_s_along_path_fast
    legacy.compute_velocity_matrix = compute_velocity_matrix_fast

    legacy.build_all_rectangles_at_one_time = build_all_rectangles_at_one_time_fast
    legacy.candidate_pairs_by_center_distance = candidate_pairs_by_center_distance_fast
    legacy.sat_intersect_rectangle = sat_intersect_rectangle_fast
    legacy.rectangle_min_distance = rectangle_min_distance_fast
    legacy.min_pairwise_clearance_at_one_time = min_pairwise_clearance_at_one_time_fast
    legacy.detect_collisions_at_one_time = detect_collisions_at_one_time_fast
    legacy.detect_collisions_over_time_p4 = detect_collisions_over_time_p4_fast


install_legacy_fastpath()


# ============================================================
# Faster Problem 2
# - Do not build and store rectangles for all time steps.
# - Stream over time and stop once the first collision is found.
# ============================================================

def solve_problem2_streaming(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    dt: float,
    width: float,
    center_dist_threshold: float,
    export_csv: bool = True,
    export_fig: bool = True,
    export_prefix: str = "problem2",
) -> dict[str, Any]:
    print_banner("问题2（流式碰撞检测）")

    coarse = None
    num_times = x_mat.shape[0]

    for t_idx in range(num_times):
        rects = build_all_rectangles_at_one_time_fast(x_mat[t_idx], y_mat[t_idx], width)
        if has_collision_at_one_time_fast(rects, center_dist_threshold):
            coarse_result = detect_collisions_at_one_time_fast(rects, center_dist_threshold)
            coarse = {
                "time_index": t_idx,
                "time": t_idx * dt,
                "pairs": coarse_result["pairs"],
                "detail": coarse_result["detail"],
            }
            break

        if t_idx % 1000 == 0:
            print(f"问题2 流式进度: {t_idx}/{num_times}")

    if coarse is None:
        print("问题2：在给定时间区间内未检测到碰撞。")
        return {
            "collision": False,
            "coarse": None,
            "refined": None,
            "details": [],
        }

    print(f"问题2：粗搜索首次碰撞时刻约为 {coarse['time']:.6f} s")

    refined = legacy.refine_first_collision_bisection(
        x_mat=x_mat,
        y_mat=y_mat,
        coarse_index=coarse["time_index"],
        dt=dt,
        width=width,
        center_dist_threshold=center_dist_threshold,
        tol=1e-5,
        max_iter=60,
    )
    print(f"问题2：细化后首次碰撞时刻 {refined['time']:.8f} s")

    refined_rectangles = legacy.interpolate_rectangles(
        x_mat, y_mat, refined["left_idx"], refined["alpha"], width
    )
    refined_result = detect_collisions_at_one_time_fast(
        refined_rectangles,
        center_dist_threshold=center_dist_threshold,
    )
    details = legacy.extract_collision_details(refined_rectangles, refined_result)

    if export_fig:
        legacy.plot_rectangles(
            refined_rectangles,
            collision_pairs=refined_result["pairs"],
            title=f"First Collision at t={refined['time']:.6f}s",
            save_path=f"{export_prefix}_first_collision.png",
        )

    df_collision = None
    if export_csv:
        df_collision = legacy.save_collision_report(
            collision_time=refined["time"],
            collision_details=details,
            save_csv_path=f"{export_prefix}_collision_report.csv",
        )

    return {
        "collision": True,
        "coarse": coarse,
        "refined": refined,
        "details": details,
        "df_collision": df_collision,
    }


# ============================================================
# Faster Problem 3
# - Search with a coarser dt first.
# - Verify final/near-final pitches with the fine dt.
# ============================================================

def evaluate_pitch_feasibility_fast(
    pitch: float,
    cfg: AppConfig,
    dt_eval: float,
) -> dict[str, Any]:
    sim = legacy.solve_problem1_with_pitch(
        pitch=pitch,
        t_start=cfg.t_start,
        t_end=cfg.t_end,
        dt=dt_eval,
        theta0_init=cfg.theta_start,
        num_handles=cfg.num_handles,
    )

    df_head = sim["df_head"]
    x_mat = sim["x_mat"]
    y_mat = sim["y_mat"]

    enter_idx = legacy.first_enter_turn_radius(df_head, cfg.turn_radius)
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
            "dt_eval": dt_eval,
        }

    min_clearance = float("inf")
    min_clearance_time = None
    min_clearance_pair = None
    collision = False
    collision_time = None
    collision_pair = None

    for t_idx in range(int(enter_idx) + 1):
        rects = build_all_rectangles_at_one_time_fast(
            x_mat[t_idx],
            y_mat[t_idx],
            cfg.bench_width,
        )
        clearance_info = min_pairwise_clearance_at_one_time_fast(
            rects,
            cfg.center_dist_threshold,
        )

        if clearance_info["min_dist"] < min_clearance:
            min_clearance = clearance_info["min_dist"]
            min_clearance_time = t_idx * dt_eval
            min_clearance_pair = clearance_info["pair"]

        if clearance_info["collision"]:
            collision = True
            collision_time = t_idx * dt_eval
            collision_pair = clearance_info["pair"]
            break

    feasible = (not collision) and (enter_idx is not None)
    return {
        "pitch": pitch,
        "reachable": True,
        "feasible": feasible,
        "enter_index": int(enter_idx),
        "enter_time": float(enter_idx * dt_eval),
        "collision": collision,
        "collision_time": collision_time,
        "collision_pair": collision_pair,
        "min_clearance": min_clearance,
        "min_clearance_time": min_clearance_time,
        "min_clearance_pair": min_clearance_pair,
        "dt_eval": dt_eval,
    }


def solve_problem3_fast(cfg: AppConfig, export_prefix: str = "problem3") -> dict[str, Any]:
    print_banner("问题3（粗到细搜索）")

    # 1) coarse scan with a larger dt
    coarse_rows: list[dict[str, Any]] = []
    pitch_values = safe_linspace(cfg.pitch_left, cfg.pitch_right, cfg.pitch_scan_points)
    for pitch in pitch_values:
        print(f"问题3 粗扫描 pitch={pitch:.6f}, dt={cfg.dt_search}")
        row = evaluate_pitch_feasibility_fast(pitch, cfg, dt_eval=cfg.dt_search)
        coarse_rows.append(row)

    df_scan = pd.DataFrame(coarse_rows)

    left_bad = None
    right_good = None
    flags = df_scan["feasible"].fillna(False).astype(bool).to_numpy()
    pitches = df_scan["pitch"].to_numpy()
    for i in range(1, len(df_scan)):
        if (not flags[i - 1]) and flags[i]:
            left_bad = float(pitches[i - 1])
            right_good = float(pitches[i])
            break

    if left_bad is None or right_good is None:
        raise RuntimeError("问题3：粗扫描未找到“左不可行、右可行”区间，请扩大 pitch 搜索范围。")

    print(f"问题3 初始区间: left_bad={left_bad:.6f}, right_good={right_good:.6f}")

    # 2) binary search still on the coarse dt
    history: list[dict[str, Any]] = []
    while right_good - left_bad > cfg.pitch_tol:
        mid = 0.5 * (left_bad + right_good)
        result = evaluate_pitch_feasibility_fast(mid, cfg, dt_eval=cfg.dt_search)
        history.append({
            "pitch": result["pitch"],
            "feasible": result["feasible"],
            "reachable": result["reachable"],
            "min_clearance": result["min_clearance"],
            "collision_time": result["collision_time"],
            "dt_eval": result["dt_eval"],
        })
        print(
            f"问题3 二分 pitch={mid:.6f}, feasible={result['feasible']}, "
            f"D_min={result['min_clearance']}"
        )
        if result["feasible"]:
            right_good = mid
        else:
            left_bad = mid

    pitch_star = right_good

    # 3) verify near the final solution with the fine dt
    deltas = [-2 * cfg.pitch_tol, -cfg.pitch_tol, 0.0, cfg.pitch_tol, 2 * cfg.pitch_tol]
    sens_rows: list[dict[str, Any]] = []
    for delta in deltas:
        pitch = pitch_star + delta
        if pitch <= 0:
            continue
        print(f"问题3 精验 pitch={pitch:.6f}, dt={cfg.dt}")
        sens_rows.append(evaluate_pitch_feasibility_fast(pitch, cfg, dt_eval=cfg.dt))

    df_sensitivity = pd.DataFrame(sens_rows)
    df_sensitivity = df_sensitivity.sort_values("pitch").reset_index(drop=True)

    feasible_verified = df_sensitivity[df_sensitivity["feasible"] == True]
    if not feasible_verified.empty:
        pitch_star_verified = float(feasible_verified["pitch"].iloc[0])
    else:
        pitch_star_verified = pitch_star

    final_result = evaluate_pitch_feasibility_fast(pitch_star_verified, cfg, dt_eval=cfg.dt)

    if cfg.export_csv:
        df_scan.to_csv(f"{export_prefix}_coarse_scan.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(history).to_csv(
            f"{export_prefix}_binary_search_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        df_sensitivity.to_csv(
            f"{export_prefix}_verification.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return {
        "pitch_star": pitch_star_verified,
        "scan_table": df_scan,
        "search_history": pd.DataFrame(history),
        "sensitivity_table": df_sensitivity,
        "final_result": final_result,
    }


# ============================================================
# Problem 4 in arc-length domain
# Key idea:
#   path is already parameterized by s.
#   Therefore positions depend on s, not on v_head.
#   For a constant v_head, actual time is t = s / v_head.
#   Handle velocities scale linearly with v_head.
# This lets Problem 5 reuse Problem 4 without re-running Problem 4.
# ============================================================

def build_composite_turning_path_ex(
    pitch: float,
    theta_start: float,
    turn_radius: float,
    R1: float,
    R2: float,
    ds: float,
) -> dict[str, Any]:
    boundary = legacy.build_turn_boundary_points(pitch, turn_radius)
    P1 = boundary["P1"]
    T1 = boundary["T1"]
    P2 = boundary["P2"]
    T2 = boundary["T2"]
    theta1 = boundary["theta1"]

    inbound = legacy.build_inbound_spiral_segment(
        pitch=pitch,
        theta_start=theta_start,
        theta_end=theta1,
        num=1500,
    )
    turn_obj = legacy.build_two_arc_turn_path(
        P1=P1,
        T1=T1,
        P2=P2,
        T2=T2,
        R1=R1,
        R2=R2,
        arc_samples=300,
    )
    outbound = legacy.build_outbound_spiral_segment(
        pitch=pitch,
        theta_start=theta1,
        theta_end=theta_start,
        num=1500,
    )

    len_in = legacy.polyline_arclength(inbound)[-1]
    len_turn = legacy.polyline_arclength(turn_obj["turn_points"])[-1]
    len_out = legacy.polyline_arclength(outbound)[-1]

    raw_path = np.vstack([inbound, turn_obj["turn_points"], outbound])
    path_points, path_s = legacy.resample_polyline_by_step(raw_path, ds)

    return {
        "boundary": boundary,
        "turn_obj": turn_obj,
        "raw_path": raw_path,
        "path_points": path_points,
        "path_s": path_s,
        "total_length": float(path_s[-1]),
        "s_stage_in_end": float(len_in),
        "s_stage_turn_end": float(len_in + len_turn),
        "len_in": float(len_in),
        "len_turn": float(len_turn),
        "len_out": float(len_out),
    }


def build_head_motion_by_arclength(
    path_obj: dict[str, Any],
    ds_sample: float,
    num_handles: int,
    margin: float = 2.0,
) -> dict[str, Any]:
    total_length = float(path_obj["total_length"])
    chain_length = float(sum(legacy.get_handle_spacing(i) for i in range(num_handles - 1)))
    if total_length <= 0.0:
        raise ValueError("复合路径长度必须为正。")

    s_turn_start = float(path_obj["s_stage_in_end"])
    s_turn_end = float(path_obj["s_stage_turn_end"])
    # 对当前参数，chain_length 往往已经超过调头段末端。
    # 因此问题4采用“调头局部窗口”近似：只对调头段附近采样，
    # 而不是要求整龙先完全铺到路径上再开始。
    s0 = max(0.0, s_turn_start - margin)
    s1 = min(total_length, s_turn_end + margin)

    if s1 <= s0:
        raise ValueError(
            f"调头窗口非法: start={s0:.6f}, end={s1:.6f}。"
            "请检查路径长度或 margin 设置。"
        )

    s_head = np.arange(s0, s1 + ds_sample, ds_sample, dtype=float)
    if len(s_head) == 0:
        s_head = np.array([s0, s1], dtype=float)
    if s_head[-1] > s1:
        s_head[-1] = s1
    elif s_head[-1] < s1:
        s_head = np.append(s_head, s1)

    head_xy = legacy.path_positions_batch(path_obj["path_points"], path_obj["path_s"], s_head)
    return {
        # Use arc-length as the time-like parameter here.
        # Then v_mat computed later is actually d(position)/ds.
        "times": s_head.copy(),
        "s_head": s_head,
        "head_xy": head_xy,
        "total_time": s1 - s0,
        "start_s_head": s0,
        "end_s_head": s1,
    }


def solve_problem4_base(cfg: AppConfig, pitch: float) -> dict[str, Any]:
    print_banner("问题4 基础解（弧长域）")
    path_obj = build_composite_turning_path_ex(
        pitch=pitch,
        theta_start=cfg.theta_start,
        turn_radius=cfg.turn_radius,
        R1=cfg.R1,
        R2=cfg.R2,
        ds=cfg.ds_path,
    )

    head_motion_s = build_head_motion_by_arclength(path_obj, cfg.ds_sample, cfg.num_handles)
    gain_motion = legacy.solve_all_handles_on_composite_path(
        head_motion=head_motion_s,
        path_points=path_obj["path_points"],
        path_s=path_obj["path_s"],
        num_handles=cfg.num_handles,
    )

    collision_result = legacy.detect_collisions_over_time_p4(
        x_mat=gain_motion["x_mat"],
        y_mat=gain_motion["y_mat"],
        width=cfg.bench_width,
        center_dist_threshold=cfg.center_dist_threshold,
    )

    gain_metrics = {
        'total_length': path_obj['total_length'],
        'sample_count': len(gain_motion['times']),
        'gain_max': float(np.max(gain_motion['v_mat'])),
        'min_clearance': collision_result['min_clearance'],
        'collision': collision_result['collision'],
    }

    vmax_idx = np.unravel_index(np.argmax(gain_motion["v_mat"]), gain_motion["v_mat"].shape)
    t_idx = int(vmax_idx[0])
    h_idx = int(vmax_idx[1])
    critical_s_head = float(gain_motion["times"][t_idx])

    return {
        "pitch": pitch,
        "path_obj": path_obj,
        "gain_motion": gain_motion,
        "collision_result": collision_result,
        "gain_metrics": gain_metrics,
        "gain_max": float(np.max(gain_motion["v_mat"])),
        "critical_time_index": t_idx,
        "critical_handle_index": h_idx,
        "critical_s_head": critical_s_head,
        "start_s_head": float(head_motion_s["start_s_head"]),
        "critical_x": float(gain_motion["x_mat"][t_idx, h_idx]),
        "critical_y": float(gain_motion["y_mat"][t_idx, h_idx]),
    }



def materialize_problem4_from_base(
    base: dict[str, Any],
    v_head: float,
) -> dict[str, Any]:
    gain_motion = base["gain_motion"]
    path_obj = base["path_obj"]
    collision_result = base["collision_result"]

    start_s_head = float(base["start_s_head"])
    handle_motion = {
        "times": (gain_motion["times"] - start_s_head) / v_head,
        "s_mat": gain_motion["s_mat"].copy(),
        "x_mat": gain_motion["x_mat"].copy(),
        "y_mat": gain_motion["y_mat"].copy(),
        "v_mat": gain_motion["v_mat"] * v_head,
    }

    metrics = {
        "total_length": path_obj["total_length"],
        "total_time": (gain_motion["times"][-1] - start_s_head) / v_head,
        "max_speed": base["gain_max"] * v_head,
        "max_speed_time_index": base["critical_time_index"],
        "max_speed_handle_index": base["critical_handle_index"],
        "min_clearance": collision_result["min_clearance"],
        "min_clearance_time_index": collision_result["min_clearance_index"],
        "min_clearance_pair": collision_result["min_clearance_pair"],
        "collision": collision_result["collision"],
        "first_collision_index": collision_result["first_collision_index"],
        "first_collision_pairs": collision_result["first_collision_pairs"],
        "critical_time": (base["critical_s_head"] - start_s_head) / v_head,
        "critical_x": base["critical_x"],
        "critical_y": base["critical_y"],
    }

    return {
        "path_obj": path_obj,
        "head_motion": {
            "times": (gain_motion["times"] - start_s_head) / v_head,
            "s_head": gain_motion["times"].copy(),
            "head_xy": legacy.path_positions_batch(
                path_obj["path_points"],
                path_obj["path_s"],
                gain_motion["times"],
            ),
            "total_time": (gain_motion["times"][-1] - start_s_head) / v_head,
        },
        "handle_motion": handle_motion,
        "collision_result": collision_result,
        "metrics": metrics,
    }



def classify_stage_exact(path_obj: dict[str, Any], s_value: float) -> str:
    if s_value <= path_obj["s_stage_in_end"]:
        return "盘入段"
    if s_value <= path_obj["s_stage_turn_end"]:
        return "调头段"
    return "盘出段"



def analyze_speed_bottlenecks_fast(final_result: dict[str, Any], top_k: int = 10) -> pd.DataFrame:
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
    for rank, idx in enumerate(flat_idx, start=1):
        t_idx, h_idx = np.unravel_index(idx, v_mat.shape)
        s_value = float(s_mat[t_idx, h_idx])
        rows.append({
            "rank": rank,
            "speed": float(v_mat[t_idx, h_idx]),
            "time_index": int(t_idx),
            "time": float(times[t_idx]),
            "handle_index": int(h_idx),
            "x": float(x_mat[t_idx, h_idx]),
            "y": float(y_mat[t_idx, h_idx]),
            "s": s_value,
            "stage": classify_stage_exact(path_obj, s_value),
        })
    return pd.DataFrame(rows)



def solve_problem4_fast(
    cfg: AppConfig,
    pitch: float,
    v_head: float,
    export_prefix: str = "problem4",
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print_banner("问题4（基于弧长域复用）")
    if base is None:
        base = solve_problem4_base(cfg, pitch)
    result_p4 = materialize_problem4_from_base(base, v_head=v_head)

    if cfg.export_fig:
        legacy.plot_composite_path(result_p4["path_obj"], save_path=f"{export_prefix}_path.png")
        legacy.plot_speed_curve(result_p4["handle_motion"], save_path=f"{export_prefix}_speed_curve.png")

        ntime = len(result_p4["handle_motion"]["times"])
        idx_list = [0, ntime // 2, ntime - 1]
        name_list = ["start", "middle", "end"]
        for idx, name in zip(idx_list, name_list):
            legacy.plot_handle_configuration(
                x_mat=result_p4["handle_motion"]["x_mat"],
                y_mat=result_p4["handle_motion"]["y_mat"],
                time_index=idx,
                width=cfg.bench_width,
                title=f"Problem 4 Configuration - {name}",
                save_path=f"{export_prefix}_config_{name}.png",
            )

    if cfg.export_csv:
        legacy.export_problem4_tables(
            path_obj=result_p4["path_obj"],
            handle_motion=result_p4["handle_motion"],
            metrics=result_p4["metrics"],
            prefix=export_prefix,
        )

    return result_p4


# ============================================================
# Faster Problem 5
# Analytic scaling on top of Problem 4 base solution.
# ============================================================

def build_problem5_final_result(base: dict[str, Any], v_head: float) -> dict[str, Any]:
    result_p4 = materialize_problem4_from_base(base, v_head=v_head)
    start_s_head = float(base["start_s_head"])
    return {
        "v_head": v_head,
        "V_max": base["gain_max"] * v_head,
        "critical_time_index": base["critical_time_index"],
        "critical_time": (base["critical_s_head"] - start_s_head) / v_head,
        "critical_handle_index": base["critical_handle_index"],
        "critical_x": base["critical_x"],
        "critical_y": base["critical_y"],
        "result_p4": result_p4,
    }



def solve_problem5_fast(
    cfg: AppConfig,
    pitch: float,
    export_prefix: str = "problem5",
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print_banner("问题5（线性缩放快速解）")
    if base is None:
        base = solve_problem4_base(cfg, pitch)

    gain = base["gain_max"]
    if gain <= 0:
        raise RuntimeError("问题5：速度增益 gain_max 非正，无法计算最大允许龙头速度。")

    v_head_star = cfg.v_lim / gain
    final_result = build_problem5_final_result(base, v_head_star)
    df_bottleneck = analyze_speed_bottlenecks_fast(final_result, top_k=10)

    v_values = safe_linspace(cfg.v_left, cfg.v_right, cfg.scan_points)
    df_scan = pd.DataFrame({
        "v_head": v_values,
        "V_max": gain * v_values,
        "feasible": gain * v_values <= cfg.v_lim,
        "critical_time": np.where(v_values > 0, (base["critical_s_head"] - base["start_s_head"]) / v_values, np.nan),
        "critical_handle_index": base["critical_handle_index"],
        "critical_x": base["critical_x"],
        "critical_y": base["critical_y"],
    })

    v_lim_list = [0.9 * cfg.v_lim, 0.95 * cfg.v_lim, cfg.v_lim, 1.05 * cfg.v_lim, 1.10 * cfg.v_lim]
    df_sensitivity = pd.DataFrame({
        "v_lim": v_lim_list,
        "v_head_star": np.array(v_lim_list, dtype=float) / gain,
        "V_max_at_star": v_lim_list,
    })

    df_validation = pd.DataFrame({
        "v_head": [0.95 * v_head_star, v_head_star, 1.05 * v_head_star],
    })
    df_validation["V_max"] = gain * df_validation["v_head"]
    df_validation["feasible"] = df_validation["V_max"] <= cfg.v_lim

    unique_handles = df_bottleneck["handle_index"].drop_duplicates().tolist()[:5]

    if cfg.export_fig:
        legacy.plot_headspeed_vs_vmax(
            df_scan=df_scan,
            v_lim=cfg.v_lim,
            save_path=f"{export_prefix}_headspeed_vs_vmax.png",
        )
        legacy.plot_key_handle_speed_curves(
            final_result=final_result,
            handle_indices=unique_handles,
            save_path=f"{export_prefix}_key_handle_speeds.png",
        )
        legacy.plot_critical_speed_position(
            final_result=final_result,
            bench_width=cfg.bench_width,
            save_path=f"{export_prefix}_critical_speed_position.png",
        )
        legacy.plot_sensitivity_curve(
            df_sens=df_sensitivity,
            save_path=f"{export_prefix}_sensitivity.png",
        )

    if cfg.export_csv:
        df_scan.to_csv(f"{export_prefix}_scan.csv", index=False, encoding="utf-8-sig")
        df_bottleneck.to_csv(f"{export_prefix}_bottleneck.csv", index=False, encoding="utf-8-sig")
        df_sensitivity.to_csv(f"{export_prefix}_sensitivity.csv", index=False, encoding="utf-8-sig")
        df_validation.to_csv(f"{export_prefix}_validation.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([
            {
                "v_head_star": v_head_star,
                "gain_max": gain,
                "critical_time": final_result["critical_time"],
                "critical_handle_index": final_result["critical_handle_index"],
                "critical_x": final_result["critical_x"],
                "critical_y": final_result["critical_y"],
            }
        ]).to_csv(f"{export_prefix}_summary.csv", index=False, encoding="utf-8-sig")

    return {
        "v_head_star": v_head_star,
        "gain_max": gain,
        "scan_table": df_scan,
        "final_result": final_result,
        "bottleneck_table": df_bottleneck,
        "sensitivity_table": df_sensitivity,
        "validation_table": df_validation,
    }


# ============================================================
# Main flow
# ============================================================

def build_config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        num_handles=args.num_handles,
        bench_width=args.bench_width,
        dt=args.dt,
        dt_search=args.dt_search,
        turn_radius=args.turn_radius,
        pitch_left=args.pitch_left,
        pitch_right=args.pitch_right,
        pitch_tol=args.pitch_tol,
        pitch_scan_points=args.pitch_scan_points,
        pitch_p4=args.pitch_p4,
        v_head_p4=args.v_head_p4,
        v_lim=args.v_lim,
        v_left=args.v_left,
        v_right=args.v_right,
        v_tol=args.v_tol,
        scan_points=args.scan_points,
        R1=args.R1,
        R2=args.R2,
        ds_path=args.ds_path,
        ds_sample=args.ds_sample,
        center_dist_threshold=args.center_dist_threshold,
        export_csv=not args.no_csv,
        export_fig=not args.no_fig,
    )



def run_pipeline(cfg: AppConfig, stage: str) -> dict[str, Any]:
    out: dict[str, Any] = {}

    # Stage selection lets you avoid re-running everything.
    need_p1 = stage in {"p1", "p2", "all"}
    need_p2 = stage in {"p2", "all"}
    need_p3 = stage in {"p3", "all"}
    need_p4 = stage in {"p4", "all"}
    need_p5 = stage in {"p5", "all"}

    if need_p1:
        print_banner("问题1")
        out["p1"] = legacy.solve_problem1(
            t_start=cfg.t_start,
            t_end=cfg.t_end,
            dt=cfg.dt,
            theta0_init=cfg.theta_start,
            num_handles=cfg.num_handles,
            export_csv=cfg.export_csv,
        )

    if need_p2:
        if "p1" not in out:
            raise RuntimeError("问题2 依赖问题1的位置结果，请先运行 p1 或 all。")
        out["p2"] = solve_problem2_streaming(
            x_mat=out["p1"]["x_mat"],
            y_mat=out["p1"]["y_mat"],
            dt=cfg.dt,
            width=cfg.bench_width,
            center_dist_threshold=cfg.center_dist_threshold,
            export_csv=cfg.export_csv,
            export_fig=cfg.export_fig,
            export_prefix="problem2_fast",
        )

    if need_p3:
        out["p3"] = solve_problem3_fast(cfg, export_prefix="problem3_fast")

    if need_p4 or need_p5:
        if "p3" in out:
            pitch_for_turn = out["p3"]["pitch_star"]
        else:
            pitch_for_turn = cfg.pitch_p4

        base_p4 = solve_problem4_base(cfg, pitch_for_turn)
        out["p4_base"] = base_p4

        if need_p4:
            out["p4"] = solve_problem4_fast(
                cfg=cfg,
                pitch=pitch_for_turn,
                v_head=cfg.v_head_p4,
                export_prefix="problem4_fast",
                base=base_p4,
            )

        if need_p5:
            out["p5"] = solve_problem5_fast(
                cfg=cfg,
                pitch=pitch_for_turn,
                export_prefix="problem5_fast",
                base=base_p4,
            )

    return out



def print_summary(results: dict[str, Any]) -> None:
    print_banner("求解摘要")

    if "p2" in results:
        if results["p2"]["collision"]:
            print(f"[问题2] 首次碰撞时刻: {results['p2']['refined']['time']:.8f} s")
        else:
            print("[问题2] 给定时间区间内未检测到碰撞")

    if "p3" in results:
        print(f"[问题3] 最小可行螺距: {results['p3']['pitch_star']:.6f}")

    if "p4" in results:
        print(f"[问题4] 调头路径总长度: {results['p4']['metrics']['total_length']:.6f}")
        print(f"[问题4] 调头过程最大把手速度: {results['p4']['metrics']['max_speed']:.6f}")
        print(f"[问题4] 调头过程最小安全距离: {results['p4']['metrics']['min_clearance']:.6f}")
        print(f"[问题4] 是否发生碰撞: {results['p4']['metrics']['collision']}")

    if "p5" in results:
        print(f"[问题5] 最大允许龙头速度: {results['p5']['v_head_star']:.6f}")
        print(f"[问题5] 单位速度增益 gain_max: {results['p5']['gain_max']:.6f}")
        print(f"[问题5] 临界把手编号: {results['p5']['final_result']['critical_handle_index']}")
        print(f"[问题5] 临界时刻: {results['p5']['final_result']['critical_time']:.6f} s")



def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refactored board-dragon solver with faster stage orchestration."
    )
    parser.add_argument(
        "--stage",
        choices=["p1", "p2", "p3", "p4", "p5", "all"],
        default="all",
        help="Only run the required stage instead of always running everything.",
    )

    parser.add_argument("--num-handles", type=int, default=224)
    parser.add_argument("--bench-width", type=float, default=0.30)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dt-search", type=float, default=0.05)
    parser.add_argument("--turn-radius", type=float, default=4.5)
    parser.add_argument("--pitch-left", type=float, default=0.20)
    parser.add_argument("--pitch-right", type=float, default=0.80)
    parser.add_argument("--pitch-tol", type=float, default=1e-3)
    parser.add_argument("--pitch-scan-points", type=int, default=13)
    parser.add_argument("--pitch-p4", type=float, default=0.55)
    parser.add_argument("--v-head-p4", type=float, default=1.0)
    parser.add_argument("--v-lim", type=float, default=2.0)
    parser.add_argument("--v-left", type=float, default=0.10)
    parser.add_argument("--v-right", type=float, default=5.0)
    parser.add_argument("--v-tol", type=float, default=1e-3)
    parser.add_argument("--scan-points", type=int, default=12)
    parser.add_argument("--R1", type=float, default=1.8)
    parser.add_argument("--R2", type=float, default=1.2)
    parser.add_argument("--ds-path", type=float, default=0.01)
    parser.add_argument("--ds-sample", type=float, default=0.01)
    parser.add_argument("--center-dist-threshold", type=float, default=1.5)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-fig", action="store_true")
    return parser



def main() -> None:
    args = make_parser().parse_args()
    cfg = build_config_from_args(args)
    results = run_pipeline(cfg, stage=args.stage)
    print_summary(results)


if __name__ == "__main__":
    main()
