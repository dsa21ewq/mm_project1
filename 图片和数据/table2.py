"""根据问题1生成的 CSV（task3_positions_1s.csv），输出 LaTeX 格式表格。

生成内容类似：
  表 2 部分把手坐标随时间变化情况

输出到文件：table_problem1.tex

使用方法：
    python make_table_problem1.py
"""

import os

import pandas as pd


def format_num(x: float) -> str:
    return f"{x:.6f}"


def build_latex_table(df: pd.DataFrame,
                      times: list[int],
                      handle_specs: list[tuple[str, str, str]],
                      out_path: str = "table_problem1.tex") -> None:
    """构造 LaTeX 表格并写入文件。

    handle_specs: [(label, x_col, y_col), ...]
    """

    headers = ["", *[f"{t} s" for t in times]]

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{部分把手坐标随时间变化情况}")
    lines.append("\\label{tab:handle_positions}")
    lines.append(
        "\\begin{tabular}{l" + "r" * len(times) + "}"  # 第一列左对齐，其余右对齐
    )
    lines.append("\\toprule")

    # 表头
    header_line = " & ".join(headers) + " \\"
    lines.append(header_line)
    lines.append("\\midrule")

    # 输出每组（x,y）
    for label, x_col, y_col in handle_specs:
        # x 行
        row_x = [f"{label} x (m)"]
        # y 行
        row_y = [f"{label} y (m)"]

        for t in times:
            row = df[df["t"] == t]
            if row.empty:
                row_x.append("")
                row_y.append("")
                continue
            row_x.append(format_num(row.iloc[0][x_col]))
            row_y.append(format_num(row.iloc[0][y_col]))

        lines.append(" & ".join(row_x) + " \\\n")
        lines.append(" & ".join(row_y) + " \\"
)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已生成 LaTeX 表格文件: {out_path}")


def main():
    csv_path = os.path.join(os.getcwd(), "task3_positions_1s.csv")
    df = pd.read_csv(csv_path)

    times = [0, 60, 120, 180, 240, 300]

    # 参考题目截图，选取：
    #  - 龙头（front）
    #  - 第1节龙身（handle_1）
    #  - 第51节、101节、151节、201节（handle_51/101/151/201）
    #  - 龙尾前、后（tail_front/tail_back）
    handle_specs = [
        ("龙头", "head_front_x", "head_front_y"),
        ("第 1 节龙身", "handle_1_x", "handle_1_y"),
        ("第 51 节龙身", "handle_51_x", "handle_51_y"),
        ("第 101 节龙身", "handle_101_x", "handle_101_y"),
        ("第 151 节龙身", "handle_151_x", "handle_151_y"),
        ("第 201 节龙身", "handle_201_x", "handle_201_y"),
        ("龙尾（前）", "tail_front_x", "tail_front_y"),
        ("龙尾（后）", "tail_back_x", "tail_back_y"),
    ]

    build_latex_table(df, times, handle_specs, out_path="table_problem1.tex")


if __name__ == "__main__":
    main()
