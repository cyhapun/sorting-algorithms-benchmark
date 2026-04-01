import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, NullFormatter
from matplotlib.colors import LogNorm
import statistics
import os
import json

plt.rcParams["font.family"] = "Times New Roman"


def plot_execution_times(
    results_dict,
    sizes,
    datasets,
    algo_name="Thuật toán",
    use_log_y=False,
    annotate_last_point=True,
):
    """
    Hàm vẽ đồ thị trực quan hóa thời gian thực thi của thuật toán.
    Phiên bản nâng cao với phong cách hiện đại, dễ đọc và dễ so sánh.
    Tích hợp thuật toán chống đè nhãn (Label Collision Avoidance) - Vẽ đường nối cho TẤT CẢ.

    Tham số (Inputs):
    -----------------
    results_dict : dict
        Từ điển chứa kết quả thời gian chạy.
        Cấu trúc: {'tên_dataset': [thời_gian_n1, thời_gian_n2, ...]}
    sizes : list
        Danh sách các kích thước mảng n (ví dụ: [100, 1000, 10000, 100000]).
    datasets : list
        Danh sách các loại tập dữ liệu (ví dụ: ['random', 'nearly_sorted', ...]).
    algo_name : str, optional
        Tên thuật toán và độ phức tạp để hiển thị trên tiêu đề đồ thị.
        (Mặc định là "Thuật toán").
    use_log_y : bool, optional
        Nếu True, dùng trục y log để so sánh khi thời gian chênh lệch lớn.
    annotate_last_point : bool, optional
        Nếu True, ghi nhãn trực tiếp tại điểm cuối mỗi đường.
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    fig.patch.set_facecolor("#F7F8FB")
    ax.set_facecolor("#F7F8FB")

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    markers = ["o", "s", "D", "^", "P", "X"]
    labels = {
        "random": "Ngẫu nhiên",
        "nearly_sorted": "Gần như đã sắp",
        "many_duplicates": "Nhiều khóa trùng",
        "reverse_sorted": "Sắp ngược",
    }

    plotted_series = []

    for idx, data_type in enumerate(datasets):
        times = results_dict[data_type]

        valid_sizes = [sizes[i] for i in range(len(sizes)) if times[i] is not None]
        valid_times = [times[i] for i in range(len(times)) if times[i] is not None]

        if valid_sizes:
            x_vals = np.array(valid_sizes, dtype=float)
            y_vals = np.array(valid_times, dtype=float)
            display_label = labels.get(data_type, data_type)

            ax.plot(
                x_vals,
                y_vals,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linewidth=2.8,
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.8,
                label=display_label,
                zorder=3,
            )

            ax.scatter(
                [x_vals[-1]],
                [y_vals[-1]],
                color=colors[idx % len(colors)],
                s=70,
                zorder=4,
            )

            plotted_series.append(
                (display_label, x_vals, y_vals, colors[idx % len(colors)])
            )

    ax.set_xscale("log", base=10)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{int(s):,}".replace(",", ".") for s in sizes])

    if use_log_y:
        positive_values = [
            y for _, _, y_vals, _ in plotted_series for y in y_vals.tolist() if y > 0
        ]
        if positive_values:
            ax.set_yscale("log")

    ax.set_title(
        f"Thời gian thực thi của thuật toán {algo_name}",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Kích thước tập dữ liệu (n)", fontsize=13, labelpad=10)
    ax.set_ylabel("Thời gian (s) - Log scale", fontsize=13, labelpad=10)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if annotate_last_point:
        x_min, x_max = min(sizes), max(sizes)
        x_text = x_max * 1.35

        sorted_series = sorted(plotted_series, key=lambda s: s[2][-1])

        last_text_y = 1e-10

        for display_label, x_vals, y_vals, color in sorted_series:
            actual_y = y_vals[-1]

            if use_log_y:
                text_y = max(actual_y, last_text_y * 1.5)
            else:
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                text_y = max(actual_y, last_text_y + y_range * 0.05)

            ax.text(
                x_text,
                text_y,
                f"{actual_y:.4f}s",
                color=color,
                fontsize=10.5,
                va="center",
                fontweight="semibold",
            )

            ax.plot(
                [x_max * 1.05, x_text * 0.95],
                [actual_y, text_y],
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.6,
            )

            last_text_y = text_y

        ax.set_xlim(left=x_min * 0.95, right=x_max * 3.0)

    ax.legend(
        loc="upper left",
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="#D9DDE5",
        framealpha=0.95,
    )

    plt.tight_layout(pad=2.0)
    plt.show()


def measure_algorithm_time(algorithm_func, data_dict, datasets, sizes, num_runs=5):
    """
    Hàm đo thời gian thực thi trung bình của một thuật toán qua nhiều lần chạy.

    Inputs:
    - algorithm_func: Tên hàm của thuật toán (VD: merge_sort, insertion_sort)
    - data_dict: Từ điển chứa dữ liệu mảng đã được nạp sẵn
    - datasets: Danh sách các loại tập dữ liệu
    - sizes: Danh sách các kích thước n
    - num_runs: Số lần chạy để lấy trung bình (mặc định = 5)

    Output:
    - results: Từ điển chứa thời gian thực thi trung bình.
    """
    results = {data_type: [] for data_type in datasets}

    for data_type in datasets:
        for size in sizes:
            data = data_dict[data_type].get(size)

            if data is not None:
                algorithm_func(data.copy())
                run_times = []

                for _ in range(num_runs):
                    arr_to_sort = data.copy()

                    start_time = time.perf_counter()
                    algorithm_func(arr_to_sort)
                    end_time = time.perf_counter()

                    run_times.append(end_time - start_time)

                median_time = statistics.median(run_times)
                results[data_type].append(median_time)
            else:
                results[data_type].append(None)

    return results


def plot_comprehensive_barchart_grid(
    all_results,
    sizes,
    datasets,
    title="Tổng quan Hiệu năng Thuật toán Lọc trên các Phân bố Dữ liệu",
):
    """
    Vẽ lưới 2x2 Subplots (Small Multiples).
    Mỗi subplot là một Bar Chart so sánh các thuật toán theo Size trên 1 loại dataset.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor("#F7F8FB")

    axes = axes.flatten()

    algorithms = list(all_results.keys())
    num_algos = len(algorithms)
    num_sizes = len(sizes)

    x = np.arange(num_sizes)
    width = 0.8 / num_algos
    colors = ["#D55E00", "#0072B2", "#009E73", "#E69F00"]

    labels_vn = {
        "random": "Ngẫu nhiên",
        "nearly_sorted": "Gần như đã sắp",
        "many_duplicates": "Nhiều khóa trùng",
        "reverse_sorted": "Sắp ngược",
    }

    for ax_idx, target_dataset in enumerate(datasets):
        ax = axes[ax_idx]
        ax.set_facecolor("#F7F8FB")
        dataset_name = labels_vn.get(target_dataset, target_dataset)

        for i, algo_name in enumerate(algorithms):
            times = all_results[algo_name][target_dataset]
            safe_times = [t if (t is not None and t > 0) else 1e-6 for t in times]
            offset = (i - num_algos / 2 + 0.5) * width

            bars = ax.bar(
                x + offset,
                safe_times,
                width,
                label=algo_name if ax_idx == 0 else "",
                color=colors[i % len(colors)],
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )

            for bar in bars:
                yval = bar.get_height()
                if yval > 1e-5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval * 1.25,
                        f"{yval:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                        color="#333333",
                        fontweight="bold",
                    )

        ax.set_yscale("log")

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=ymin, top=ymax * 4)

        ax.set_title(
            f"Tập dữ liệu: {dataset_name}",
            fontsize=16,
            fontweight="bold",
            color="#2C3E50",
            pad=30,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{s:,}".replace(",", ".") for s in sizes], fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Kích thước tập dữ liệu (n)", fontsize=13)

        if ax_idx % 2 == 0:
            ax.set_ylabel("Thời gian (s) - Log scale", fontsize=13)

        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=24, fontweight="bold", y=0.98, color="#1A252F")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=num_algos,
        fontsize=14,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.88])
    plt.show()


def plot_danger_zone_heatmap(
    all_results,
    target_size_idx,
    sizes,
    datasets,
    title="Bản đồ Nhiệt: Vùng Nguy Hiểm Tại n = ",
):
    """
    Vẽ Heatmap so sánh toàn bộ thuật toán trên mọi dataset ở một kích thước n cụ thể.
    """
    algorithms = list(all_results.keys())
    target_n = sizes[target_size_idx]

    labels_vn = {
        "random": "Ngẫu nhiên",
        "nearly_sorted": "Gần sắp",
        "many_duplicates": "Nhiều trùng",
        "reverse_sorted": "Sắp ngược",
    }
    y_labels = [labels_vn.get(ds, ds) for ds in datasets]

    data_matrix = np.zeros((len(datasets), len(algorithms)))

    for i, ds in enumerate(datasets):
        for j, algo in enumerate(algorithms):
            val = all_results[algo][ds][target_size_idx]
            data_matrix[i, j] = val if (val is not None and val > 0) else 1e-6

    plt.figure(figsize=(10, 6), dpi=140)

    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        norm=LogNorm(vmin=data_matrix.min(), vmax=data_matrix.max()),
        xticklabels=algorithms,
        yticklabels=y_labels,
        linewidths=1.5,
        linecolor="white",
        cbar_kws={"label": "Thời gian (giây) - Log scale"},
    )

    plt.title(
        f"{title} {target_n:,}".replace(",", "."),
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xticks(rotation=15, ha="right", fontsize=10, fontweight="bold")
    plt.yticks(rotation=0, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_algorithm_speedup_grid(
    all_results,
    sizes,
    datasets,
    baseline_algo_name,
    title="Bảng Tỷ lệ Tăng tốc so với Selection Sort",
):
    """
    Vẽ lưới 2x2 Subplots so sánh Speedup Factor so với Baseline tồi nhất (ví dụ Selection Sort).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    axes = axes.flatten()

    algorithms = list(all_results.keys())
    algos_to_compare = [a for a in algorithms if a != baseline_algo_name]

    num_compare = len(algos_to_compare)
    x = np.arange(len(sizes))
    width = 0.8 / num_compare

    algo_colors = ["#E69F00", "#009E73", "#0072B2", "#CC79A7"]
    labels_vn = {
        "random": "Ngẫu nhiên",
        "nearly_sorted": "Gần như đã sắp",
        "many_duplicates": "Nhiều khóa trùng",
        "reverse_sorted": "Sắp ngược",
    }

    for ax_idx, target_dataset in enumerate(datasets):
        ax = axes[ax_idx]
        ax.set_facecolor("#FFFFFF")
        dataset_name = labels_vn.get(target_dataset, target_dataset)

        baseline_times = all_results[baseline_algo_name][target_dataset]
        baseline_times = [
            t if (t is not None and t > 0) else 1e-7 for t in baseline_times
        ]

        for i, algo_name in enumerate(algos_to_compare):
            times = all_results[algo_name][target_dataset]
            safe_times = [t if (t is not None and t > 0) else 1e-7 for t in times]

            # Tính toán Speedup: Baseline_Time / Algo_Time
            speedup_factors = [
                baseline / algo for baseline, algo in zip(baseline_times, safe_times)
            ]

            offset = (i - num_compare / 2 + 0.5) * width

            bars = ax.bar(
                x + offset,
                speedup_factors,
                width,
                label=algo_name if ax_idx == 0 else "",
                color=algo_colors[i % len(algo_colors)],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
                alpha=0.9,
            )

            # Ghi nhãn Speedup Factor trên từng cột (LOẠI BỎ CHỮ 'E')
            for bar in bars:
                yval = bar.get_height()
                if yval > 2:  # Chỉ ghi số nếu nhanh hơn ít nhất 2 lần
                    # Nếu >= 1000 thì hiển thị kiểu 10.000x, ngược lại hiển thị kiểu 15.5x
                    if yval >= 1000:
                        label_text = f"{int(yval):,}x".replace(",", ".")
                    else:
                        label_text = f"{yval:.1f}x"

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        yval * 1.15,
                        label_text,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        rotation=90,
                        color="#2C3E50",
                        fontweight="bold",
                    )

        ax.set_yscale("log")

        def y_axis_formatter(y, _):
            if y >= 1000:
                return f"{int(y):,}x".replace(",", ".")
            elif y >= 1:
                return f"{int(y)}x"
            else:
                return f"{y:g}x"

        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        ax.set_title(
            f"Tập dữ liệu: {dataset_name}", fontsize=15, fontweight="bold", pad=15
        )
        ax.grid(
            True, which="major", axis="y", linestyle="-", linewidth=0.8, color="#EAEAEA"
        )

        ax.axhline(1, color="#BDC3C7", linestyle="--", linewidth=1.5, zorder=2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{s:,}".replace(",", ".") for s in sizes], fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Kích thước tập dữ liệu (n)", fontsize=12)

        if ax_idx % 2 == 0:
            ax.set_ylabel("Hệ số tăng tốc - Log scale", fontsize=12)

    fig.suptitle(f"{title}", fontsize=22, fontweight="heavy", y=0.96, color="#1A252F")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=num_compare,
        fontsize=13,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.show()


def calculate_baseline_n_log_n(all_results, sizes):
    """
    Hàm tính toán đường cơ sở (Baseline) cho nhóm thuật toán O(n log n).
    Inputs:
    - all_results: Dictionary chứa kết quả chạy của cả Merge Sort và Quick Sort.
    - sizes: Danh sách các kích thước n (VD: [100, 1000, 10000, 100000]).

    Output:
    - Danh sách các giá trị Baseline tương ứng với từng kích thước n.
    """
    baselines = []

    for i in range(len(sizes)):
        valid_times = []

        if "Merge Sort" in all_results:
            valid_times.append(all_results["Merge Sort"]["random"][i])
            valid_times.append(all_results["Merge Sort"]["nearly_sorted"][i])
            valid_times.append(all_results["Merge Sort"]["many_duplicates"][i])
            valid_times.append(all_results["Merge Sort"]["reverse_sorted"][i])

        if "Quick Sort" in all_results:
            valid_times.append(all_results["Quick Sort"]["random"][i])
            valid_times.append(all_results["Quick Sort"]["nearly_sorted"][i])
            valid_times.append(all_results["Quick Sort"]["reverse_sorted"][i])

        valid_times = [t for t in valid_times if t is not None]

        if valid_times:
            baseline_value = sum(valid_times) / len(valid_times)
            baselines.append(baseline_value)
        else:
            baselines.append(None)

    return baselines


def plot_baseline_n_log_n(
    sizes, baseline_results, title="Đường Cơ Sở Đại Diện Cho Nhóm O(n log n)"
):
    """
    Hàm trực quan hóa đường Baseline phong cách tối giản (Minimalist).
    """
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(
        sizes,
        baseline_results,
        marker="o",
        linestyle="-",
        color="#2E5A88",
        linewidth=3,
        markersize=10,
        markerfacecolor="white",
        markeredgewidth=2.5,
        label=r"Baseline $O(n \log n)$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    for i in range(len(sizes)):
        if baseline_results[i] is not None:
            ax.text(
                sizes[i],
                baseline_results[i] * 1.3,
                f"{baseline_results[i]:.5f}s",
                fontsize=10,
                fontweight="bold",
                ha="center",
                color="#333333",
            )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=25, color="#222222")
    ax.set_xlabel(
        "Kích thước tập dữ liệu (n) - Log scale",
        fontsize=12,
        fontweight="600",
        labelpad=12,
    )
    ax.set_ylabel(
        "Thời gian (s) - Log scale", fontsize=12, fontweight="600", labelpad=12
    )

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}".replace(",", ".") for n in sizes])

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Làm đậm đường biên trục X và Y
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.legend(loc="upper left", fontsize=11, frameon=False)

    plt.tight_layout()
    plt.show()

def plot_multiple_baselines(sizes, baselines_dict, title="So Sánh Các Đường Cơ Sở"):
    """
    Hàm kết hợp nhiều đường Baseline vào một biểu đồ duy nhất.
    baselines_dict: { 'Tên nhãn': [danh_sách_thời_gian], ... }
    """
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    colors = ['#2E5A88', '#D9534F', '#5CB85C', '#F0AD4E', '#5BC0DE']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (label, results) in enumerate(baselines_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(sizes, results, 
                marker=markers[i % len(markers)], 
                linestyle='-', 
                color=color,
                linewidth=2.5, 
                markersize=8, 
                markerfacecolor='white',
                markeredgewidth=2,
                label=label)

        # Hiển thị giá trị tại điểm cuối
        ax.text(sizes[-1], results[-1] * 1.15, 
                f"{results[-1]:.4f}s", 
                fontsize=9, fontweight='bold', ha='center', color=color)

    # --- PHẦN CHỈNH SỬA ĐỂ XÓA VIỀN ---
    for spine in ax.spines.values():
        spine.set_visible(False)  # Ẩn toàn bộ 4 đường viền (trên, dưới, trái, phải)

    ax.tick_params(axis='both', which='both', length=0) # Ẩn các dấu gạch nhỏ trên trục
    # ---------------------------------

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=30, color='#222222')
    ax.set_xlabel('Kích thước tập dữ liệu (n)', fontsize=11, color='#666666', labelpad=10)
    ax.set_ylabel('Thời gian thực thi (s)', fontsize=11, color='#666666', labelpad=10)

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}".replace(',', '.') for n in sizes])

    ax.legend(loc='upper left', fontsize=10, frameon=False)

    plt.tight_layout()
    plt.show()

def calculate_baseline_n2(all_results, sizes, datasets):
    """
    Hàm tính toán đường cơ sở (Baseline) cho nhóm thuật toán O(n^2).
    Inputs:
    - all_results: Dictionary chứa kết quả chạy của các thuật toán O(n^2).
    - sizes: Danh sách các kích thước n (VD: [100, 1000, 10000, 100000]).
    - datasets: Danh sách các loại tập dữ liệu.

    Output:
    - Danh sách các giá trị Baseline tương ứng với từng kích thước n.
    """
    baselines = []

    for i in range(len(sizes)):
        valid_times = []
        for algo_name in all_results:
            for data_type in datasets:
                t = all_results[algo_name][data_type][i]
                if t is not None:
                    valid_times.append(t)

        if valid_times:
            baseline_value = sum(valid_times) / len(valid_times)
            baselines.append(baseline_value)
        else:
            baselines.append(None)

    return baselines


def plot_baseline_n2(
    sizes, baseline_results, title="Đường Cơ Sở Đại Diện Cho Nhóm $O(n^2)$"
):
    """
    Hàm trực quan hóa đường Baseline cho nhóm O(n^2).
    """
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(
        sizes,
        baseline_results,
        marker="o",
        linestyle="-",
        color="#2E5A88",
        linewidth=3,
        markersize=10,
        markerfacecolor="white",
        markeredgewidth=2.5,
        label=r"Baseline $O(n^2)$",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    for i in range(len(sizes)):
        if baseline_results[i] is not None:
            ax.text(
                sizes[i],
                baseline_results[i] * 1.3,
                f"{baseline_results[i]:.5f}s",
                fontsize=10,
                fontweight="bold",
                ha="center",
                color="#333333",
            )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=25, color="#222222")
    ax.set_xlabel(
        "Kích thước tập dữ liệu (n) - Log scale",
        fontsize=12,
        fontweight="600",
        labelpad=12,
    )
    ax.set_ylabel(
        "Thời gian (s) - Log scale", fontsize=12, fontweight="600", labelpad=12
    )

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}".replace(",", ".") for n in sizes])

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.legend(loc="upper left", fontsize=11, frameon=False)

    plt.tight_layout()
    plt.show()

def load_data_from_folder(folder_path):
    sizes = [100, 1000, 10000, 100000] 
    data = {}
    
    # Duyệt qua tất cả các file trong folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                # Đọc dữ liệu từ file JSON
                content = json.load(f)
                
                # Tạo nhãn (label) đẹp từ tên file
                # Ví dụ: merge_sort.json -> Merge Sort
                label = filename.replace('.json', '').replace('_', ' ').title()
                
                # Xử lý các trường hợp đặc biệt cho Baseline để có ký hiệu Toán học
                if 'Baseline O N Logn' in label: label = r'Baseline $O(n \log n)$'
                elif 'Baseline O N2' in label: label = r'Baseline $O(n^2)$'
                elif 'Baseline O N' in label: label = r'Baseline $O(n)$'
                
                # Giả sử trong file JSON là một list thời gian, hoặc dict có key 'times'
                if isinstance(content, list):
                    data[label] = content
                elif isinstance(content, dict) and 'times' in content:
                    data[label] = content['times']
                    
    return sizes, data