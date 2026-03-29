import matplotlib.pyplot as plt
import time

def plot_execution_times(results_dict, sizes, datasets, algo_name="Thuật toán"):
    """
    Hàm vẽ đồ thị trực quan hóa thời gian thực thi của thuật toán.
    
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
    """
    
    # Thiết lập kích thước đồ thị
    plt.figure(figsize=(10, 6))

    # Định nghĩa màu cho từng loại dữ liệu
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels = {
        'random': 'Ngẫu nhiên',
        'nearly_sorted': 'Gần như đã sắp',
        'many_duplicates': 'Nhiều khóa trùng',
        'reverse_sorted': 'Sắp ngược'
    }

    # Vẽ đường cho từng loại tập dữ liệu
    for idx, data_type in enumerate(datasets):
        times = results_dict[data_type]
        
        # Lọc các giá trị None (nếu có lỗi khi đọc file)
        valid_sizes = [sizes[i] for i in range(len(sizes)) if times[i] is not None]
        valid_times = [times[i] for i in range(len(times)) if times[i] is not None]
        
        if valid_sizes: # Chỉ vẽ nếu có dữ liệu
            plt.plot(valid_sizes, valid_times, color=colors[idx % len(colors)], 
                     linewidth=2, markersize=8, label=labels.get(data_type, data_type))

    # Trang trí chi tiết cho đồ thị
    plt.title(f'Thời gian thực thi của thuật toán {algo_name}', fontsize=15, fontweight='bold')
    plt.xlabel('Kích thước tập dữ liệu n', fontsize=12)
    plt.ylabel('Thời gian thực thi tính bằng giây', fontsize=12)

    # Thêm lưới để dễ dóng các giá trị
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)

    # Hiển thị đồ thị
    plt.tight_layout()
    plt.show()

def measure_algorithm_time(algorithm_func, data_dict, datasets, sizes):
    """
    Hàm đo thời gian thực thi của một thuật toán sắp xếp.
    
    Inputs:
    - algorithm_func: Tên hàm của thuật toán (VD: merge_sort, insertion_sort)
    - data_dict: Từ điển chứa dữ liệu mảng đã được nạp sẵn (loaded_data)
    - datasets: Danh sách các loại tập dữ liệu
    - sizes: Danh sách các kích thước n
    
    Output:
    - results: Từ điển chứa kết quả thời gian thực thi theo từng loại dataset.
    """
    results = {data_type: [] for data_type in datasets}
    
    for data_type in datasets:
        for size in sizes:
            # Lấy mảng dữ liệu tương ứng từ RAM
            data = data_dict[data_type].get(size)
            
            if data is not None:
                arr_to_sort = data.copy()
                
                # Bắt đầu đo
                start_time = time.perf_counter()
                algorithm_func(arr_to_sort)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                results[data_type].append(execution_time)
            else:
                results[data_type].append(None)
                
    return results