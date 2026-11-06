"""
Lab 3 - Bài 1: Xây dựng mô hình Hồi quy tuyến tính (Linear Regression)
Yêu cầu:
- Tạo dữ liệu giả lập có mối quan hệ tuyến tính: y = 2X + 1 + noise.
- Sử dụng sklearn.linear_model.LinearRegression để huấn luyện mô hình.
- In ra các hệ số của mô hình (coefficient - độ dốc w và intercept - hệ số chặn b).
"""

import numpy as np
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 65)
    print("LAB 3 - BÀI 1: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH CƠ BẢN")
    print("=" * 65)

    # 1. Tạo dữ liệu giả lập với mối quan hệ y = 2X + 1 + noise
    np.random.seed(42)
    n_samples = 100
    # X ngẫu nhiên trong khoảng [0, 10]
    X = np.random.uniform(0, 10, size=(n_samples, 1))
    
    # Nhiễu trắng phân phối chuẩn Gaussian Mean = 0, Std = 1.2
    noise = np.random.normal(0, 1.2, size=(n_samples, 1))
    
    # Công thức lý thuyết thực tế: y = 2*X + 1 + noise
    true_w = 2.0
    true_b = 1.0
    y = true_w * X + true_b + noise

    print("\n1. Thông tin dữ liệu giả lập:")
    print(f"   - Số lượng mẫu: {n_samples}")
    print(f"   - Hàm số gốc: y = {true_w} * X + {true_b} + noise")
    print(f"   - 5 mẫu đầu tiên (X, y):")
    for i in range(5):
        print(f"     Mẫu {i+1}: X = {X[i, 0]:.4f} -> y = {y[i, 0]:.4f}")

    # 2. Khởi tạo và huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # 3. Trích xuất hệ số góc và hệ số chặn
    learned_w = model.coef_[0, 0]
    learned_b = model.intercept_[0]

    print("\n2. Kết quả huấn luyện mô hình LinearRegression:")
    print(f"   - Hệ số góc đã học (Coefficient - w) : {learned_w:.4f} (Lý thuyết: {true_w})")
    print(f"   - Hệ số chặn đã học (Intercept - b)   : {learned_b:.4f} (Lý thuyết: {true_b})")
    print(f"   - Phương trình hồi quy ước lượng      : y_hat = {learned_w:.4f} * X + {learned_b:.4f}")

    # 4. Đánh giá độ sai lệch so với giá trị thực tế ban đầu
    error_w = abs(learned_w - true_w)
    error_b = abs(learned_b - true_b)
    print(f"\n3. Đánh giá độ lệch tham số:")
    print(f"   - Độ lệch của w: {error_w:.4f}")
    print(f"   - Độ lệch của b: {error_b:.4f}")
    print("   -> Mô hình đã khôi phục lại các tham số gần như hoàn hảo từ dữ liệu có nhiễu.")
    print("=" * 65)

if __name__ == "__main__":
    main()

