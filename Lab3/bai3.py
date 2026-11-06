"""
Lab 3 - Bài 3: Trực quan hóa kết quả Hồi quy tuyến tính
Yêu cầu:
- Sử dụng thư viện matplotlib để vẽ biểu đồ tán xạ (scatter plot) của dữ liệu thực tế.
- Vẽ đường thẳng hồi quy (đường dự đoán) của mô hình lên cùng một biểu đồ để so sánh.
- Lưu biểu đồ ra file hình ảnh PNG.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    print("=" * 65)
    print("LAB 3 - BÀI 3: TRỰC QUAN HÓA KẾT QUẢ HỒI QUY TUYẾN TÍNH")
    print("=" * 65)

    # 1. Tải dữ liệu và lấy 1 đặc trưng BMI
    diabetes = load_diabetes()
    X = diabetes.data[:, np.newaxis, 2] # Cột BMI
    y = diabetes.target

    # 2. Phân chia Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Tạo đường hồi quy dự đoán
    # Tạo các điểm dàn đều từ min(X) đến max(X) để vẽ đường thẳng mượt mà
    X_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_line = model.predict(X_line)

    # 5. Vẽ biểu đồ tán xạ và đường thẳng hồi quy
    plt.figure(figsize=(10, 6))

    # Điểm dữ liệu huấn luyện (Train)
    plt.scatter(X_train, y_train, color='royalblue', alpha=0.5, label='Dữ liệu huấn luyện (Train)')
    # Điểm dữ liệu kiểm thử (Test)
    plt.scatter(X_test, y_test, color='darkorange', alpha=0.8, edgecolors='black', label='Dữ liệu kiểm thử (Test)')
    # Đường thẳng hồi quy
    plt.plot(X_line, y_line, color='crimson', linewidth=2.5,
             label=f'Đường hồi quy: y = {model.coef_[0]:.2f}X + {model.intercept_:.2f}')

    plt.title("Hồi quy tuyến tính: Dự đoán tiến triển bệnh theo chỉ số BMI (Diabetes Dataset)", fontsize=13, fontweight='bold')
    plt.xlabel("Chỉ số khối cơ thể chuẩn hóa (BMI)", fontsize=11)
    plt.ylabel("Tiến triển bệnh tiểu đường (Target)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='upper left')

    save_path = os.path.join(os.path.dirname(__file__), "linear_regression_plot.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n1. Đã vẽ biểu đồ tán xạ kết hợp đường hồi quy thành công!")
    print(f"   Đường dẫn lưu file: {save_path}")
    print(f"   - Hệ số góc (Slope w)   : {model.coef_[0]:.4f}")
    print(f"   - Hệ số chặn (Intercept b): {model.intercept_:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()

