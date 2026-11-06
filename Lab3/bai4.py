"""
Lab 3 - Bài 4: Quy trình ML hoàn chỉnh - Phần 3: Thử nghiệm mô hình Regression
Yêu cầu:
- Áp dụng Hồi quy tuyến tính vào bài toán hồi quy đa biến thực tế.
- Sử dụng tập dữ liệu có sẵn trong sklearn (California Housing hoặc Diabetes toàn diện).
- Thực hiện đầy đủ các bước:
  1. Load dữ liệu
  2. Preprocess (Chuẩn hóa Scaling)
  3. Phân chia Train/Test
  4. Huấn luyện mô hình Linear Regression
  5. Đánh giá (MSE, RMSE, MAE, R2 Score)
  6. Visualize so sánh giữa Giá trị thực tế và Giá trị dự đoán (Actual vs Predicted)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_regression_dataset():
    """Tải dữ liệu hồi quy: thử California Housing, nếu không có mạng/quyền thì dùng Diabetes."""
    try:
        # Thử tải california housing với thư mục tạm
        import tempfile
        temp_dir = os.path.join(tempfile.gettempdir(), 'sklearn_cache')
        data = fetch_california_housing(data_home=temp_dir)
        name = "California Housing"
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_name = "Giá nhà trung bình ($100k)"
    except Exception:
        data = load_diabetes()
        name = "Diabetes (Toàn bộ 10 đặc trưng lâm sàng)"
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_name = "Mức độ tiến triển bệnh"
    return X, y, name, feature_names, target_name

def main():
    print("=" * 70)
    print("LAB 3 - BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - PHẦN 3")
    print("MÔ HÌNH HỒI QUY ĐA BIẾN (MULTIPLE LINEAR REGRESSION)")
    print("=" * 70)

    # Bước 1: Load dữ liệu
    X, y, dataset_name, feature_names, target_name = load_regression_dataset()
    print(f"\n[Bước 1] Tải dữ liệu thành công: {dataset_name}")
    print(f"   - Kích thước tập dữ liệu X: {X.shape} (Số lượng mẫu: {X.shape[0]}, Số đặc trưng: {X.shape[1]})")
    print(f"   - Tên các đặc trưng: {feature_names[:6]}...")

    # Bước 2 & 3: Phân chia Train/Test và Chuẩn hóa dữ liệu qua Pipeline
    print("\n[Bước 2 & 3] Phân chia Train/Test (80/20) và Tiền xử lý với StandardScaler...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Bước 4: Huấn luyện mô hình (Train)
    print("\n[Bước 4] Huấn luyện mô hình...")
    pipeline.fit(X_train, y_train)
    print("   -> Mô hình hồi quy đã được huấn luyện thành công!")

    # Bước 5: Đánh giá mô hình (Evaluate)
    print("\n[Bước 5] Đánh giá kết quả trên tập Test:")
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   - MAE  : {mae:.4f}")
    print(f"   - MSE  : {mse:.4f}")
    print(f"   - RMSE : {rmse:.4f}")
    print(f"   - R2   : {r2:.4f}")

    # Bước 6: Trực quan hóa kết quả so sánh Thực tế vs Dự đoán (Actual vs Predicted)
    print("\n[Bước 6] Trực quan hóa kết quả...")
    save_path = os.path.join(os.path.dirname(__file__), "actual_vs_predicted.png")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='dodgerblue', edgecolors='k', alpha=0.6, label='Các mẫu kiểm thử')
    
    # Vẽ đường chéo lý tưởng 45 độ (y_test == y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Đường hoàn hảo (Lý tưởng y = y_hat)')
    
    plt.title(f"So sánh Giá trị thực tế và Dự đoán ({dataset_name})", fontsize=13, fontweight='bold')
    plt.xlabel(f"Giá trị thực tế ({target_name})", fontsize=11)
    plt.ylabel(f"Giá trị dự đoán ({target_name})", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"   -> Đã lưu biểu đồ Actual vs Predicted tại: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

