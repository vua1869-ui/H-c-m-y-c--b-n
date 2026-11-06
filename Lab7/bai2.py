"""
Lab 7 - Bài 2: Xây dựng Pipeline hoàn chỉnh xử lý Missing Values và Scaling
Yêu cầu:
- Tạo một tập dữ liệu giả lập có giá trị khuyết thiếu (NaN).
- Sử dụng Pipeline từ sklearn.pipeline nối các bước:
  1. SimpleImputer: điền giá trị khuyết thiếu bằng trung bình (mean).
  2. StandardScaler: chuẩn hóa thang đo dữ liệu.
  3. RandomForestClassifier: mô hình phân loại rừng ngẫu nhiên.
- Huấn luyện và đánh giá Pipeline trên dữ liệu.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 70)
    print("LAB 7 - BÀI 2: XÂY DỰNG PIPELINE HOÀN CHỈNH VỚI RANDOM FOREST")
    print("=" * 70)

    # 1. Tạo tập dữ liệu giả lập có chứa giá trị NaN
    np.random.seed(42)
    n_samples = 150
    n_features = 4

    # Tạo ma trận đặc trưng
    X_raw = np.random.randn(n_samples, n_features) * 10 + 50 # Mean=50, Std=10
    
    # Nhãn mục tiêu phụ thuộc vào đặc trưng đầu tiên: nếu X0 > 50 thì nhãn 1, ngược lại 0
    y = (X_raw[:, 0] + X_raw[:, 1] > 100).astype(int)

    # Cố tình đưa vào các giá trị NaN ngẫu nhiên (khoảng 10% dữ liệu)
    nan_mask = np.random.rand(*X_raw.shape) < 0.10
    X_with_nan = X_raw.copy()
    X_with_nan[nan_mask] = np.nan

    print("\n1. Thông tin tập dữ liệu giả lập:")
    print(f"   - Số lượng mẫu: {n_samples}, Số đặc trưng: {n_features}")
    print(f"   - Tổng số giá trị bị thiếu (NaN): {np.isnan(X_with_nan).sum()} ô")
    print(f"   - 5 dòng đầu tiên của dữ liệu có NaN:\n{pd.DataFrame(X_with_nan[:5], columns=[f'F{i+1}' for i in range(n_features)])}")

    # 2. Phân chia Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_with_nan, y, test_size=0.25, random_state=42
    )

    # 3. Xây dựng Pipeline hoàn chỉnh
    # Bước 1: SimpleImputer (điền NaN bằng mean)
    # Bước 2: StandardScaler (chuẩn hóa dữ liệu)
    # Bước 3: RandomForestClassifier (mô hình học kết hợp)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])

    print("\n2. Cấu trúc Pipeline hoàn chỉnh:")
    for step_name, step_obj in pipeline.steps:
        print(f"   -> Bước [{step_name}]: {step_obj}")

    # 4. Huấn luyện Pipeline
    print("\n3. Đang huấn luyện toàn bộ Pipeline...")
    pipeline.fit(X_train, y_train)
    print("   -> Pipeline đã xử lý tự động điền khuyết, chuẩn hóa và huấn luyện Random Forest!")

    # 5. Đánh giá trên tập Test
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n4. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f"   - Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print("\nBáo cáo phân loại chi tiết:")
    print(classification_report(y_test, y_pred))
    print("=" * 70)

if __name__ == "__main__":
    main()

