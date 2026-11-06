"""
Lab 1 - Bài 3: Chạy thử mô hình đơn giản với Scikit-Learn
Yêu cầu:
- Sử dụng tập dữ liệu có sẵn iris từ sklearn.datasets.
- Khởi tạo một mô hình đơn giản (Linear Regression - để xem cách model hoạt động theo yêu cầu đề bài).
- Thực hiện các bước: Load dữ liệu -> Khởi tạo model -> Fit model (huấn luyện) -> Predict (dự đoán).
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

def main():
    print("=" * 60)
    print("LAB 1 - BÀI 3: CHẠY THỬ MÔ HÌNH VỚI SCIKIT-LEARN")
    print("=" * 60)

    # Bước 1: Load dữ liệu
    print("\n[Bước 1] Tải tập dữ liệu Iris...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"- Kích thước ma trận đặc trưng X: {X.shape} (150 mẫu, 4 đặc trưng)")
    print(f"- Kích thước vector nhãn y: {y.shape}")
    print(f"- Tên các đặc trưng: {feature_names}")
    print(f"- Các lớp hoa (target names): {target_names}")

    # Bước 2: Khởi tạo model (Linear Regression theo đề bài)
    print("\n[Bước 2] Khởi tạo mô hình Linear Regression...")
    model_lr = LinearRegression()

    # Bước 3: Huấn luyện mô hình (Fit)
    print("\n[Bước 3] Huấn luyện mô hình (Fit)...")
    model_lr.fit(X, y)
    print("Mô hình đã được huấn luyện xong!")
    print(f"- Trọng số (coefficients): {model_lr.coef_}")
    print(f"- Hệ số chặn (intercept): {model_lr.intercept_:.4f}")

    # Bước 4: Dự đoán (Predict)
    print("\n[Bước 4] Thực hiện dự đoán trên 5 mẫu đầu tiên...")
    sample_X = X[:5]
    real_y = y[:5]
    predictions_continuous = model_lr.predict(sample_X)

    print("So sánh nhãn thực tế và kết quả dự đoán của Linear Regression:")
    for i in range(len(sample_X)):
        print(f"  Mẫu {i+1}: Thực tế = {real_y[i]} ({target_names[real_y[i]]}) "
              f"| Dự đoán liên tục = {predictions_continuous[i]:.3f} "
              f"| Làm tròn thành lớp = {int(round(predictions_continuous[i]))}")

    # Mở rộng đối chiếu: Sử dụng đúng mô hình Logistic Regression cho bài toán phân loại
    print("\n--- Đối chiếu bổ sung: Dùng Logistic Regression (Mô hình chuẩn cho bài toán phân loại) ---")
    model_clf = LogisticRegression(max_iter=200)
    model_clf.fit(X, y)
    preds_clf = model_clf.predict(sample_X)
    print("Kết quả dự đoán của Logistic Regression:")
    for i in range(len(sample_X)):
        print(f"  Mẫu {i+1}: Thực tế = {real_y[i]} -> Dự đoán = {preds_clf[i]} ({target_names[preds_clf[i]]})")

    print("=" * 60)

if __name__ == "__main__":
    main()

