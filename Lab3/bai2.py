"""
Lab 3 - Bài 2: Đánh giá mô hình Hồi quy tuyến tính
Yêu cầu:
- Sử dụng tập dữ liệu diabetes từ sklearn.datasets (lấy 1 đặc trưng duy nhất để dễ trực quan hóa).
- Chia dữ liệu Train/Test.
- Huấn luyện mô hình và tính toán các chỉ số đánh giá: MAE, MSE, R-squared trên tập Test.
- Giải thích ý nghĩa của các chỉ số này thông qua comment và in ra màn hình.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=" * 70)
    print("LAB 3 - BÀI 2: ĐÁNH GIÁ MÔ HÌNH HỒI QUY TUYẾN TÍNH (DIABETES 1 FEATURE)")
    print("=" * 70)

    # 1. Tải dữ liệu diabetes và lấy 1 đặc trưng (cột chỉ số khối cơ thể BMI - index 2)
    diabetes = load_diabetes()
    feature_index = 2 # Chỉ số BMI
    feature_name = diabetes.feature_names[feature_index]
    
    # Định dạng X thành mảng 2D (n_samples, 1)
    X = diabetes.data[:, np.newaxis, feature_index]
    y = diabetes.target # Tiến triển bệnh tiểu đường sau 1 năm

    print(f"\n1. Tập dữ liệu Diabetes:")
    print(f"   - Đặc trưng được chọn: '{feature_name}' (Chỉ số khối cơ thể - BMI)")
    print(f"   - Số lượng mẫu: {X.shape[0]}")

    # 2. Phân chia Train/Test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   - Số mẫu Train: {X_train.shape[0]}, Số mẫu Test: {X_test.shape[0]}")

    # 3. Huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Dự đoán trên tập Test
    y_pred = model.predict(X_test)

    # 5. Tính toán các chỉ số đánh giá mô hình
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n2. CÁC CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f"   - MAE (Mean Absolute Error)     : {mae:.4f}")
    print(f"   - MSE (Mean Squared Error)      : {mse:.4f}")
    print(f"   - RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"   - R-squared (R2 Score)          : {r2:.4f}")

    # 6. Giải thích chi tiết ý nghĩa từng chỉ số
    print("\n3. GIẢI THÍCH CHI TIẾT Ý NGHĨA CÁC CHỈ SỐ:")
    print("""
   a) MAE (Mean Absolute Error - Sai số tuyệt đối trung bình):
      - Công thức: MAE = (1/n) * sum(|y_thuc_te - y_du_doan|)
      - Ý nghĩa: Cho biết mức độ chênh lệch trung bình tuyệt đối giữa giá trị thực tế
        và giá trị dự đoán theo đúng đơn vị của biến mục tiêu y. Càng nhỏ càng tốt.
      - Ưu điểm: Ít bị chi phối bởi các giá trị ngoại lệ (outliers).

   b) MSE (Mean Squared Error - Sai số bình phương trung bình):
      - Công thức: MSE = (1/n) * sum((y_thuc_te - y_du_doan)^2)
      - Ý nghĩa: Bình phương các độ lệch nên phạt rất nặng những lỗi sai lớn. Càng nhỏ càng tốt.
      - Hạn chế: Đơn vị bị bình phương nên khó hình dung theo đơn vị thực tế.

   c) RMSE (Root Mean Squared Error - Căn bậc hai sai số bình phương):
      - Công thức: RMSE = sqrt(MSE)
      - Ý nghĩa: Khôi phục lại đúng đơn vị đo lường của biến mục tiêu nhưng vẫn giữ ưu điểm phạt nặng sai số lớn.

   d) R-squared (Hệ số xác định - Coefficient of Determination):
      - Công thức: R2 = 1 - (SS_res / SS_tot)
      - Ý nghĩa: Đo lường tỷ lệ phương sai của biến mục tiêu được giải thích bởi mô hình.
      - Giá trị nằm trong khoảng (-vô cùng, 1.0]:
        + R2 = 1.0: Mô hình dự đoán chính xác tuyệt đối 100%.
        + R2 = 0.0: Mô hình dự đoán chỉ ngang bằng mức trung bình của dữ liệu (Baseline).
        + R2 < 0.0: Mô hình dự đoán tệ hơn cả mức trung bình.
    """)
    print("=" * 70)

if __name__ == "__main__":
    main()

