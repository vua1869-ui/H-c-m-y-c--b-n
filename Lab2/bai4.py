"""
Lab 2 - Bài 4: Quy trình ML hoàn chỉnh - Phần 2: Tiền xử lý dữ liệu (Digits Dataset)
Yêu cầu:
- Tiếp tục với tập dữ liệu digits từ Lab 1.
- Thực hiện các bước tiền xử lý:
  1. Chuẩn hóa dữ liệu (Scaling) vì giá trị pixel nằm trong khoảng 0-16.
  2. Chia tập dữ liệu thành Train và Test.
  3. Xây dựng một Pipeline đơn giản chứa bước chuẩn hóa (StandardScaler/MinMaxScaler)
     và một model (Logistic Regression hoặc SVM).
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 70)
    print("LAB 2 - BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - PHẦN 2: TIỀN XỬ LÝ & PIPELINE")
    print("=" * 70)

    # 1. Tải dữ liệu digits
    digits = load_digits()
    X = digits.data
    y = digits.target

    print("\n1. Thông tin dữ liệu ban đầu:")
    print(f"   - Kích thước tập dữ liệu X: {X.shape}")
    print(f"   - Dải giá trị pixel gốc: Min = {X.min()}, Max = {X.max()}")

    # 2. Phân chia Train/Test (80% train, 20% test)
    # Lưu ý: Nên chia trước khi fit Pipeline để tránh rò rỉ dữ liệu (Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n2. Phân chia dữ liệu: Train = {X_train.shape[0]} mẫu, Test = {X_test.shape[0]} mẫu.")

    # 3. Xây dựng Pipeline 1: MinMaxScaler + Logistic Regression
    print("\n3. Xây dựng Pipeline 1: [MinMaxScaler -> LogisticRegression]")
    pipe_lr = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', LogisticRegression(max_iter=500, random_state=42))
    ])

    # Huấn luyện pipeline
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"   -> Độ chính xác của Pipeline Logistic Regression trên Test set: {acc_lr * 100:.2f}%")

    # 4. Xây dựng Pipeline 2: StandardScaler + Support Vector Machine (SVC)
    print("\n4. Xây dựng Pipeline 2: [StandardScaler -> SVC]")
    pipe_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=10.0, random_state=42))
    ])

    # Huấn luyện pipeline
    pipe_svm.fit(X_train, y_train)
    y_pred_svm = pipe_svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"   -> Độ chính xác của Pipeline SVM trên Test set: {acc_svm * 100:.2f}%")

    # 5. So sánh và báo cáo chi tiết cho Pipeline tốt nhất
    print("\n5. Báo cáo chi tiết kết quả (Classification Report) của mô hình tốt nhất (SVM):")
    print(classification_report(y_test, y_pred_svm))

    print("6. Ưu điểm nổi bật của việc sử dụng Pipeline trong Machine Learning:")
    print("   - Đóng gói toàn bộ các bước tiền xử lý và mô hình thành một luồng duy nhất.")
    print("   - Tránh hiện tượng rò rỉ dữ liệu (Data Leakage) từ tập Test sang tập Train.")
    print("   - Giúp mã nguồn gọn gàng, dễ bảo trì và dễ dàng triển khai thực tế (Production).")
    print("=" * 70)

if __name__ == "__main__":
    main()

