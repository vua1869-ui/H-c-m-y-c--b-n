"""
Lab 4 - Bài 1: Xây dựng mô hình Logistic Regression
Yêu cầu:
- Sử dụng tập dữ liệu breast_cancer từ sklearn.datasets.
- Thực hiện các bước: Load dữ liệu -> Chia Train/Test -> Chuẩn hóa dữ liệu (Scaling).
- Huấn luyện mô hình LogisticRegression từ sklearn.linear_model.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_data_and_model():
    """Hàm tiện ích chuẩn bị dữ liệu và mô hình để tái sử dụng ở Bài 2 & 3"""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Chia Train/Test 80/20 với phân tầng (stratify) theo nhãn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Chuẩn hóa dữ liệu với StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, X_train_scaled, X_test_scaled, y_train, y_test, data

def main():
    print("=" * 65)
    print("LAB 4 - BÀI 1: XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION")
    print("=" * 65)

    model, X_train, X_test, y_train, y_test, data = get_data_and_model()

    print("\n1. Thông tin tập dữ liệu Breast Cancer:")
    print(f"   - Tổng số mẫu: {data.data.shape[0]}")
    print(f"   - Số đặc trưng: {data.data.shape[1]}")
    print(f"   - Tên các lớp: {data.target_names} (0: Ác tính - Malignant, 1: Lành tính - Benign)")
    print(f"   - Số mẫu Train: {X_train.shape[0]}, Số mẫu Test: {X_test.shape[0]}")

    print("\n2. Chuẩn hóa dữ liệu (StandardScaler):")
    print("   - Đã fit và biến đổi trên X_train, áp dụng transform trên X_test.")

    print("\n3. Huấn luyện mô hình Logistic Regression:")
    print(f"   - Mô hình: {model}")
    print(f"   - Số lần lặp hội tụ: {model.n_iter_[0]}")
    print("   -> Đã huấn luyện mô hình thành công!")
    print("=" * 65)

if __name__ == "__main__":
    main()

