"""
Lab 5 - Bài 1: Phân loại với K-Nearest Neighbors (KNN)
Yêu cầu:
- Sử dụng tập dữ liệu 'iris' từ 'sklearn.datasets'.
- Chia dữ liệu thành tập huấn luyện và tập kiểm tra (tỉ lệ 7:3).
- Huấn luyện mô hình KNN với 'n_neighbors=5'.
- Dự đoán kết quả trên tập kiểm tra và in ra độ chính xác (accuracy).
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_iris_split():
    """Hàm tiện ích chia dữ liệu Iris theo tỉ lệ 7:3 cho các bài tiếp theo"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, iris

def main():
    print("=" * 65)
    print("LAB 5 - BÀI 1: PHÂN LOẠI VỚI K-NEAREST NEIGHBORS (KNN)")
    print("=" * 65)

    # 1. Tải dữ liệu và chia 7:3
    X_train, X_test, y_train, y_test, iris = get_iris_split()

    print("\n1. Thông tin tập dữ liệu Iris:")
    print(f"   - Tổng số mẫu: {len(iris.data)}")
    print(f"   - Số mẫu tập Train (70%): {X_train.shape[0]}")
    print(f"   - Số mẫu tập Test  (30%): {X_test.shape[0]}")
    print(f"   - Tên các loài hoa: {iris.target_names}")

    # 2. Khởi tạo và huấn luyện mô hình KNN với k=5
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # 3. Dự đoán trên tập kiểm tra
    y_pred = knn.predict(X_test)

    # 4. Tính toán và in độ chính xác
    acc = accuracy_score(y_test, y_pred)
    print("\n2. Kết quả đánh giá mô hình KNN:")
    print(f"   - Tham số k (n_neighbors): 5")
    print(f"   - Khoảng cách đo lường: Euclidean")
    print(f"   - Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()

