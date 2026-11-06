"""
Lab 6 - Bài 3: Tinh chỉnh siêu tham số (Hyperparameter Tuning) với Grid Search
Yêu cầu:
- Tải tập dữ liệu 'breast_cancer' từ sklearn.datasets.
- Sử dụng mô hình K-Nearest Neighbors (KNN).
- Cấu hình param_grid gồm:
  + 'n_neighbors': từ 1 đến 20
  + 'weights': ['uniform', 'distance']
- Sử dụng 'GridSearchCV' với k=5 để tìm bộ tham số tối ưu.
- In ra bộ tham số tốt nhất (best_params_) và độ chính xác cao nhất (best_score_).
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def main():
    print("=" * 70)
    print("LAB 6 - BÀI 3: TINH CHỈNH SIÊU THAM SỐ VỚI GRID SEARCH CV")
    print("=" * 70)

    # 1. Tải dữ liệu breast_cancer
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    # Tiền xử lý: Chuẩn hóa thang đo đặc trưng cho KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n1. Thông tin tập dữ liệu Breast Cancer:")
    print(f"   - Số lượng mẫu: {X.shape[0]}, Số đặc trưng: {X.shape[1]}")
    print(f"   - Dữ liệu đã được chuẩn hóa với StandardScaler.")

    # 2. Khởi tạo mô hình KNN và lưới tham số (param_grid)
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance']
    }

    print(f"\n2. Thiết lập không gian tìm kiếm siêu tham số (Grid):")
    print(f"   - n_neighbors: từ 1 đến 20 (20 giá trị)")
    print(f"   - weights: ['uniform', 'distance'] (2 giá trị)")
    print(f"   -> Tổng số mô hình cần đánh giá: 20 x 2 = 40 mô hình x 5 folds = 200 lần huấn luyện.")

    # 3. Chạy GridSearchCV với k=5
    print("\n3. Đang thực thi GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1
    )
    grid_search.fit(X_scaled, y)

    # 4. In kết quả tối ưu
    print("\n4. KẾT QUẢ TỐI ƯU CỦA GRID SEARCH CV:")
    print(f"   - Bộ tham số tốt nhất (best_params_): {grid_search.best_params_}")
    print(f"   - Độ chính xác cao nhất (best_score_): {grid_search.best_score_ * 100:.2f}%")
    print(f"   - Mô hình tốt nhất (best_estimator_) : {grid_search.best_estimator_}")
    print("=" * 70)

if __name__ == "__main__":
    main()

