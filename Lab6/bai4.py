"""
Lab 6 - Bài 4: Chương trình hoàn chỉnh - Xây dựng và Tối ưu mô hình SVC (Wine Dataset)
Yêu cầu:
1. Tải dữ liệu 'wine' từ sklearn.datasets.
2. Chia dữ liệu thành Train và Test set.
3. Chuẩn hóa dữ liệu với 'StandardScaler'.
4. Sử dụng mô hình 'SVC' (Support Vector Classifier).
5. Dùng 'GridSearchCV' trên tập Train để tìm tham số tốt nhất ('C', 'kernel', 'gamma').
6. Huấn luyện mô hình tốt nhất bằng toàn bộ tập Train (GridSearchCV tự động refit).
7. Dự đoán trên tập Test và in ra 'classification_report' cuối cùng.
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("=" * 70)
    print("LAB 6 - BÀI 4: CHƯƠNG TRÌNH HOÀN CHỈNH - XÂY DỰNG & TỐI ƯU MÔ HÌNH SVC")
    print("=" * 70)

    # 1. Tải dữ liệu Wine
    wine = load_wine()
    X = wine.data
    y = wine.target

    print(f"\n1. Tải tập dữ liệu Wine:")
    print(f"   - Số lượng mẫu: {X.shape[0]}, Số đặc trưng hóa học: {X.shape[1]}")
    print(f"   - Tên các loại rượu: {wine.target_names}")

    # 2. Phân chia Train và Test set (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n2. Phân chia dữ liệu: Train = {X_train.shape[0]} mẫu, Test = {X_test.shape[0]} mẫu.")

    # 3 & 4. Xây dựng Pipeline [StandardScaler -> SVC]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=42))
    ])

    # 5. Cấu hình lưới tham số GridSearchCV trên tập Train
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1]
    }

    print("\n3. Thực hiện tinh chỉnh siêu tham số bằng GridSearchCV trên tập Train...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        refit=True # Tự động huấn luyện lại mô hình tốt nhất trên toàn bộ tập Train
    )

    grid_search.fit(X_train, y_train)

    print("\n4. Kết quả tìm kiếm siêu tham số trên tập Train:")
    print(f"   - Tham số tốt nhất (best_params_): {grid_search.best_params_}")
    print(f"   - Điểm Cross-validation cao nhất trên Train: {grid_search.best_score_ * 100:.2f}%")

    # 6 & 7. Đánh giá mô hình tốt nhất trên tập kiểm thử độc lập (Test set)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n5. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP:")
    print(f"   -> Độ chính xác tổng quan (Test Accuracy): {test_acc * 100:.2f}%")
    print("\nBÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    print("=" * 70)

if __name__ == "__main__":
    main()

