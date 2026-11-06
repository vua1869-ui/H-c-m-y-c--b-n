"""
Lab 7 - Bài 4: Chương trình Demo Pipeline kết hợp GridSearchCV cho Random Forest
Yêu cầu:
- Áp dụng Pipeline trên tập dữ liệu breast_cancer.
- Xây dựng Pipeline gồm: StandardScaler và RandomForestClassifier.
- Sử dụng GridSearchCV để tinh chỉnh siêu tham số bên trong Pipeline:
  + 'randomforestclassifier__n_estimators': [50, 100, 200]
  + 'randomforestclassifier__max_depth': [None, 10, 20]
- In ra bộ tham số tốt nhất và độ chính xác kiểm định tương ứng.
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 70)
    print("LAB 7 - BÀI 4: PIPELINE KẾT HỢP GRIDSEARCHCV CHO RANDOM FOREST")
    print("=" * 70)

    # 1. Tải dữ liệu Breast Cancer
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    print(f"\n1. Tải tập dữ liệu Breast Cancer:")
    print(f"   - Số lượng mẫu: {X.shape[0]}, Số đặc trưng: {X.shape[1]}")

    # 2. Phân chia Train/Test (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Xây dựng Pipeline: StandardScaler + RandomForestClassifier
    # Đặt tên các bước rõ ràng để định tuyến tham số trong GridSearchCV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('randomforestclassifier', RandomForestClassifier(random_state=42))
    ])

    # 4. Định nghĩa không gian siêu tham số cần tìm kiếm theo đúng định dạng đề bài
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100, 200],
        'randomforestclassifier__max_depth': [None, 10, 20]
    }

    print("\n2. Không gian siêu tham số cần tinh chỉnh:")
    print("   - n_estimators: [50, 100, 200]")
    print("   - max_depth   : [None, 10, 20]")
    print("   -> Tổng cộng 9 tổ hợp tham số x 5 folds = 45 mô hình.")

    # 5. Chạy GridSearchCV trên tập Train
    print("\n3. Đang thực thi GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    # 6. In kết quả tốt nhất
    print("\n4. KẾT QUẢ TỐI ƯU CỦA GRIDSEARCHCV:")
    print(f"   - Siêu tham số tốt nhất (best_params_):")
    for param_name, param_val in grid_search.best_params_.items():
        print(f"     + {param_name}: {param_val}")
    print(f"   - Điểm số Cross-validation cao nhất trên Train: {grid_search.best_score_ * 100:.2f}%")

    # 7. Đánh giá mô hình tối ưu trên tập Test độc lập
    best_pipe = grid_search.best_estimator_
    y_pred = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n5. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP:")
    print(f"   -> Độ chính xác kiểm thử (Test Accuracy): {test_acc * 100:.2f}%")
    print("\nBáo cáo phân loại chi tiết:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    print("=" * 70)

if __name__ == "__main__":
    main()

