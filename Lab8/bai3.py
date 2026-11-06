"""
Lab 8 - Bài 3: Tối ưu và Đánh giá nâng cao (Random Forest & GridSearchCV)
Yêu cầu:
- Nâng cấp mô hình trong Pipeline từ Baseline lên RandomForestClassifier.
- Sử dụng GridSearchCV để tối ưu hóa siêu tham số: 'n_estimators', 'max_depth'.
- Đánh giá mô hình tối ưu trên tập Test.
- In ra classification_report (Precision, Recall, F1-score) và vẽ Confusion Matrix.
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl')
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from bai1 import get_prepared_data

def get_optimized_pipeline(X_train, y_train):
    """Xây dựng và tối ưu Pipeline Random Forest bằng GridSearchCV"""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [50, 100, 150],
        'rf__max_depth': [4, 6, 8, None],
        'rf__min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=1,
        refit=True
    )
    grid.fit(X_train, y_train)
    return grid

def main():
    print("=" * 70)
    print("LAB 8 - BÀI 3: TỐI ƯU SIÊU THAM SỐ VÀ ĐÁNH GIÁ NÂNG CAO")
    print("=" * 70)

    # 1. Lấy dữ liệu chia 75% - 25%
    X_train, X_test, y_train, y_test, _ = get_prepared_data()

    # 2. Thực hiện GridSearch tối ưu Pipeline Random Forest
    print("\n1. Đang thực thi GridSearchCV tìm kiếm siêu tham số tối ưu...")
    grid = get_optimized_pipeline(X_train, y_train)

    print(f"\n2. KẾT QUẢ TỐI ƯU:")
    print(f"   - Siêu tham số tốt nhất (best_params_): {grid.best_params_}")
    print(f"   - Điểm F1-score kiểm định cao nhất (best_score_): {grid.best_score_:.4f}")

    # 3. Dự đoán trên tập kiểm thử Test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n3. ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH TỐI ƯU TRÊN TEST SET:")
    print(f"   -> Độ chính xác kiểm thử (Accuracy): {acc * 100:.2f}%")
    print("\nBáo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Ở lại (0)', 'Rời mạng (1)']))

    # 4. Trực quan hóa ma trận nhầm lẫn Confusion Matrix
    save_path = os.path.join(os.path.dirname(__file__), "best_confusion_matrix.png")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ở lại', 'Rời mạng'])
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Ma trận nhầm lẫn mô hình Churn Prediction tối ưu", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"4. Đã vẽ và lưu biểu đồ Confusion Matrix tại: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

