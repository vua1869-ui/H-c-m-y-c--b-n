"""
Lab 8 - Bài 2: Khởi tạo Baseline Model với Pipeline
Yêu cầu:
- Xây dựng một Pipeline cơ bản gồm 2 bước:
  1. StandardScaler để chuẩn hóa các thuộc tính số.
  2. LogisticRegression đóng vai trò làm Baseline Model (Mô hình nền tảng đối chiếu).
- Huấn luyện Pipeline và đánh giá độ chính xác sơ bộ trên tập Test.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from bai1 import get_prepared_data

def get_baseline_pipeline():
    """Khởi tạo Pipeline Baseline"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

def main():
    print("=" * 70)
    print("LAB 8 - BÀI 2: KHỞI TẠO BASELINE MODEL VỚI PIPELINE")
    print("=" * 70)

    # 1. Lấy dữ liệu chia 75% - 25% từ Bài 1
    X_train, X_test, y_train, y_test, _ = get_prepared_data()

    # 2. Xây dựng Pipeline Baseline: StandardScaler + Logistic Regression
    baseline_pipe = get_baseline_pipeline()

    print("\n1. Cấu trúc Pipeline Baseline:")
    for step_name, step_obj in baseline_pipe.steps:
        print(f"   -> Bước [{step_name}]: {step_obj}")

    # 3. Huấn luyện Baseline Model
    print("\n2. Đang huấn luyện Baseline Model trên 75% dữ liệu Train...")
    baseline_pipe.fit(X_train, y_train)

    # 4. Đánh giá độ chính xác sơ bộ trên tập Test
    y_pred = baseline_pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n3. KẾT QUẢ ĐÁNH GIÁ BASELINE TRÊN TẬP TEST:")
    print(f"   -> Độ chính xác Baseline (Accuracy): {acc * 100:.2f}%")
    print("\nBáo cáo phân loại Baseline (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['Ở lại (0)', 'Rời mạng (1)']))

    print("4. VAI TRÒ CỦA BASELINE MODEL TRONG QUY TRÌNH ML:")
    print("   - Đóng vai trò làm chuẩn đo lường tối thiểu (Benchmark). Mọi mô hình phức tạp hơn sau này")
    print("     (như Random Forest, Boosting) bắt buộc phải vượt qua điểm số Baseline này mới có giá trị sử dụng.")
    print("=" * 70)

if __name__ == "__main__":
    main()

