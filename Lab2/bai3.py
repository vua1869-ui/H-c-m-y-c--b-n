"""
Lab 2 - Bài 3: Chia tập dữ liệu Train / Test
Yêu cầu:
- Sử dụng tập dữ liệu diabetes từ sklearn.datasets.
- Sử dụng hàm train_test_split để chia dữ liệu thành 2 tập: Train (80%) và Test (20%).
- In ra kích thước của tập dữ liệu gốc và kích thước của hai tập sau khi chia.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def main():
    print("=" * 65)
    print("LAB 2 - BÀI 3: PHÂN CHIA TẬP DỮ LIỆU TRAIN / TEST")
    print("=" * 65)

    # 1. Tải tập dữ liệu Diabetes
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names

    print("\n1. Tập dữ liệu gốc Diabetes:")
    print(f"   - Số lượng mẫu (samples): {X.shape[0]}")
    print(f"   - Số lượng đặc trưng (features): {X.shape[1]}")
    print(f"   - Kích thước ma trận đặc trưng X: {X.shape}")
    print(f"   - Kích thước vector nhãn y: {y.shape}")
    print(f"   - Danh sách các đặc trưng: {feature_names}")

    # 2. Phân chia Train/Test theo tỉ lệ 80% - 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("\n2. Kết quả phân chia Train (80%) và Test (20%):")
    print(f"   - Kích thước tập X_train : {X_train.shape} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - Kích thước tập y_train : {y_train.shape}")
    print(f"   - Kích thước tập X_test  : {X_test.shape} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   - Kích thước tập y_test  : {y_test.shape}")

    # 3. Giải thích ý nghĩa của việc chia Train/Test
    print("\n3. Ý nghĩa của việc phân chia dữ liệu:")
    print("   - Tập Huấn luyện (Train set): Dùng để huấn luyện mô hình học quy luật.")
    print("   - Tập Kiểm thử (Test set): Giữ kín, đóng vai trò như dữ liệu mới trong thực tế")
    print("     nhằm đánh giá độ chính xác khách quan và kiểm tra hiện tượng Overfitting.")
    print("   - random_state: Cố định hạt giống ngẫu nhiên để kết quả có thể tái lập (reproducible).")
    print("=" * 65)

if __name__ == "__main__":
    main()

