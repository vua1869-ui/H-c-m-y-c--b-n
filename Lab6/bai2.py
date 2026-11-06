"""
Lab 6 - Bài 2: Đánh giá mô hình với Cross-Validation (K-Fold Kiểm định chéo)
Yêu cầu:
- Tải tập dữ liệu 'digits' từ sklearn.datasets.
- Khởi tạo mô hình Decision Tree Classifier.
- Sử dụng hàm 'cross_val_score' với k=5 (5-fold cross-validation) để đánh giá độ chính xác.
- In ra mảng kết quả accuracy của 5 lần chạy và tính giá trị accuracy trung bình kèm độ lệch chuẩn.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold

def main():
    print("=" * 70)
    print("LAB 6 - BÀI 2: ĐÁNH GIÁ MÔ HÌNH VỚI 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    # 1. Tải tập dữ liệu Digits
    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"\n1. Thông tin tập dữ liệu Digits:")
    print(f"   - Tổng số mẫu: {X.shape[0]}")
    print(f"   - Số đặc trưng: {X.shape[1]}")

    # 2. Khởi tạo mô hình Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)

    # 3. Thực hiện 5-Fold Cross-Validation
    # K-Fold: Chia ngẫu nhiên 5 phần, mỗi lần lấy 4 phần train và 1 phần kiểm tra
    k_folds = 5
    cv_strategy = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = cross_val_score(dt_model, X, y, cv=cv_strategy, scoring='accuracy')

    # 4. Hiển thị kết quả chi tiết của từng fold
    print(f"\n2. Kết quả 5-Fold Cross-Validation:")
    for i, score in enumerate(scores, 1):
        print(f"   - Fold {i}: Accuracy = {score * 100:.2f}%")

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"\n3. Thống kê tổng hợp:")
    print(f"   - Mảng Accuracy 5 folds: {np.round(scores, 4)}")
    print(f"   - Độ chính xác trung bình (Mean Accuracy): {mean_acc * 100:.2f}%")
    print(f"   - Độ lệch chuẩn (Std Deviation)         : {std_acc * 100:.2f}%")
    print(f"   -> Khoảng tin cậy ước lượng             : {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

    # 5. Giải thích lý do sử dụng Cross-Validation
    print("\n4. Ý NGHĨA CỦA K-FOLD CROSS-VALIDATION:")
    print("   - Đánh giá khách quan và ổn định hơn so với việc chỉ chia Train/Test ngẫu nhiên một lần.")
    print("   - Tránh việc điểm số bị 'may mắn' hoặc 'xui xẻo' do việc phân chia ngẫu nhiên dữ liệu.")
    print("   - Mọi mẫu dữ liệu đều được xuất hiện trong tập kiểm tra đúng 1 lần.")
    print("=" * 70)

if __name__ == "__main__":
    main()

