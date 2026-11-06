"""
Lab 4 - Bài 4: Quy trình ML hoàn chỉnh - Phần 4: Áp dụng Logistic Regression vào tập Digits
Yêu cầu:
- Tải tập dữ liệu digits.
- Tiền xử lý với StandardScaler và chia Train/Test.
- Huấn luyện Logistic Regression cho bài toán phân loại đa lớp (Multi-class).
- Đánh giá mô hình trên tập Test.
- Tìm và in ra 5 mẫu dự đoán sai của mô hình (so sánh nhãn thực tế và nhãn dự đoán),
  đồng thời trực quan hóa hình ảnh các mẫu sai đó.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=" * 70)
    print("LAB 4 - BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - PHẦN 4")
    print("LOGISTIC REGRESSION CHO PHÂN LOẠI CHỮ SỐ ĐA LỚP (DIGITS)")
    print("=" * 70)

    # 1. Load dữ liệu digits
    digits = load_digits()
    X = digits.data
    y = digits.target
    images = digits.images

    print(f"\n1. Thông tin tập dữ liệu Digits:")
    print(f"   - Số lượng mẫu: {X.shape[0]}, Số đặc trưng pixel: {X.shape[1]}")
    print(f"   - 10 lớp phân loại: {list(range(10))}")

    # 2. Phân chia Train/Test (Lưu ý giữ chỉ số index để truy xuất lại hình ảnh gốc)
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.25, random_state=42, stratify=y
    )

    # 3. Xây dựng Pipeline [StandardScaler -> LogisticRegression]
    # Với phân loại đa lớp, LogisticRegression hỗ trợ 'ovr' (One-vs-Rest) hoặc 'multinomial'
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("\n2. Huấn luyện mô hình Logistic Regression đa lớp...")
    pipeline.fit(X_train, y_train)

    # 4. Đánh giá mô hình trên tập Test
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n3. Độ chính xác tổng thể trên tập Test: {acc * 100:.2f}%")
    print("\nBáo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred))

    # 5. Tìm các mẫu dự đoán sai (Misclassified samples)
    wrong_mask = (y_test != y_pred)
    wrong_indices_in_test = np.where(wrong_mask)[0]
    total_wrong = len(wrong_indices_in_test)

    print(f"\n4. Số lượng mẫu dự đoán sai: {total_wrong} / {len(y_test)} mẫu ({total_wrong/len(y_test)*100:.2f}%)")
    print("   Chi tiết 5 mẫu dự đoán sai đầu tiên:")

    num_samples_to_show = min(5, total_wrong)
    wrong_samples_info = []

    for i in range(num_samples_to_show):
        test_pos = wrong_indices_in_test[i]
        orig_idx = idx_test[test_pos]
        true_label = y_test[test_pos]
        pred_label = y_pred[test_pos]
        wrong_samples_info.append((orig_idx, true_label, pred_label))
        print(f"   - Mẫu #{i+1} (Chỉ số gốc: {orig_idx}): Thực tế = {true_label} | Mô hình dự đoán nhầm thành = {pred_label}")

    # 6. Trực quan hóa hình ảnh 5 mẫu đoán sai
    if num_samples_to_show > 0:
        save_path = os.path.join(os.path.dirname(__file__), "misclassified_digits.png")
        fig, axes = plt.subplots(1, num_samples_to_show, figsize=(3 * num_samples_to_show, 3))
        if num_samples_to_show == 1:
            axes = [axes]

        for i in range(num_samples_to_show):
            orig_idx, true_label, pred_label = wrong_samples_info[i]
            axes[i].imshow(images[orig_idx], cmap='gray_r')
            axes[i].set_title(f"Thực tế: {true_label}\nĐoán sai: {pred_label}", color='crimson', fontsize=12)
            axes[i].axis('off')

        plt.suptitle("Các mẫu chữ số bị dự đoán sai bởi Logistic Regression", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n5. Đã lưu hình ảnh trực quan 5 mẫu đoán sai tại: {save_path}")

    print("=" * 70)

if __name__ == "__main__":
    main()

