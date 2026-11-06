"""
Lab 4 - Bài 3: Ma trận nhầm lẫn (Confusion Matrix)
Yêu cầu:
- Sử dụng hàm confusion_matrix để tạo ma trận nhầm lẫn từ kết quả dự đoán ở Bài 2.
- Sử dụng ConfusionMatrixDisplay để trực quan hóa ma trận và lưu file ảnh.
- Giải thích chi tiết các thành phần: TP, TN, FP, FN thông qua comment và hiển thị.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from bai1 import get_data_and_model

def main():
    print("=" * 70)
    print("LAB 4 - BÀI 3: MA TRẬN NHẦM LẪN (CONFUSION MATRIX)")
    print("=" * 70)

    # Lấy mô hình và dữ liệu từ Bài 1
    model, _, X_test, _, y_test, data = get_data_and_model()
    y_pred = model.predict(X_test)

    # 1. Tạo ma trận nhầm lẫn
    # Trong sklearn breast_cancer: 0 = 'malignant' (Ác tính), 1 = 'benign' (Lành tính)
    cm = confusion_matrix(y_test, y_pred)
    print("\n1. Ma trận nhầm lẫn (Confusion Matrix mảng 2x2):")
    print(cm)

    # Trích xuất các thành phần:
    # Lớp 0 (Malignant - Ác tính / Dương tính với bệnh nguy hiểm)
    # Lớp 1 (Benign - Lành tính / Âm tính)
    # Quy ước chuẩn theo sklearn:
    tn = cm[0, 0] # Thực tế 0, dự đoán 0
    fp = cm[0, 1] # Thực tế 0, dự đoán 1
    fn = cm[1, 0] # Thực tế 1, dự đoán 0
    tp = cm[1, 1] # Thực tế 1, dự đoán 1

    print("\n2. Phân tích các thành phần của ma trận nhầm lẫn:")
    print(f"   - [0,0] True Malignant (Thực tế Ác tính, đoán đúng Ác tính)     : {cm[0, 0]}")
    print(f"   - [0,1] False Benign   (Thực tế Ác tính, đoán nhầm Lành tính)   : {cm[0, 1]} (Cực kỳ nguy hiểm)")
    print(f"   - [1,0] False Malignant(Thực tế Lành tính, đoán nhầm Ác tính)   : {cm[1, 0]} (Báo động giả)")
    print(f"   - [1,1] True Benign    (Thực tế Lành tính, đoán đúng Lành tính) : {cm[1, 1]}")

    print("\n3. GIẢI THÍCH CHI TIẾT CÁC KHÁI NIỆM (TP, TN, FP, FN):")
    print("""
   - TP (True Positive - Dương tính thật): Số mẫu thực tế là Positive và mô hình dự đoán chính xác là Positive.
   - TN (True Negative - Âm tính thật): Số mẫu thực tế là Negative và mô hình dự đoán chính xác là Negative.
   - FP (False Positive - Dương tính giả / Type I Error): Thực tế là Âm tính nhưng mô hình dự đoán nhầm là Dương tính (Báo động giả).
   - FN (False Negative - Âm tính giả / Type II Error): Thực tế là Dương tính nhưng mô hình bỏ sót và dự đoán là Âm tính.
     Trong y tế, FN là lỗi nguy hiểm nhất vì người bệnh ung thư bị kết luận là khỏe mạnh và không được chữa trị kịp thời.
    """)

    # 4. Trực quan hóa ma trận nhầm lẫn bằng ConfusionMatrixDisplay
    save_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Ma trận nhầm lẫn (Breast Cancer Classification)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"4. Đã trực quan hóa và lưu biểu đồ tại: {save_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

