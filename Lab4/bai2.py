"""
Lab 4 - Bài 2: Đánh giá mô hình phân loại (Evaluation Metrics)
Yêu cầu:
- Sử dụng mô hình đã huấn luyện ở Bài 1 để dự đoán trên tập Test.
- Tính toán và in ra các chỉ số:
  1. accuracy_score
  2. precision_score
  3. recall_score
- In báo cáo phân loại chi tiết bằng classification_report.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from bai1 import get_data_and_model

def main():
    print("=" * 65)
    print("LAB 4 - BÀI 2: ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI (BREAST CANCER)")
    print("=" * 65)

    # Lấy mô hình và dữ liệu đã chuẩn hóa từ Bài 1
    model, _, X_test, _, y_test, data = get_data_and_model()

    # 1. Dự đoán trên tập Test
    y_pred = model.predict(X_test)

    # 2. Tính toán các chỉ số cơ bản
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n1. CÁC CHỈ SỐ ĐÁNH GIÁ ĐỘ CHÍNH XÁC:")
    print(f"   - Accuracy  (Độ chính xác tổng quan) : {acc * 100:.2f}%")
    print(f"   - Precision (Độ chuẩn xác)            : {prec * 100:.2f}%")
    print(f"   - Recall    (Độ thu hồi / Nhạy)       : {rec * 100:.2f}%")
    print(f"   - F1-Score  (Trung bình điều hòa)     : {f1 * 100:.2f}%")

    # 3. Báo cáo phân loại chi tiết
    print("\n2. BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
    target_names = [f"{name} ({idx})" for idx, name in enumerate(data.target_names)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 4. Ý nghĩa y khoa
    print("3. Ý NGHĨA TRONG BÀI TOÁN Y TẾ (CHẨN ĐOÁN UNG THƯ):")
    print("   - Trong chẩn đoán y tế, chỉ số RECALL cực kỳ quan trọng vì mục tiêu tối thượng")
    print("     là KHÔNG ĐƯỢC BỎ SÓT bệnh nhân mắc ung thư (hạn chế tối đa False Negative).")
    print("=" * 65)

if __name__ == "__main__":
    main()

