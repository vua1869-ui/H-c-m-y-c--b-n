"""
Lab 2 - Bài 2: Chuẩn hóa dữ liệu (Feature Scaling)
Yêu cầu:
- Tạo một mảng dữ liệu có các cột với đơn vị và thang đo khác nhau (ví dụ: Tuổi từ 20-60, Thu nhập từ 10tr - 100tr).
- Sử dụng StandardScaler và MinMaxScaler từ sklearn.preprocessing.
- In kết quả sau khi chuẩn hóa để so sánh và giải thích.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    print("=" * 70)
    print("LAB 2 - BÀI 2: CHUẨN HÓA DỮ LIỆU (FEATURE SCALING)")
    print("=" * 70)

    # 1. Tạo mảng dữ liệu mẫu: [Tuổi, Thu nhập (triệu VNĐ), Điểm tín dụng]
    # Tuổi: ~20 - 60
    # Thu nhập: ~10 - 100
    # Điểm tín dụng: ~300 - 850
    np.random.seed(42)
    raw_data = np.array([
        [22, 12.0, 580],
        [35, 45.0, 720],
        [48, 85.0, 650],
        [28, 25.0, 610],
        [58, 98.0, 800],
        [24, 15.0, 590],
        [41, 60.0, 710]
    ])
    feature_names = ['Tuổi (năm)', 'Thu nhập (triệu/tháng)', 'Điểm tín dụng (điểm)']
    df_raw = pd.DataFrame(raw_data, columns=feature_names)

    print("\n1. Dữ liệu gốc (Chênh lệch thang đo lớn giữa các cột):")
    print(df_raw.to_string())

    # 2. Chuẩn hóa bằng StandardScaler: z = (x - mean) / std (phân phối chuẩn Mean=0, Std=1)
    standard_scaler = StandardScaler()
    scaled_standard = standard_scaler.fit_transform(raw_data)
    df_standard = pd.DataFrame(scaled_standard, columns=feature_names)

    print("\n2. Dữ liệu sau khi áp dụng StandardScaler (Mean = 0, Std = 1):")
    print(df_standard.round(4).to_string())
    print("\nKiểm tra giá trị Mean và Std sau chuẩn hóa chuẩn tắc:")
    print("   Mean của các cột:", np.round(scaled_standard.mean(axis=0), 4))
    print("   Std của các cột :", np.round(scaled_standard.std(axis=0), 4))

    # 3. Chuẩn hóa bằng MinMaxScaler: x_scaled = (x - min) / (max - min) (đưa về đoạn [0, 1])
    minmax_scaler = MinMaxScaler()
    scaled_minmax = minmax_scaler.fit_transform(raw_data)
    df_minmax = pd.DataFrame(scaled_minmax, columns=feature_names)

    print("\n3. Dữ liệu sau khi áp dụng MinMaxScaler (Đưa về khoảng [0, 1]):")
    print(df_minmax.round(4).to_string())
    print("\nKiểm tra giá trị Min và Max sau chuẩn hóa Min-Max:")
    print("   Min của các cột:", np.round(scaled_minmax.min(axis=0), 4))
    print("   Max của các cột:", np.round(scaled_minmax.max(axis=0), 4))

    # 4. Nhận xét và so sánh
    print("\n4. NHẬN XÉT SO SÁNH:")
    print("   - StandardScaler: Thích hợp khi dữ liệu tuân theo phân phối chuẩn (Gaussian) hoặc các mô hình")
    print("     dựa trên khoảng cách/gradient (SVM, Logistic Regression, Linear Regression, PCA).")
    print("   - MinMaxScaler: Thích hợp khi cần dữ liệu nằm chính xác trong khoảng cố định [0, 1]")
    print("     (như xử lý điểm ảnh hình ảnh hoặc mạng nơ-ron Neural Networks).")
    print("=" * 70)

if __name__ == "__main__":
    main()

