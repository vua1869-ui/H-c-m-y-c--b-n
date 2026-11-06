"""
Lab 8 - Bài 1: Nắm bắt dữ liệu và Tiền xử lý (Mini Project: Customer Churn Prediction)
Yêu cầu:
- Tạo một tập dữ liệu giả lập chuẩn cấu trúc bài toán Phân loại khách hàng rời mạng (Churn Prediction).
- Chia nhỏ thành các đặc trưng (Features - X) và nhãn (Labels - y).
- Phân chia tập dữ liệu thành Train set và Test set theo tỉ lệ 75% - 25%.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_churn_dataset(n_samples=1000, random_state=42):
    """
    Hàm sinh dữ liệu giả lập bài toán Churn Prediction (Khách hàng rời mạng):
    - Tuoi (18 - 70 tuổi)
    - So_thang_dung (1 - 72 tháng gắn bó)
    - Cuoc_thang (20 - 120 USD/tháng)
    - Tong_chi_tieu (Tương quan giữa cước tháng và số tháng dùng)
    - So_lan_khieu_nai (0 - 8 lần gọi tổng đài)
    - Churn: 1 (Rời mạng) hoặc 0 (Ở lại)
    """
    np.random.seed(random_state)
    tuoi = np.random.randint(18, 70, size=n_samples)
    so_thang = np.random.randint(1, 72, size=n_samples)
    cuoc_thang = np.random.uniform(20.0, 120.0, size=n_samples)
    tong_chi_tieu = cuoc_thang * so_thang + np.random.normal(0, 50, size=n_samples)
    tong_chi_tieu = np.maximum(tong_chi_tieu, cuoc_thang) # Đảm bảo > cuoc_thang
    so_lan_khieu_nai = np.random.poisson(lam=1.5, size=n_samples)

    # Quy luật logic: Cước cao, thời gian gắn bó ngắn và khiếu nại nhiều -> khả năng Churn cao
    churn_prob = (
        0.35 * (cuoc_thang / 120.0) +
        0.45 * (so_lan_khieu_nai / 8.0) -
        0.40 * (so_thang / 72.0) +
        0.15 * (tuoi > 50).astype(int) +
        np.random.normal(0, 0.1, size=n_samples)
    )
    # Chuẩn hóa về xác suất [0, 1] qua hàm Sigmoid
    prob = 1 / (1 + np.exp(-3 * (churn_prob - 0.2)))
    churn = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        'Tuoi': tuoi,
        'So_thang_dung': so_thang,
        'Cuoc_thang': cuoc_thang.round(2),
        'Tong_chi_tieu': tong_chi_tieu.round(2),
        'So_lan_khieu_nai': so_lan_khieu_nai,
        'Churn': churn
    })
    return df

def get_prepared_data():
    """Hàm tiện ích chia dữ liệu chuẩn theo tỉ lệ 75% - 25% cho các bài tiếp theo"""
    df = generate_churn_dataset(n_samples=1000, random_state=42)
    feature_cols = ['Tuoi', 'So_thang_dung', 'Cuoc_thang', 'Tong_chi_tieu', 'So_lan_khieu_nai']
    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, df

def main():
    print("=" * 70)
    print("LAB 8 - BÀI 1: NẮM BẮT DỮ LIỆU VÀ TIỀN XỬ LÝ (CHURN PREDICTION)")
    print("=" * 70)

    X_train, X_test, y_train, y_test, df = get_prepared_data()

    print("\n1. 5 dòng đầu tiên của tập dữ liệu khách hàng:")
    print(df.head().to_string())

    print("\n2. Thống kê mô tả tổng quan:")
    print(df.describe().round(2).to_string())

    print("\n3. Tỷ lệ khách hàng rời mạng (Class Distribution):")
    churn_counts = df['Churn'].value_counts()
    print(f"   - Ở lại (0)  : {churn_counts[0]} khách hàng ({churn_counts[0]/len(df)*100:.1f}%)")
    print(f"   - Rời mạng (1): {churn_counts[1]} khách hàng ({churn_counts[1]/len(df)*100:.1f}%)")

    print("\n4. Phân chia Train/Test theo tỉ lệ 75% - 25%:")
    print(f"   - Kích thước tập Train: {X_train.shape[0]} mẫu (75%)")
    print(f"   - Kích thước tập Test : {X_test.shape[0]} mẫu (25%)")
    print("=" * 70)

if __name__ == "__main__":
    main()

