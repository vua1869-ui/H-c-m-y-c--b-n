"""
Lab 2 - Bài 1: Làm sạch dữ liệu (Data Cleaning)
Yêu cầu:
- Tạo một DataFrame chứa dữ liệu bị thiếu (NaN) và có giá trị ngoại lệ (outlier).
- Sử dụng Pandas để:
  1. Điền các giá trị thiếu bằng giá trị trung bình (mean) của cột.
  2. Phát hiện và xử lý giá trị ngoại lệ đơn giản (ví dụ: các giá trị lớn hơn 3 lần độ lệch chuẩn so với giá trị trung bình).
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 65)
    print("LAB 2 - BÀI 1: LÀM SẠCH DỮ LIỆU (DATA CLEANING)")
    print("=" * 65)

    # 1. Tạo DataFrame mẫu chứa giá trị khuyết thiếu (NaN) và giá trị ngoại lệ (outlier)
    # Giả lập dữ liệu điểm thi và tuổi của học viên
    np.random.seed(42)
    ages = [20, 21, 22, np.nan, 21, 23, 20, np.nan, 22, 21, 20, 22, 23, 85, 21] # 85 là outlier
    scores = [7.5, 8.0, np.nan, 7.0, 8.5, 9.0, np.nan, 6.5, 8.0, 7.5, 8.0, 25.0, 7.0, 8.5, 9.0] # 25.0 là outlier
    names = [f"Học viên {i+1}" for i in range(len(ages))]

    df = pd.DataFrame({
        'Ho_ten': names,
        'Tuoi': ages,
        'Diem': scores
    })
    print("\n1. Dữ liệu ban đầu (chứa NaN và Outliers):")
    print(df)

    # 2. Xử lý giá trị thiếu (Missing values - NaN) bằng giá trị trung bình (Mean)
    print("\n2. Điền giá trị thiếu (NaN) bằng Mean của từng cột số:")
    numeric_cols = ['Tuoi', 'Diem']
    df_imputed = df.copy()

    for col in numeric_cols:
        col_mean = df_imputed[col].mean()
        df_imputed[col] = df_imputed[col].fillna(col_mean)
        print(f"   - Cột '{col}': Mean = {col_mean:.2f} -> đã điền vào các vị trí NaN.")

    print("\nDataFrame sau khi điền giá trị thiếu:")
    print(df_imputed)

    # 3. Phát hiện và xử lý ngoại lệ theo quy tắc 3-Sigma (|x - mean| > 3 * std)
    print("\n3. Phát hiện và xử lý giá trị ngoại lệ (Outlier > 3*std):")
    df_cleaned = df_imputed.copy()

    for col in numeric_cols:
        mean_val = df_cleaned[col].mean()
        std_val = df_cleaned[col].std()
        lower_bound = mean_val - 2.5 * std_val
        upper_bound = mean_val + 2.5 * std_val

        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        print(f"\n   Cột '{col}':")
        print(f"     Mean = {mean_val:.2f}, Std = {std_val:.2f}")
        print(f"     Ngưỡng hợp lệ: [{lower_bound:.2f}, {upper_bound:.2f}]")

        if len(outliers) > 0:
            print(f"     -> Phát hiện {len(outliers)} ngoại lệ:")
            for idx, val in outliers[col].items():
                print(f"        Dòng {idx} ({df_cleaned.loc[idx, 'Ho_ten']}): giá trị = {val}")
            
            # Xử lý ngoại lệ: thay thế bằng giá trị Median
            median_val = df_cleaned[col].median()
            df_cleaned.loc[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound), col] = median_val
            print(f"     -> Đã xử lý: Thay thế giá trị ngoại lệ bằng Median ({median_val:.2f})")
        else:
            print("     -> Không có ngoại lệ vượt ngưỡng.")

    print("\nDataFrame hoàn chỉnh sau khi làm sạch toàn diện:")
    print(df_cleaned)
    print("=" * 65)

if __name__ == "__main__":
    main()
