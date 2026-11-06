"""
Lab 1 - Bài 1: Làm quen với Numpy và Pandas
Mục tiêu:
- Tạo mảng Numpy 1 chiều từ 1 đến 10.
- Tạo DataFrame Pandas chứa thông tin 5 sinh viên: Họ tên, Tuổi, Điểm toán.
- Hiển thị thông tin tổng quan và tính điểm trung bình môn toán.
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 60)
    print("LAB 1 - BÀI 1: LÀM QUEN VỚI NUMPY VÀ PANDAS")
    print("=" * 60)

    # 1. Tạo mảng Numpy 1 chiều chứa các số từ 1 đến 10
    numpy_array = np.arange(1, 11)
    print("\n1. Mảng Numpy 1 chiều từ 1 đến 10:")
    print("   Dữ liệu:", numpy_array)
    print("   Kiểu dữ liệu:", type(numpy_array))
    print("   Shape (kích thước):", numpy_array.shape)

    # 2. Tạo DataFrame từ thư viện Pandas chứa thông tin 5 sinh viên
    data = {
        'Họ tên': ['Nguyễn Văn An', 'Trần Thị Bình', 'Lê Hoàng Cường', 'Phạm Minh Đức', 'Võ Thị Kim Em'],
        'Tuổi': [19, 20, 19, 21, 20],
        'Điểm toán': [8.5, 7.0, 9.2, 6.5, 8.8]
    }
    df_students = pd.DataFrame(data)

    print("\n2. Danh sách sinh viên (DataFrame):")
    print(df_students)

    # 3. Hiển thị thông tin tổng quan của DataFrame
    print("\n3. Thông tin tổng quan của DataFrame (info()):")
    df_students.info()

    print("\nThống kê mô tả (describe()):")
    print(df_students.describe())

    # 4. Tính điểm trung bình môn toán
    avg_math = df_students['Điểm toán'].mean()
    print(f"\n4. Điểm trung bình môn toán của 5 sinh viên: {avg_math:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

