"""
Lab 7 - Bài 3: Chuẩn bị dữ liệu cho Mini Project (Titanic Dataset)
Yêu cầu:
- Sử dụng pandas để tải tập dữ liệu phân loại Titanic từ URL công khai:
  'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
  (Có cơ chế fallback tự động nạp từ file titanic.csv nội bộ nếu không có kết nối internet).
- Hiển thị 5 dòng đầu tiên.
- Kiểm tra thông tin kiểu dữ liệu (.info()) và tóm tắt thống kê (.describe()).
- Nhận diện các cột có giá trị khuyết thiếu và tỷ lệ phần trăm thiếu.
"""

import os
import pandas as pd

def load_titanic_dataset():
    """Tải tập dữ liệu Titanic từ URL trực tuyến, nếu lỗi mạng thì đọc file cục bộ"""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    local_path = os.path.join(os.path.dirname(__file__), "titanic.csv")

    try:
        print(f"-> Đang thử tải từ URL: {url}...")
        df = pd.read_csv(url)
        print("-> Tải thành công trực tuyến từ GitHub!")
    except Exception as e:
        print(f"-> Không thể kết nối Internet ({e}). Chuyển sang đọc file dự phòng cục bộ: {local_path}")
        df = pd.read_csv(local_path)
        print("-> Đã tải thành công từ file titanic.csv nội bộ!")
    return df

def main():
    print("=" * 70)
    print("LAB 7 - BÀI 3: CHUẨN BỊ VÀ KHÁM PHÁ DỮ LIỆU TITANIC (EDA)")
    print("=" * 70)

    # 1. Tải tập dữ liệu Titanic
    df = load_titanic_dataset()

    # 2. Hiển thị 5 dòng đầu tiên
    print("\n1. 5 DÒNG ĐẦU TIÊN CỦA TẬP DỮ LIỆU TITANIC (head(5)):")
    print(df.head(5).to_string())

    # 3. Kiểm tra thông tin cấu trúc dữ liệu (.info())
    print("\n2. THÔNG TIN CẤU TRÚC VÀ KIỂU DỮ LIỆU (.info()):")
    df.info()

    # 4. Tóm tắt thống kê mô tả (.describe())
    print("\n3. TÓM TẮT THỐNG KÊ CÁC ĐẶC TRƯNG SỐ (.describe()):")
    print(df.describe().round(2).to_string())

    # 5. Nhận diện các cột có giá trị khuyết thiếu (Missing Values)
    print("\n4. NHẬN DIỆN CÁC CỘT CÓ GIÁ TRỊ KHUYẾT THIẾU (MISSING VALUES):")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Số ô thiếu': missing_count,
        'Tỷ lệ thiếu (%)': missing_percent
    })
    # Lọc các cột có giá trị thiếu
    missing_cols = missing_summary[missing_summary['Số ô thiếu'] > 0]
    
    if len(missing_cols) > 0:
        print(missing_cols.to_string())
        print("\nNhận xét về dữ liệu khuyết thiếu:")
        for col in missing_cols.index:
            print(f"   - Cột '{col}': Thiếu {missing_summary.loc[col, 'Số ô thiếu']} giá trị ({missing_summary.loc[col, 'Tỷ lệ thiếu (%)']}%).")
            if col == 'Age':
                print("     -> Hướng xử lý: Điền bằng trung vị (median) theo nhóm hành khách.")
            elif col == 'Cabin':
                print("     -> Hướng xử lý: Tỷ lệ thiếu rất cao, có thể bỏ cột hoặc tạo cờ Has_Cabin (0/1).")
            elif col == 'Embarked':
                print("     -> Hướng xử lý: Điền bằng giá trị xuất hiện nhiều nhất (mode - 'S').")
    else:
        print("Tập dữ liệu không có giá trị khuyết thiếu.")

    print("=" * 70)

if __name__ == "__main__":
    main()

