"""
Lab 1 - Bài 4: Quy trình ML hoàn chỉnh - Phần 1: Thu thập và Kiểm tra dữ liệu
Yêu cầu:
- Tải tập dữ liệu digits từ sklearn.datasets.
- In ra hình dạng (shape) của dữ liệu và nhãn.
- Hiển thị một mẫu dữ liệu dưới dạng hình ảnh (lưu file digit_sample.png) và in ma trận giá trị của mẫu đầu tiên.
"""

import os
import matplotlib
matplotlib.use('Agg') # Sử dụng backend không cần cửa sổ tương tác
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

def main():
    print("=" * 60)
    print("LAB 1 - BÀI 4: QUY TRÌNH ML HOÀN CHỈNH - PHẦN 1")
    print("THU THẬP VÀ KIỂM TRA DỮ LIỆU (DIGITS DATASET)")
    print("=" * 60)

    # 1. Tải tập dữ liệu digits
    digits = load_digits()
    X = digits.data
    y = digits.target
    images = digits.images

    # 2. In ra hình dạng (shape) của dữ liệu và nhãn
    print("\n1. Hình dạng (Shape) dữ liệu:")
    print(f"   - Ma trận đặc trưng 2D (X): {X.shape} (1797 mẫu, mỗi mẫu có 64 đặc trưng điểm ảnh)")
    print(f"   - Mảng hình ảnh 3D (images): {images.shape} (1797 ảnh kích thước 8x8)")
    print(f"   - Vector nhãn (y): {y.shape} (1797 nhãn từ 0 đến 9)")

    # 3. In giá trị ma trận của mẫu đầu tiên (ảnh 8x8)
    print("\n2. Ma trận giá trị điểm ảnh (pixel) của mẫu đầu tiên (kích thước 8x8):")
    print(images[0])
    print(f"   Nhãn thực tế của mẫu này: {y[0]}")

    # 4. Hiển thị và lưu hình ảnh mẫu đầu tiên
    save_path = os.path.join(os.path.dirname(__file__), "digit_sample.png")
    fig, axes = plt.subplots(1, 4, figsize=(10, 3))
    
    for i in range(4):
        axes[i].imshow(images[i], cmap='gray_r')
        axes[i].set_title(f"Nhãn: {y[i]}", fontsize=12)
        axes[i].axis('off')

    plt.suptitle("Lab 1 - Mẫu dữ liệu từ tập Scikit-Learn Digits", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n3. Đã vẽ và lưu hình ảnh 4 mẫu chữ số đầu tiên thành công:")
    print(f"   Đường dẫn: {save_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

