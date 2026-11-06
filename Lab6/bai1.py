"""
Lab 6 - Bài 1: Nhận biết Overfitting và Underfitting (Bias vs Variance)
Yêu cầu:
- Tạo một tập dữ liệu giả lập (ví dụ: hàm sin có nhiễu).
- Huấn luyện 3 mô hình Linear Regression kết hợp Polynomial Features với các bậc khác nhau:
  + Bậc 1: Underfitting (Thiếu khớp / Bias cao)
  + Bậc 4: Good fit (Khớp tốt / Cân bằng Bias-Variance)
  + Bậc 15: Overfitting (Quá khớp / Variance cao)
- Vẽ biểu đồ hiển thị đường dự đoán của 3 mô hình cùng với dữ liệu thực tế.
- Giải thích chi tiết hiện tượng Bias cao và Variance cao.
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def true_func(X):
    return np.cos(1.5 * np.pi * X)

def main():
    print("=" * 70)
    print("LAB 6 - BÀI 1: NHẬN BIẾT OVERFITTING VÀ UNDERFITTING")
    print("=" * 70)

    # 1. Tạo tập dữ liệu giả lập có nhiễu
    np.random.seed(0)
    n_samples = 30
    X = np.sort(np.random.rand(n_samples))
    y = true_func(X) + np.random.randn(n_samples) * 0.15

    X_seq = np.linspace(0, 1, 200)

    degrees = [1, 4, 15]
    titles = [
        "Bậc 1: Underfitting (High Bias)",
        "Bậc 4: Good Fit (Balanced)",
        "Bậc 15: Overfitting (High Variance)"
    ]

    save_path = os.path.join(os.path.dirname(__file__), "overfitting_underfitting.png")
    plt.figure(figsize=(15, 5))

    print("\n1. Huấn luyện các mô hình đa thức và tính MSE:")

    for i, deg in enumerate(degrees):
        ax = plt.subplot(1, len(degrees), i + 1)

        # Xây dựng Pipeline: Polynomial Features -> Linear Regression
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        linear = LinearRegression()
        pipe = Pipeline([("poly", poly), ("linear", linear)])
        
        pipe.fit(X[:, np.newaxis], y)
        y_pred_train = pipe.predict(X[:, np.newaxis])
        y_pred_seq = pipe.predict(X_seq[:, np.newaxis])
        mse = mean_squared_error(y, y_pred_train)

        print(f"   - {titles[i]}: Training MSE = {mse:.4f}")

        # Vẽ đồ thị
        plt.plot(X_seq, y_pred_seq, label="Đường dự đoán", color='crimson', lw=2)
        plt.plot(X_seq, true_func(X_seq), label="Hàm thực tế gốc", color='navy', linestyle='--', lw=1.5)
        plt.scatter(X, y, edgecolors='b', s=25, label="Mẫu dữ liệu thực", alpha=0.7)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best", fontsize=9)
        plt.title(f"{titles[i]}\n(Train MSE = {mse:.3f})", fontsize=11, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n2. Đã lưu biểu đồ trực quan hóa tại: {save_path}")

    # 2. Giải thích chi tiết hiện tượng Bias và Variance
    print("\n3. GIẢI THÍCH CHI TIẾT BIAS CAO VÀ VARIANCE CAO:")
    print("""
   a) Bias cao (High Bias - Hiện tượng Underfitting / Thiếu khớp):
      - Biểu hiện ở Bậc 1: Mô hình đường thẳng quá đơn giản, không đủ năng lực biểu diễn dạng phi tuyến của dữ liệu.
      - Đặc điểm: Sai số cao trên cả tập huấn luyện (Train) lẫn tập kiểm thử (Test). Mô hình bỏ qua các quy luật cốt lõi.

   b) Variance cao (High Variance - Hiện tượng Overfitting / Quá khớp):
      - Biểu hiện ở Bậc 15: Mô hình đường cong uốn lượn dữ dội để đi qua từng điểm dữ liệu huấn luyện (học vẹt cả nhiễu).
      - Đặc điểm: Sai số trên tập huấn luyện cực thấp (gần bằng 0), nhưng khi gặp dữ liệu mới thì sai số cực lớn.
      - Khả năng khái quát hóa (Generalization) rất kém.

   c) Good Fit (Mô hình tối ưu - Đạt Trade-off cân bằng giữa Bias và Variance):
      - Biểu hiện ở Bậc 4: Bắt trọn xu hướng cốt lõi của hàm sin/cos mà không bị nhiễu làm sai lệch.
    """)
    print("=" * 70)

if __name__ == "__main__":
    main()

