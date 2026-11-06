"""
Lab 5 - Bài 5: Giảng viên giao thêm (Mở rộng nâng cao)
Đề tài:
1. Tìm giá trị K tối ưu cho thuật toán KNN (Accuracy vs K Curve).
2. Trực quan hóa cấu trúc đồ thị cây quyết định bằng plot_tree và lưu ảnh.
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from bai1 import get_iris_split

def main():
    print("=" * 75)
    print("LAB 5 - BÀI 5: GIẢNG VIÊN GIAO THÊM (MỞ RỘNG)")
    print("TÌM K TỐI ƯU CHO KNN & TRỰC QUAN HÓA CẤU TRÚC CÂY QUYẾT ĐỊNH")
    print("=" * 75)

    X_train, X_test, y_train, y_test, iris = get_iris_split()

    # 1. Tìm giá trị K tối ưu cho KNN từ 1 đến 20
    k_range = list(range(1, 21))
    accuracies = []

    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        accuracies.append(acc)

    best_k = k_range[np.argmax(accuracies)]
    best_acc = max(accuracies)

    print(f"\n1. Phân tích giá trị K tối ưu cho KNN:")
    print(f"   - K tốt nhất được tìm thấy: k = {best_k}")
    print(f"   - Độ chính xác tương ứng: {best_acc * 100:.2f}%")

    # 2. Vẽ đồ thị Accuracy theo K
    knn_plot_path = os.path.join(os.path.dirname(__file__), "knn_accuracy_vs_k.png")
    plt.figure(figsize=(8, 4.5))
    plt.plot(k_range, [a * 100 for a in accuracies], marker='o', color='purple', lw=2)
    plt.title("Độ chính xác mô hình KNN theo số lượng láng giềng (K)", fontsize=13, fontweight='bold')
    plt.xlabel("Giá trị K (n_neighbors)", fontsize=11)
    plt.ylabel("Độ chính xác Accuracy (%)", fontsize=11)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(knn_plot_path, dpi=150)
    plt.close()
    print(f"   -> Đã lưu biểu đồ Accuracy vs K tại: {knn_plot_path}")

    # 3. Trực quan hóa cấu trúc Cây quyết định (Decision Tree Plot)
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)

    tree_plot_path = os.path.join(os.path.dirname(__file__), "decision_tree_visualization.png")
    plt.figure(figsize=(14, 8))
    plot_tree(
        dt_model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Trực quan hóa cấu trúc cây quyết định (Decision Tree Rules)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(tree_plot_path, dpi=150)
    plt.close()

    print(f"\n2. Trực quan hóa cây quyết định:")
    print(f"   -> Đã lưu hình ảnh cấu trúc cây tại: {tree_plot_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()

