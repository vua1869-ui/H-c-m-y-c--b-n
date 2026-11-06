"""
Lab 5 - Bài 2: Gom cụm với K-Means (Unsupervised Learning)
Yêu cầu:
- Sử dụng tập dữ liệu make_blobs từ sklearn.datasets để tạo 3 cụm dữ liệu giả lập.
- Áp dụng thuật toán K-Means để gom dữ liệu thành 3 cụm (k=3).
- Trực quan hóa kết quả gom cụm bằng biểu đồ tán xạ (scatter plot), hiển thị các tâm cụm (centroids).
"""

import os
import tempfile
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def main():
    print("=" * 65)
    print("LAB 5 - BÀI 2: GOM CỤM VỚI K-MEANS (UNSUPERVISED LEARNING)")
    print("=" * 65)

    # 1. Tạo dữ liệu giả lập 3 cụm bằng make_blobs
    n_samples = 300
    n_clusters = 3
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=0.85,
        random_state=42
    )

    print(f"\n1. Khởi tạo dữ liệu giả lập:")
    print(f"   - Số lượng mẫu: {n_samples}")
    print(f"   - Số cụm thực tế: {n_clusters}")
    print(f"   - Kích thước tập dữ liệu X: {X.shape}")

    # 2. Huấn luyện mô hình K-Means gom cụm (không sử dụng nhãn y_true)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    print("\n2. Kết quả gom cụm K-Means:")
    print(f"   - Số lần lặp để hội tụ: {kmeans.n_iter_}")
    print(f"   - Tọa độ các tâm cụm (Centroids):\n{centroids}")
    print(f"   - Quán tính gom cụm (Inertia - SSE): {kmeans.inertia_:.2f}")

    # 3. Trực quan hóa kết quả gom cụm
    save_path = os.path.join(os.path.dirname(__file__), "kmeans_clusters.png")
    plt.figure(figsize=(8, 6))

    # Vẽ các điểm dữ liệu theo màu của từng cụm
    scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.65, edgecolors='k', s=45)
    
    # Vẽ các tâm cụm (Centroids) bằng dấu X lớn màu đỏ
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        c='red', marker='X', s=250, linewidths=2, edgecolors='black',
        label='Tâm cụm (Centroids)'
    )

    plt.title("Gom cụm K-Means (K = 3) với Make_Blobs", fontsize=13, fontweight='bold')
    plt.xlabel("Đặc trưng X1", fontsize=11)
    plt.ylabel("Đặc trưng X2", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n3. Đã vẽ biểu đồ gom cụm và lưu tại: {save_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()

