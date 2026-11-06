"""
Lab 5 - Bài 3: Phân loại với Cây quyết định (Decision Tree)
Yêu cầu:
- Sử dụng lại tập dữ liệu 'iris' từ Bài 1 (tỉ lệ 7:3).
- Huấn luyện mô hình Cây quyết định (DecisionTreeClassifier).
- So sánh độ chính xác của Cây quyết định với mô hình KNN ở Bài 1.
- In ra báo cáo phân loại (classification report) cho Cây quyết định.
- Nhận xét chi tiết so sánh ưu/nhược điểm giữa KNN và Cây quyết định trong thực tế.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from bai1 import get_iris_split

def main():
    print("=" * 70)
    print("LAB 5 - BÀI 3: PHÂN LOẠI VỚI CÂY QUYẾT ĐỊNH (DECISION TREE)")
    print("=" * 70)

    # Lấy dữ liệu Iris chia 7:3 từ Bài 1
    X_train, X_test, y_train, y_test, iris = get_iris_split()

    # 1. Huấn luyện mô hình Decision Tree
    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # 2. Huấn luyện lại KNN để đối chiếu trực tiếp
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    # 3. So sánh độ chính xác giữa hai thuật toán
    print("\n1. SO SÁNH ĐỘ CHÍNH XÁC (ACCURACY) TRÊN TẬP TEST:")
    print(f"   - K-Nearest Neighbors (KNN - k=5)  : {acc_knn * 100:.2f}%")
    print(f"   - Decision Tree Classifier         : {acc_dt * 100:.2f}%")

    # 4. Báo cáo phân loại cho Cây quyết định
    print("\n2. BÁO CÁO PHÂN LOẠI CỦA CÂY QUYẾT ĐỊNH (CLASSIFICATION REPORT):")
    print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

    # 5. Nhận xét so sánh ưu/nhược điểm chuyên sâu
    print("3. NHẬN XÉT SO SÁNH ƯU VÀ NHƯỢC ĐIỂM TRONG ỨNG DỤNG THỰC TẾ:")
    print("""
   a) Cây quyết định (Decision Tree):
      - Ưu điểm:
        + Tính dễ hiểu (White-box model): Con người có thể xem toàn bộ các quy tắc If-Else,
          rất thích hợp cho các bài toán kinh doanh, y tế cần giải trình lý do ra quyết định.
        + Tốc độ dự đoán cực nhanh (O(chiều sâu cây)).
        + Không yêu cầu chuẩn hóa thang đo dữ liệu (Feature Scaling).
      - Nhược điểm: Rất dễ bị hiện tượng quá khớp (Overfitting) nếu cây quá sâu. Cần cắt tỉa (pruning).

   b) K-Nearest Neighbors (KNN):
      - Ưu điểm:
        + Bản chất đơn giản, không cần giai đoạn huấn luyện phức tạp (Lazy Learner).
        + Hoạt động tốt với ranh giới phân lớp phi tuyến tính phức tạp.
      - Nhược điểm:
        + Tốc độ dự đoán chậm khi dữ liệu lớn vì phải tính khoảng cách tới toàn bộ tập huấn luyện.
        + Tốn bộ nhớ vì phải lưu trữ toàn bộ tập train.
        + Rất nhạy cảm với thang đo đặc trưng (bắt buộc phải chuẩn hóa) và nhiễu (outliers).
    """)
    print("=" * 70)

if __name__ == "__main__":
    main()

