"""
Lab 7 - Bài 1: Sử dụng Random Forest Classifier (Học kết hợp - Ensemble Learning)
Yêu cầu:
- Tải tập dữ liệu 'wine' từ sklearn.datasets.
- Chia dữ liệu thành tập Train và Test (tỷ lệ 80/20).
- Khởi tạo và huấn luyện hai mô hình:
  + DecisionTreeClassifier (Cây quyết định đơn lẻ)
  + RandomForestClassifier (Rừng ngẫu nhiên - tập hợp nhiều cây)
- Dự đoán và tính toán độ chính xác (accuracy) của cả hai mô hình trên tập Test.
- So sánh và giải thích kết quả.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_wine_split():
    """Hàm tiện ích phân chia dữ liệu Wine theo tỷ lệ 80/20"""
    wine = load_wine()
    X = wine.data
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, wine

def main():
    print("=" * 70)
    print("LAB 7 - BÀI 1: SO SÁNH DECISION TREE VÀ RANDOM FOREST CLASSIFIER")
    print("=" * 70)

    # 1. Tải và chia dữ liệu Wine
    X_train, X_test, y_train, y_test, wine = get_wine_split()

    print(f"\n1. Thông tin tập dữ liệu Wine:")
    print(f"   - Tổng số mẫu: {wine.data.shape[0]}, Số đặc trưng: {wine.data.shape[1]}")
    print(f"   - Train set: {X_train.shape[0]} mẫu, Test set: {X_test.shape[0]} mẫu")

    # 2. Huấn luyện Decision Tree (Mô hình đơn lẻ)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # 3. Huấn luyện Random Forest (Mô hình học kết hợp Ensemble)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # 4. So sánh kết quả độ chính xác
    print("\n2. KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f"   - Decision Tree (Cây đơn lẻ)    : {acc_dt * 100:.2f}%")
    print(f"   - Random Forest (100 cây kết hợp): {acc_rf * 100:.2f}%")
    diff = (acc_rf - acc_dt) * 100
    print(f"   -> Mức chênh lệch cải thiện     : {diff:+.2f}%")

    # 5. Giải thích lý thuyết
    print("\n3. TẠI SAO RANDOM FOREST THƯỜNG VƯỢT TRỘI HƠN DECISION TREE?")
    print("""
   - Cơ chế Bagging (Bootstrap Aggregating): Random Forest tạo ra hàng trăm cây quyết định
     được huấn luyện trên các tập mẫu con ngẫu nhiên (có hoàn lại).
   - Chọn lọc đặc trưng ngẫu nhiên (Feature Subsampling): Tại mỗi điểm chia nhánh, chỉ một nhóm
     ngẫu nhiên các đặc trưng được xem xét, giúp giảm thiểu sự tương quan giữa các cây.
   - Nguyên lý 'Trí tuệ tập thể' (Wisdom of the Crowd): Kết hợp dự đoán bằng biểu quyết đa số (Majority Voting)
     giúp triệt tiêu phương sai (Variance), hạn chế triệt để hiện tượng Overfitting của từng cây đơn lẻ.
    """)
    print("=" * 70)

if __name__ == "__main__":
    main()

