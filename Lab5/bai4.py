"""
Lab 5 - Bài 4: Xây dựng quy trình hoàn chỉnh - Lựa chọn mô hình tốt nhất
Yêu cầu:
- Viết một hàm evaluate_models(X_train, X_test, y_train, y_test) nhận vào dữ liệu đã chia.
- Huấn luyện và đánh giá đồng thời 3 mô hình:
  1. Logistic Regression
  2. K-Nearest Neighbors (KNN)
  3. Cây quyết định (Decision Tree)
- Trả về bảng kết quả so sánh độ chính xác và gợi ý mô hình tối ưu nhất.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from bai1 import get_iris_split

def evaluate_models(X_train, X_test, y_train, y_test):
    """
    Hàm huấn luyện và đánh giá đồng thời 3 mô hình phân loại:
    - Logistic Regression
    - K-Nearest Neighbors
    - Decision Tree
    Trả về DataFrame kết quả và tên mô hình tốt nhất.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "K-Nearest Neighbors (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree (Max Depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42)
    }

    records = []
    best_acc = -1.0
    best_model_name = ""

    for name, model in models.items():
        # Huấn luyện mô hình
        model.fit(X_train, y_train)

        # Đánh giá trên tập kiểm thử
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        records.append({
            "Tên mô hình": name,
            "Độ chính xác (Accuracy)": f"{acc * 100:.2f}%",
            "F1-Score (Weighted)": f"{f1:.4f}",
            "acc_raw": acc
        })

        if acc > best_acc:
            best_acc = acc
            best_model_name = name

    df_report = pd.DataFrame(records)
    return df_report, best_model_name, best_acc

def main():
    print("=" * 70)
    print("LAB 5 - BÀI 4: QUY TRÌNH TỰ ĐỘNG ĐÁNH GIÁ VÀ LỰA CHỌN MÔ HÌNH")
    print("=" * 70)

    # Lấy dữ liệu chia 7:3
    X_train, X_test, y_train, y_test, iris = get_iris_split()

    print("\n1. Thực hiện gọi hàm evaluate_models(X_train, X_test, y_train, y_test)...")
    df_results, best_name, best_score = evaluate_models(X_train, X_test, y_train, y_test)

    # Hiển thị bảng tổng hợp
    display_df = df_results.drop(columns=['acc_raw'])
    print("\n2. BẢNG SO SÁNH HIỆU NĂNG 3 MÔ HÌNH TRÊN TẬP TEST:")
    print(display_df.to_string(index=False))

    # Đưa ra kết luận lựa chọn
    print(f"\n3. KẾT LUẬN & ĐỀ XUẤT:")
    print(f"   -> Mô hình đạt hiệu quả tốt nhất: [{best_name}]")
    print(f"   -> Độ chính xác: {best_score * 100:.2f}%")
    print("   -> Lời khuyên: Trên tập dữ liệu Iris kích thước nhỏ, KNN và Decision Tree đều đạt hiệu quả")
    print("      rất cao, tuy nhiên Decision Tree có ưu thế vượt trội về khả năng giải thích quy luật (If-Else).")
    print("=" * 70)

if __name__ == "__main__":
    main()

