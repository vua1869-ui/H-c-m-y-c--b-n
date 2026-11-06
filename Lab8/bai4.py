"""
Lab 8 - Bài 4: Chương trình hoàn chỉnh & Xuất mô hình (Model Persistence)
Yêu cầu:
- Kết hợp Bài 1, 2, 3 thành một kịch bản hoàn chỉnh (Code có comment chi tiết).
- Sử dụng thư viện 'joblib' để lưu Pipeline đã tối ưu ra file '.pkl'.
- Nạp lại (Load) mô hình từ file '.pkl' và thực hiện dự đoán với mẫu dữ liệu khách hàng mới.
"""

import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from bai1 import generate_churn_dataset

def main():
    print("=" * 75)
    print("LAB 8 - BÀI 4: QUY TRÌNH HOÀN CHỈNH & XUẤT LƯU TRỮ MÔ HÌNH (JOBLIB)")
    print("=" * 75)

    # -------------------------------------------------------------
    # BƯỚC 1: THU THẬP VÀ TIỀN XỬ LÝ DỮ LIỆU
    # -------------------------------------------------------------
    print("\n[BƯỚC 1] Thu thập và Tiền xử lý dữ liệu...")
    df = generate_churn_dataset(n_samples=1000, random_state=42)
    feature_cols = ['Tuoi', 'So_thang_dung', 'Cuoc_thang', 'Tong_chi_tieu', 'So_lan_khieu_nai']
    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"   - Dữ liệu hoàn chỉnh: {len(df)} mẫu")
    print(f"   - Huấn luyện: {len(X_train)} mẫu | Kiểm thử: {len(X_test)} mẫu")

    # -------------------------------------------------------------
    # BƯỚC 2: XÂY DỰNG PIPELINE VÀ TỐI ƯU SIÊU THAM SỐ
    # -------------------------------------------------------------
    print("\n[BƯỚC 2] Xây dựng Pipeline và Tinh chỉnh siêu tham số...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 8, None],
        'classifier__min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        refit=True # Tự động fit mô hình tốt nhất trên toàn bộ Train set
    )
    grid_search.fit(X_train, y_train)

    print(f"   - Siêu tham số tối ưu: {grid_search.best_params_}")
    print(f"   - Điểm số Cross-validation trên Train: {grid_search.best_score_ * 100:.2f}%")

    # -------------------------------------------------------------
    # BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET)
    # -------------------------------------------------------------
    print("\n[BƯỚC 3] Đánh giá mô hình trên tập Test...")
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   -> Độ chính xác kiểm thử: {test_acc * 100:.2f}%")
    print("\nBáo cáo phân loại tổng kết:")
    print(classification_report(y_test, y_pred, target_names=['Ở lại (0)', 'Rời mạng (1)']))

    # -------------------------------------------------------------
    # BƯỚC 4: XUẤT VÀ LƯU MÔ HÌNH RA FILE .PKL BẰNG JOBLIB
    # -------------------------------------------------------------
    print("\n[BƯỚC 4] Lưu trữ mô hình ra file .pkl...")
    model_file_path = os.path.join(os.path.dirname(__file__), "churn_model.pkl")
    joblib.dump(best_pipeline, model_file_path)
    print(f"   -> Đã xuất mô hình thành công vào: {model_file_path}")
    print(f"   -> Kích thước file: {os.path.getsize(model_file_path) / 1024:.2f} KB")

    # -------------------------------------------------------------
    # BƯỚC 5: NẠP LẠI (LOAD) MÔ HÌNH VÀ DỰ ĐOÁN MẪU MỚI
    # -------------------------------------------------------------
    print("\n[BƯỚC 5] Nạp lại mô hình từ đĩa và dự đoán khách hàng mới...")
    loaded_model = joblib.load(model_file_path)

    # Giả lập 2 khách hàng mới tinh chưa từng có trong tập dữ liệu:
    # Khách 1: Trẻ, gắn bó lâu (50 tháng), cước thấp ($30), 0 lần khiếu nại -> Kỳ vọng: Ở lại (0)
    # Khách 2: Lớn tuổi, mới dùng (2 tháng), cước cao ($110), 5 lần khiếu nại -> Kỳ vọng: Rời mạng (1)
    new_customers = pd.DataFrame([
        {
            'Tuoi': 28,
            'So_thang_dung': 50,
            'Cuoc_thang': 35.0,
            'Tong_chi_tieu': 1750.0,
            'So_lan_khieu_nai': 0
        },
        {
            'Tuoi': 55,
            'So_thang_dung': 3,
            'Cuoc_thang': 115.0,
            'Tong_chi_tieu': 345.0,
            'So_lan_khieu_nai': 5
        }
    ])

    print("\nThông tin 2 khách hàng mới cần thẩm định:")
    print(new_customers)

    # Dự đoán bằng mô hình đã nạp từ file .pkl
    predictions = loaded_model.predict(new_customers)
    probabilities = loaded_model.predict_proba(new_customers)

    labels = {0: "Ở LẠI DỊCH VỤ", 1: "CÓ NGUY CƠ RỜI MẠNG"}

    print("\nKẾT QUẢ DỰ BÁO TỪ MÔ HÌNH PRODUCTION:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        churn_risk = prob[1] * 100
        print(f"   - Khách hàng #{i+1}: Dự đoán = [{labels[pred]}] | Tỷ lệ rủi ro rời mạng: {churn_risk:.2f}%")

    print("\n" + "=" * 75)

if __name__ == "__main__":
    main()

