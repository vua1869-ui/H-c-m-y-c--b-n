"""
Lab 1 - Bài 2: Phân loại bài toán Machine Learning
Yêu cầu:
Dựa trên kiến thức đã học, hãy viết ghi chú ngắn (comment trong code) phân loại
các ví dụ sau thuộc nhóm Supervised (Có giám sát) hay Unsupervised (Không giám sát):
1. Dự đoán giá nhà dựa trên diện tích.
2. Gom nhóm khách hàng dựa trên hành vi mua sắm.
3. Phân loại email là Spam hay không.
4. Nhận diện chữ viết tay từ hình ảnh.
"""

def main():
    print("=" * 65)
    print("LAB 1 - BÀI 2: PHÂN LOẠI CÁC BÀI TOÁN MACHINE LEARNING")
    print("=" * 65)

    answers = [
        {
            "stt": 1,
            "bai_toan": "Dự đoán giá nhà dựa trên diện tích",
            "nhom": "Supervised Learning (Học có giám sát)",
            "loai": "Regression (Hồi quy)",
            "giai_thich": (
                "Dữ liệu huấn luyện có chứa nhãn đầu ra liên tục là 'giá nhà' tương ứng với 'diện tích'.\n"
                "Mô hình học mối quan hệ giữa đặc trưng (diện tích) và nhãn liên tục (giá bán)."
            )
        },
        {
            "stt": 2,
            "bai_toan": "Gom nhóm khách hàng dựa trên hành vi mua sắm",
            "nhom": "Unsupervised Learning (Học không giám sát)",
            "loai": "Clustering (Gom cụm)",
            "giai_thich": (
                "Dữ liệu chỉ có hành vi mua sắm (tần suất, số tiền, loại hàng...) mà KHÔNG có nhãn trước.\n"
                "Mục tiêu là tự động phát hiện cấu trúc tiềm ẩn để phân chia khách hàng thành các nhóm tương đồng."
            )
        },
        {
            "stt": 3,
            "bai_toan": "Phân loại email là Spam hay không",
            "nhom": "Supervised Learning (Học có giám sát)",
            "loai": "Classification - Binary Classification (Phân loại nhị phân)",
            "giai_thich": (
                "Dữ liệu huấn luyện gồm nội dung email kèm nhãn rõ ràng: 1 (Spam) hoặc 0 (Không Spam / Ham).\n"
                "Mô hình học để gán một mẫu mới vào một trong hai lớp cụ thể."
            )
        },
        {
            "stt": 4,
            "bai_toan": "Nhận diện chữ viết tay từ hình ảnh",
            "nhom": "Supervised Learning (Học có giám sát)",
            "loai": "Classification - Multi-class Classification (Phân loại đa lớp)",
            "giai_thich": (
                "Dữ liệu gồm ảnh các chữ số/chữ cái và nhãn tương ứng (ví dụ: nhãn từ 0 đến 9 trong tập Digits/MNIST).\n"
                "Mô hình học để nhận diện ảnh thuộc chữ số hoặc chữ cái nào."
            )
        }
    ]

    for item in answers:
        print(f"\n[{item['stt']}] Bài toán: {item['bai_toan']}")
        print(f"    -> Nhóm: {item['nhom']}")
        print(f"    -> Dạng bài: {item['loai']}")
        print(f"    -> Giải thích: {item['giai_thich']}")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()

