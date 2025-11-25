# Phương pháp nhận diện khuôn mặt

Pipeline gồm hai bước chính:

1. Xây dựng database khuôn mặt (embeddings) từ ảnh tĩnh.
2. Nhận diện khuôn mặt realtime từ webcam/video bằng cách so khớp với database đó.

### 1. Xây dựng database (enrollment)

Script: `enroll_faces.py`

#### 1.1. Cấu trúc dữ liệu ảnh

Ảnh được tổ chức dạng thư mục như sau:

```text
data/
  images/
    <folder_nguoi_1>/
      img1.jpg
      img2.jpg
      ...
    <folder_nguoi_2>/
      img1.jpg
      ...
```

- Mỗi thư mục con trong `data/images` tương ứng với **một người** (có thể là học sinh, giáo viên, nhân viên, v.v.).
- Bên trong là nhiều ảnh khuôn mặt của người đó.

#### 1.2. Trích xuất embedding với InsightFace

Đối với từng thư mục người:

- Đọc lần lượt các ảnh khuôn mặt.
- Với mỗi ảnh:
  - Dùng `FaceAnalysis` (InsightFace) để phát hiện mặt trong ảnh.
  - Chọn khuôn mặt có diện tích lớn nhất (nếu có nhiều mặt).
  - Lấy vector đặc trưng khuôn mặt `face.embedding`.
- Tập hợp tất cả embedding của người đó và **lấy trung bình** → một vector đại diện duy nhất.
- Gom các vector đại diện của mọi người lại thành mảng `embeddings.npy`.

#### 1.3. Chuẩn hoá nhãn (label)

Tên thư mục ban đầu chính là tên của một người, ví dụ: `NguyenNgocPhuTy`.

Khi enroll, nhãn được lưu vào `labels.npy` đúng như tên thư mục (chỉ bỏ phần đuôi file nếu có).

Khi hiển thị, hàm `format_label` trong `recognize_realtime.py` sẽ chèn khoảng trắng dựa trên chữ cái in hoa và ký tự `_`, ví dụ:

- `NguyenNgocPhuTy` → `Nguyen Ngoc Phu Ty`
- `Nguyen_Ngoc_Phu_Ty` → `Nguyen Ngoc Phu Ty`

### 2. Nhận diện khuôn mặt realtime

Script: `recognize_realtime.py`

#### 2.1. Tải và chuẩn hoá database

- Đọc `embeddings.npy` (vector đặc trưng) và `labels.npy` (tên người) từ thư mục embeddings.
- Chuẩn hoá từng embedding theo chuẩn L2 để có vector đơn vị. Khi đó **tích vô hướng** giữa hai vector xấp xỉ **cosine similarity**.

#### 2.2. Mô hình được sử dụng

**InsightFace – `FaceAnalysis`** Chạy trên GPU thông qua `onnxruntime-gpu` (nếu có), hoặc trên CPU nếu không có GPU.

#### 2.3. Xử lý mỗi frame

Với mỗi frame lấy từ webcam/video:

1. Đọc frame bằng OpenCV.
2. Gọi `FaceAnalysis.get(frame)` để phát hiện tất cả khuôn mặt trong frame.
3. Với mỗi khuôn mặt:
  - Lấy vector đặc trưng khuôn mặt `face.embedding`.
  - Chuẩn hoá embedding theo chuẩn L2.
  - Tính similarity với toàn bộ embedding trong database bằng tích vô hướng.
  - Tìm người có độ tương đồng cao nhất và giá trị similarity tương ứng.
  - Nếu similarity ≥ ngưỡng `threshold` → coi là nhận diện được, hiển thị tên; nếu không thì hiển thị `Unknown`.
4. Vẽ khung chữ nhật quanh khuôn mặt và text (tên + similarity) lên frame.
5. Hiển thị frame bằng OpenCV, nhấn phím `q` để thoát.

#### 2.4. Tối ưu bước so khớp (FAISS – tuỳ chọn)

- Về mặt thuật toán, bước so khớp vẫn là **tìm người có độ tương đồng cao nhất** giữa embedding khuôn mặt hiện tại và các embedding trong database (dựa trên tích vô hướng/cosine similarity).
- Để tăng tốc khi số lượng người lớn, code cho phép dùng **FAISS** (file `faiss_index.py`) để xây dựng index tìm kiếm nhanh:
  - Các vector trong `embeddings.npy` sau khi chuẩn hoá được truyền vào hàm `build_faiss_index` để tạo index `IndexFlatIP`.
  - Ở bước 3 bên trên, thay vì luôn luôn tính toàn bộ tích vô hướng bằng NumPy, code gọi hàm `search_embedding`:
    - Nếu FAISS được import và khởi tạo thành công → dùng `index.search` của FAISS để tìm **top-1** vector gần nhất.
    - Nếu FAISS không có hoặc lỗi khởi tạo → **tự động fallback** về tính toán brute-force bằng NumPy đúng như mô tả ở trên.

### 3. Ý nghĩa của repo này

- Một ví dụ **cơ bản** và dễ hiểu về cách dùng InsightFace để nhận diện khuôn mặt realtime.
- Phù hợp để làm nền tảng cho các demo, bài tập, hoặc dự án nhỏ.

- **Không phải** Một hệ thống điểm danh/nhận diện sản phẩm hoàn chỉnh (chưa có cơ sở dữ liệu thật, logging, UI, bảo mật, v.v.).

Bạn có thể mở rộng dựa trên phương pháp lõi này (thêm lưu log, điểm danh, giao diện, tích hợp server, v.v.), nhưng ý tưởng trung tâm vẫn là pipeline đơn giản: **database embeddings từ ảnh tĩnh + so khớp realtime trên video**.
