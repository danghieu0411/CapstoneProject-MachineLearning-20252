# Kế hoạch và Nhật ký Công việc (Walkthrough)

Tài liệu này ghi lại những gì đã thực hiện và các bước tiếp theo cần hoàn thành để xây dựng báo cáo cuối cùng.

## 1. Công việc đã hoàn thành (Done)
- [x] Đọc và hiểu dàn ý báo cáo trong [Report_Guide.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/Report_Guide.md).
- [x] Đọc và tóm tắt tài liệu kỹ thuật của 5 mô hình:
  - [ANN.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/ANN/ANN.md)
  - [CNN-Resnet.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/CNN-ResNet/CNN-Resnet.md)
  - [CNN-FromScratch.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/CNN-FromScratch/CNN-FromScratch.md)
  - [HOG_SVM.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/HOG-SVM/HOG_SVM.md)
  - [SIFT-HOVW-RF.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/SIFT-HOVW-RF/SIFT-HOVW-RF.md)
- [x] Cập nhật các nguồn tài liệu cần tuân theo vào file chỉ dẫn [AGENTS.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/AGENTS.md).

## 2. Công việc cần làm tiếp theo (To-Do List)
Chúng ta cần xây dựng nội dung cho báo cáo cuối cùng tại [Report.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/Report.md) theo các bước:

- [x] **Bước 1: Khởi tạo cấu trúc Report**
  - Tạo khung đề mục chính cho [Report.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/Report.md) dựa theo [Report_Guide.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/Report_Guide.md).
- [x] **Bước 2: Viết nội dung Phần I (Introduction) & Phần II (Dataset)**
  - [x] Giới thiệu bài toán phân loại chó/mèo và liệt kê các model.
  - [x] Mô tả dataset gốc (Microsoft Cats and Dogs) và cách phân chia dữ liệu cho từng mô hình.
- [/] **Bước 3: Viết nội dung chi tiết Phần III (Model and Training)**
  - [/] **Artificial Neural Network (ANN)** (Hoàn thành)
    - [x] Lý thuyết nền tảng (Theoretical Background)
    - [x] Cấu trúc mô hình (Model Structure)
    - [x] Quá trình huấn luyện (Training Process)
    - [x] Kết quả và kết luận (Results and Conclusion)
  - [x] **HOG + Support Vector Machine (SVM)** (Hoàn thành)
    - [x] Lý thuyết nền tảng (Theoretical Background)
    - [x] Cấu trúc mô hình (Model Structure)
    - [x] Quá trình huấn luyện (Training Process)
    - [x] Kết quả và kết luận (Results and Conclusion)
  - [/] **SIFT + BoVW + Random Forest** (Đang thực hiện)
    - [x] Lý thuyết nền tảng (Theoretical Background)
    - [ ] Cấu trúc mô hình (Model Structure)
    - [ ] Quá trình huấn luyện (Training Process)
    - [ ] Kết quả và kết luận (Results and Conclusion)
  - [ ] **CNN from Scratch** (Chưa bắt đầu)
    - [ ] Lý thuyết nền tảng (Theoretical Background)
    - [ ] Cấu trúc mô hình (Model Structure)
    - [ ] Quá trình huấn luyện (Training Process)
    - [ ] Kết quả và kết luận (Results and Conclusion)
  - [ ] **CNN-ResNet (ResNet-18)** (Chưa bắt đầu)
    - [ ] Lý thuyết nền tảng (Theoretical Background)
    - [ ] Cấu trúc mô hình (Model Structure)
    - [ ] Quá trình huấn luyện (Training Process)
    - [ ] Kết quả và kết luận (Results and Conclusion)
- [ ] **Bước 4: Xây dựng Phần IV (So sánh, nhận xét)**
  - Lập bảng so sánh tổng hợp hiệu năng (thời gian training, tài nguyên) và chất lượng (Accuracy, Loss, Precision, Recall, F1-score) giữa các mô hình.
  - Đưa ra khuyến nghị lựa chọn mô hình tối ưu.
- [ ] **Bước 5: Viết Phần V (Hướng phát triển)**
  - Đề xuất hướng cải tiến, mở rộng mô hình (ResNet-34/50, Transformer) và cải thiện phần cứng/dữ liệu.
- [ ] **Bước 6: Kiểm tra và hoàn thiện**
  - Rà soát định dạng hiển thị, tính chính xác của công thức toán học và các liên kết hình ảnh trong [Report.md](file:///home/duy/Downloads/CapstoneProject-MachineLearning-20252/Report.md).
