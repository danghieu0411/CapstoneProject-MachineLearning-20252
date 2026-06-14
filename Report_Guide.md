Nội dung trong ảnh:

# Dàn ý report

## I. Introduction

* Tổng quan về problem, liệt kê các model được dùng trong project.

## II. Dataset

* Mô tả về dataset, nguồn.
* Custom dataset cho mỗi model:

  * ANN, CNN, ResNet dùng chung dataset.
  * RF, SVM dùng random sample với dataset gốc.

## III. Model and Training

Với mỗi model cần có:

### 1. Lý thuyết

* Tổng hợp từ các file mọi người gửi.

### 2. Model Structure

* Các hyperparameter.
* Phần code construct model.
* Hình vẽ minh họa model.

### 3. Train Process

* Ảnh terminal train.
* Train graph.
* Nhận xét về train process và model được chọn:

  * Validation.
  * Loss.
  * Training time.

### 4. Test Result

* Kết quả test.
* Nhận xét về kết quả test.

### 5. Conclusion

* Nhận xét tổng quan về model:

  * Hiệu năng (training time, dataset size).
  * Chất lượng model.

### Nội dung báo cáo

#### 1. ML Models

* SVM
* Random Forest
* ANN

#### 2. DL Models

* CNN
* CNN ResNet

---

## IV. So sánh, nhận xét

Lập bảng so sánh các model về:

### Hiệu năng train

* Training time.
* Tài nguyên sử dụng.

### Chất lượng

* Accuracy.
* Loss.
* Các metric đánh giá trong bài Model Assessment (quan trọng nhất).

### Conclusion

* Recommendation model nên sử dụng.

---

## V. Hướng phát triển

* Project vẫn còn hạn chế về tài nguyên phần cứng và dataset.
* Có thể phát triển thêm các model đã được chứng minh hiệu quả như:

  * ResNet-34
  * ResNet-50
  * Transformer (trong ảnh ghi nhầm là "transformative").
