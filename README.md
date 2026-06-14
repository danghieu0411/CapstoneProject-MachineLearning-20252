# Phân loại ảnh Chó / Mèo (Capstone Machine Learning)

Dự án so sánh **5 mô hình** trên bài toán nhị phân **cats vs dogs**: hai pipeline **học máy truyền thống** (đặc trưng thủ công) và **học sâu** (tự học đặc trưng). Báo cáo LaTeX đầy đủ: [`report/Report.md`](report/Report.md) (và `report/Report.pdf` sau khi build).

**Môn:** Machine Learning — **GV:** PGS.TS. Thân Quang Khoát — **HUST / SOICT**

---

## Kết quả tổng hợp (test set)

| Mô hình | Dữ liệu | Thời gian train (ước lượng) | GPU | Test Accuracy |
|---------|---------|------------------------------|-----|---------------|
| ANN (MLP) | 25k, 64×64 RGB | ~10 phút | Tùy chọn | **64.48%** |
| HOG + Linear SVM | 4k, 64×64 gray | < 1 phút | Không | **69.75%** |
| SIFT + BoVW + Random Forest | 4k, 256×256 gray | ~3 phút | Không | **69.00%** |
| CNN (PyTorch, tự xây) | 25k, 224×224 RGB | ~1.5 giờ | Có | **96.20%** |
| ResNet-18 (PyTorch) | 25k, 224×224 RGB | ~1.5 giờ | Có | **96.33%** |

Khuyến nghị: **ResNet-18** khi có GPU; **HOG + SVM** khi chỉ có CPU / tài nguyên hạn chế.

---

## Cấu trúc repository

```
CapstoneProject-MachineLearning-20252/
├── ANN/                    # TensorFlow/Keras — MLP baseline
├── HOG-SVM/                # OpenCV HOG + SVM (SGD)
├── SIFT-HOVW-RF/           # SIFT, BoVW (K-Means), Random Forest
├── CNN-FromScratch/        # PyTorch CNN custom
├── CNN-ResNet/             # PyTorch ResNet-18
├── report/                 # Báo cáo + hình minh họa
├── scratch/                # sync LaTeX, kiểm tra build
├── Report_Guide.md         # Dàn ý báo cáo
├── Walkthrough.md          # Checklist tiến độ báo cáo
└── build_report.sh         # Build PDF (Linux + Docker)
```

Mỗi thư mục mô hình gồm:

- **Notebook** — code chạy thực nghiệm (chính).
- **`*.md`** — lý thuyết, hyperparameter, kết quả (nguồn cho Phần III báo cáo).

---

## Dataset

- **Nguồn:** [Kaggle — Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) / bộ **Microsoft Cats and Dogs** (~25.000 ảnh JPEG, 12.500 chó + 12.500 mèo).
- **Không có sẵn trong repo** — tải và giải nén trên máy / Colab.
- Ảnh lỗi (không decode được) được bỏ qua khi load.

**Chia dữ liệu (theo báo cáo):**

| Nhóm | Mô hình | Tiền xử lý | Train / Val / Test |
|------|---------|------------|---------------------|
| ML | HOG-SVM, SIFT-RF | Lấy mẫu **4.000** cân bằng | **3.200 / — / 800** (80/20 stratified) |
| DL | ANN, CNN, ResNet | Gần full **25k** | **20.000 / 2.500 / 2.500** |

---

## Chạy từng notebook

### 1. ANN — `ANN/Final_ANN.ipynb`

- **Framework:** TensorFlow / Keras
- **Tải data:** `kagglehub` + competition `dogs-vs-cats` (cần chấp nhận rules Kaggle). Trên Colab: lưu `KAGGLE_USERNAME`, `KAGGLE_KEY` trong **Secrets** (`google.colab.userdata`).
- **Luồng:** `data/train/` (`cat.*.jpg`, `dog.*.jpg`) → resize **64×64 RGB** → flatten 12.288 → MLP (augmentation, L2, Dropout) → **30 epoch**.
- **Seed:** `numpy`/`tf` = 36.

### 2. CNN from scratch — `CNN-FromScratch/CNN_FromScratch.ipynb`

- **Framework:** PyTorch + `torchvision`
- **Tải data:** giống ANN (`kagglehub`, `train.zip` → `data/train`).
- **Hyperparameter (trong notebook):** `lr=1e-3`, `batch_size=64`, `epochs=25`.
- **Input:** **224×224**, augmentation + chuẩn hóa ImageNet.

### 3. ResNet-18 — `CNN-ResNet/CNN_resnet_final.ipynb`

- **Framework:** PyTorch
- **Data:** notebook mount **Google Drive**, đường dẫn mẫu: `.../cat_dog_project/data/train|val|test` (đã chia sẵn trên Drive).
- Cần **tự chia train/val/test** (20k/2.5k/2.5k) hoặc dùng bộ đã upload như trong notebook.
- **~20 epoch**, ResNet-18 + GAP + FC 2 lớp.

### 4. HOG + SVM — `HOG-SVM/HOG and SVM.ipynb`

- **Thư viện:** OpenCV (HOG), `sklearn`, SVM linear + **SGD** (custom trong notebook).
- **4.000** ảnh, gray **64×64**, vector HOG **1.764-D**, **500** bước SGD (theo báo cáo).

### 5. SIFT + BoVW + RF — `SIFT-HOVW-RF/Classifier.ipynb`

- **Thư viện:** OpenCV (SIFT), `sklearn` (MiniBatchKMeans, RandomForest).
- Unzip archive Microsoft/Kaggle vào `dataset/` (xem `SIFT-HOVW-RF.md`).
- Gray **256×256**, từ điển **k=500**, Random Forest trên histogram 500-D.

**Gợi ý:** Colab + GPU cho ANN/CNN/ResNet; CPU đủ cho HOG-SVM và SIFT-RF.

---

## Phụ thuộc chính

| Mô hình | Gói |
|---------|-----|
| ANN | `tensorflow`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `kagglehub` |
| CNN / ResNet | `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `Pillow`, `tqdm` |
| HOG-SVM | `opencv-python`, `numpy`, `scikit-learn`, `matplotlib` |
| SIFT-RF | `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib` |

```bash
pip install tensorflow torch torchvision opencv-python scikit-learn pandas matplotlib pillow kagglehub tqdm
```

---

## Build báo cáo PDF

```bash
./build_report.sh   # Linux + Docker → report/Report.pdf
```

Trên Windows: TeX Live / Overleaf với `report/Report.tex`.

---

## Tài liệu trong repo

| File | Mục đích |
|------|----------|
| [`AGENTS.md`](AGENTS.md) | Quy tắc chỉnh báo cáo |
| [`Report_Guide.md`](Report_Guide.md) | Dàn ý báo cáo |
| [`Walkthrough.md`](Walkthrough.md) | Tiến độ |
| `*/**.md` trong từng folder | Chi tiết mô hình |

---

## Nhóm thực hiện

| STT | Họ tên | MSSV |
|-----|--------|------|
| 1 | Lê Đăng Hiếu | 202416796 |
| 2 | Lê Thế Nhật Anh | 202416770 |
| 3 | Phạm Trọng Bình | 202416781 |
| 4 | Phạm Đức Hoàng | 202416696 |
| 5 | Nguyễn Trung Thông | 202416829 |
| 6 | Nguyễn Văn Đức Duy | 202416791 |
