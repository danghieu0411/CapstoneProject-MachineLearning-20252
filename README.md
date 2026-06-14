# Cat and Dog Image Classification (Machine Learning Capstone)

Comparison of **five models** on binary **cats vs. dogs** classification: **traditional ML** (handcrafted features) and **deep learning** (learned features).

Full written report: [`report/Report.md`](report/Report.md) / [`report/Report.tex`](report/Report.tex) — pre-built PDF: [`report/Report.pdf`](report/Report.pdf).

**Course:** Machine Learning — **Instructor:** Assoc. Prof. Thân Quang Khoát — **HUST / SOICT**

---

## Table of contents

- [Results summary](#results-summary)
- [Open in Colab](#open-in-colab)
- [System requirements](#system-requirements)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Installation and running notebooks](#installation-and-running-notebooks)
- [Demo outputs](#demo-outputs)
- [Troubleshooting](#troubleshooting)
- [Building the report PDF](#building-the-report-pdf)
- [Submission package](#submission-package)
- [References](#references)
- [Team](#team)

---

## Results summary (test set)

| Model | Training data | Training time | GPU | Test accuracy | F1-score |
|-------|---------------|---------------|-----|---------------|----------|
| ANN (MLP baseline) | 25k images, 64×64 RGB | ~10 min | Optional | 64.48% | 0.64 |
| HOG + linear SVM | 4k images, 64×64 gray | < 1 min | No | 69.75% | 0.70 |
| SIFT + BoVW + random forest | 4k images, 256×256 gray | ~3 min | No | 69.00% | 0.69 |
| Custom CNN (PyTorch) | 25k images, 224×224 RGB | ~1.5 h | Yes | 96.20% | 0.96 |
| **ResNet-18 (PyTorch)** | 25k images, 224×224 RGB | ~1.5 h | Yes | **96.33%** | **0.96** |

**Recommendation:** use **ResNet-18** when a GPU is available; use **HOG + SVM** on CPU-only or resource-constrained setups.

---

## Open in Colab

[![Open ANN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danghieu0411/CapstoneProject-MachineLearning-20252/blob/report/ANN/Final_ANN.ipynb)
[![Open CNN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danghieu0411/CapstoneProject-MachineLearning-20252/blob/report/CNN-FromScratch/CNN_FromScratch.ipynb)
[![Open ResNet](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danghieu0411/CapstoneProject-MachineLearning-20252/blob/report/CNN-ResNet/CNN_resnet_final.ipynb)
[![Open HOG-SVM](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danghieu0411/CapstoneProject-MachineLearning-20252/blob/report/HOG-SVM/HOG%20and%20SVM.ipynb)
[![Open SIFT-RF](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danghieu0411/CapstoneProject-MachineLearning-20252/blob/report/SIFT-HOVW-RF/Classifier.ipynb)

After opening: **Runtime → Change runtime type → GPU** (for ANN / CNN / ResNet) → **Runtime → Run all**.

---

## System requirements

- **Python** 3.9–3.11 (3.10 recommended).
- **pip** ≥ 22.
- **RAM:** 8 GB+ (ML notebooks), 16 GB+ (DL).
- **GPU:** NVIDIA recommended for CNN / ResNet (CUDA 11.8 / 12.1). CPU works but training is slow.
- **Disk:** ~5 GB for extracted dataset and optional checkpoints.
- **Browser:** Chrome, Edge, or Firefox (Colab / Jupyter).

Quick checks:

```bash
python --version
nvidia-smi   # if NVIDIA GPU is available
```

> **Note:** This repo does **not** ship trained weights (`.h5`, `.pth`, `.pkl`). Run the notebooks end-to-end to train, or add your own checkpoint cells for inference.

---

## Repository layout

```
CapstoneProject-MachineLearning-20252/
├── ANN/                    # TensorFlow/Keras MLP baseline
│   ├── Final_ANN.ipynb     #   train + evaluation demo
│   └── ANN.md
├── HOG-SVM/                # OpenCV HOG + linear SVM (SGD)
│   ├── HOG and SVM.ipynb
│   └── HOG_SVM.md
├── SIFT-HOVW-RF/           # SIFT, bag-of-visual-words, random forest
│   ├── Classifier.ipynb
│   └── SIFT-HOVW-RF.md
├── CNN-FromScratch/        # Custom PyTorch CNN
│   ├── CNN_FromScratch.ipynb
│   └── CNN-FromScratch.md
├── CNN-ResNet/             # PyTorch ResNet-18
│   ├── CNN_resnet_final.ipynb
│   └── CNN-Resnet.md
├── report/                 # LaTeX report + figures + Report.pdf
├── scratch/                # Report tooling (not ML training)
├── Report_Guide.md
├── Walkthrough.md
└── build_report.sh         # PDF build (Linux + Docker)
```

Each model folder contains:

- **Notebook** — full training pipeline and **evaluation demo** (metrics, plots).
- **`*.md`** — theory, hyperparameters, and results (source for the written report).

`scratch/*.py` is for **LaTeX sync / PDF checks only**, not for running models.

---

## Dataset

- **Source:** [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) / **Microsoft Cats and Dogs** (~25,000 JPEGs, 12,500 per class).
- **Not included in the repo** — download and extract when running notebooks (Colab or local).
- Corrupt / undecodable images are skipped at load time.

**Splits (as in the report):**

| Group | Models | Preprocessing | Train / val / test |
|-------|--------|---------------|---------------------|
| ML | HOG-SVM, SIFT-RF | Balanced **4,000** sample | **3,200 / — / 800** (80/20 stratified) |
| DL | ANN, CNN, ResNet | ~full **25,000** | **20,000 / 2,500 / 2,500** |

---

## Installation and running notebooks

### 1. Clone

```bash
git clone https://github.com/danghieu0411/CapstoneProject-MachineLearning-20252.git
cd CapstoneProject-MachineLearning-20252
git checkout report
```

### 2. Environment

**Google Colab (recommended for DL):** use the badges above → enable GPU → **Run all**. Install cells are included where needed.

**Local Jupyter (Windows / Linux / macOS):**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -U pip jupyter tensorflow torch torchvision opencv-python scikit-image scikit-learn pandas matplotlib seaborn pillow kagglehub tqdm
jupyter notebook
```

### 3. Data layouts (per notebook)

| Path after download | Notebooks | How to obtain data |
|---------------------|-----------|-------------------|
| `data/train/*.jpg` (`cat.N.jpg`, `dog.N.jpg`) | ANN, custom CNN | First cells: `kagglehub` + `competition_download('dogs-vs-cats')`. Accept [competition rules](https://www.kaggle.com/c/dogs-vs-cats). Colab: Secrets `KAGGLE_USERNAME`, `KAGGLE_KEY`. |
| `dataset/PetImages/Cat`, `dataset/PetImages/Dog` | HOG-SVM, SIFT-RF | Notebook `wget` + `unzip` of [Microsoft archive](https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip) (~787 MB), or download manually into `dataset/`. |
| `{PROJECT_PATH}/data/train`, `val`, `test` (`cats/` / `dogs/`) | ResNet-18 | Mount Google Drive and set `PROJECT_PATH`, or point paths to your local train/val/test split. |

**Without Colab:** skip `google.colab.userdata` / `drive.mount`; place data in the correct folders and edit `train_dir`, `base_dir`, or `PROJECT_PATH`.

### 4. Notebooks — run steps and in-notebook demo

| Notebook | Stack | How to run | Demo (end of notebook) |
|----------|-------|------------|-------------------------|
| [`ANN/Final_ANN.ipynb`](ANN/Final_ANN.ipynb) | TensorFlow/Keras | Run all — MLP **30 epochs**, 64×64 RGB, `seed=36` | Test loss/accuracy, curves, classification report, confusion matrix |
| [`CNN-FromScratch/CNN_FromScratch.ipynb`](CNN-FromScratch/CNN_FromScratch.ipynb) | PyTorch | Run all — **25 epochs** (`lr=1e-3`, `batch=64`), 224×224 + ImageNet aug | Test metrics, confusion matrix |
| [`CNN-ResNet/CNN_resnet_final.ipynb`](CNN-ResNet/CNN_resnet_final.ipynb) | PyTorch | Run all — ResNet-18 **~20 epochs** | Test on 2,500 images, curves, confusion matrix |
| [`HOG-SVM/HOG and SVM.ipynb`](HOG-SVM/HOG%20and%20SVM.ipynb) | OpenCV + sklearn | STEP 1–4: 1764-D HOG + linear SVM (500 SGD steps) | Report + confusion matrix (800 test) |
| [`SIFT-HOVW-RF/Classifier.ipynb`](SIFT-HOVW-RF/Classifier.ipynb) | OpenCV + sklearn | SIFT, BoVW **K=500**, random forest | Classification report + confusion matrix |

**Rough runtime:** HOG-SVM minutes (CPU); SIFT-RF longer (SIFT + K-Means); ANN ~10 min; CNN/ResNet ~1.5 h on Colab GPU.

### 5. Step-by-step (optional)

Many notebooks use **STEP 1–5**: data → features/preprocess → model → train → **test + plots** (demo). Use **Shift+Enter** per cell if you do not use Run all.

### 6. `scratch/` scripts (report only)

```bash
python scratch/sync_files.py
python scratch/check_latex.py
python scratch/draw_pipeline.py
```

On Windows, fix hardcoded paths in `sync_files.py` if needed.

There is **no separate demo app** (e.g. Gradio or `predict.py`); evaluation demos live **inside each notebook**.

---

## Demo outputs

Each notebook produces:

- Learning curves (loss and accuracy vs. epoch)
- Confusion matrix (2×2 heatmap)
- Classification report (precision, recall, F1 per class)
- Sample predictions (where implemented in the notebook)

Exported figures used in the report are under `report/` (e.g. `ann_confusion_matrix.png`, `cnn_training_curves.png`). See [`report/Report.md`](report/Report.md) for the full write-up.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `KAGGLE_USERNAME` not found (Colab) | Missing Secrets | Colab key icon → add `KAGGLE_USERNAME`, `KAGGLE_KEY` |
| `401` from `kagglehub` | Rules not accepted | Join / accept rules on [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) |
| CUDA OOM (CNN / ResNet) | Batch too large | Lower `batch_size` (e.g. 64 → 32 → 16) |
| `OMP: Error #15` | OpenMP conflict | `os.environ['KMP_DUPLICATE_LIB_OK']='True'` in first code cell |
| SIFT missing in `cv2` | OpenCV packaging | `pip install opencv-contrib-python` |
| `kagglecatsanddogs_5340.zip` not found | Network blocked `wget` | Download zip in browser, unzip to `dataset/` |
| PDF build font errors | Missing TeX Vietnamese bundle | Use `build_report.sh` (Docker `texlive/texlive`) or Overleaf |

---

## Building the report PDF

```bash
./build_report.sh   # Linux + Docker → report/Report.pdf
```

Runs `scratch/sync_files.py` and `pdflatex` twice (table of contents). On Windows: compile `report/Report.tex` with TeX Live or Overleaf.

---

## Submission package

Course capstone delivery typically includes (check your LMS assignment for exact names and deadlines):

| Item | Location in this repo |
|------|------------------------|
| **Report (PDF)** | `report/Report.pdf` |
| **Slides (PDF)** | **Not in repo** — create separately and export to PDF |
| **Source (ZIP)** | Zip project root: all model folders, `report/` sources, `README.md`, notebooks. **Exclude** `.git/`, virtualenvs, and raw datasets (~GB) unless the assignment requires them |

One team member usually uploads all files to the assignment portal.

---

## References

| File | Purpose |
|------|---------|
| [`AGENTS.md`](AGENTS.md) | Contributor / report editing rules |
| [`Report_Guide.md`](Report_Guide.md) | Report outline |
| [`Walkthrough.md`](Walkthrough.md) | Progress checklist |
| [`report/Report.md`](report/Report.md) | Full report |
| Per-model `*.md` | Model-specific documentation |

Papers cited in the report include ResNet (He et al.), HOG (Dalal & Triggs), and SIFT (Lowe).

---

## Team

| # | Name | Student ID |
|---|------|------------|
| 1 | Lê Đăng Hiếu | 202416796 |
| 2 | Lê Thế Nhật Anh | 202416770 |
| 3 | Phạm Trọng Bình | 202416781 |
| 4 | Phạm Đức Hoàng | 202416696 |
| 5 | Nguyễn Trung Thông | 202416829 |
| 6 | Nguyễn Văn Đức Duy | 202416791 |
