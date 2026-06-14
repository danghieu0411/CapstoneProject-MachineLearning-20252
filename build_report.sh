#!/bin/bash
# Script tự động đồng bộ và biên dịch báo cáo LaTeX sang PDF bằng Docker

# Dừng lại nếu có lỗi xảy ra
set -e

echo "===== 1. Đồng bộ hóa các tài liệu (Report.tex -> Report.md & README.md) ====="
python3 scratch/sync_files.py

echo "===== 2. Biên dịch LaTeX sang PDF bằng Docker (Lần 1) ====="
docker run --rm --user $(id -u):$(id -g) -v $(pwd)/report:/work -w /work texlive/texlive pdflatex -interaction=nonstopmode Report.tex

echo "===== 3. Biên dịch LaTeX sang PDF bằng Docker (Lần 2 để cập nhật mục lục & link) ====="
docker run --rm --user $(id -u):$(id -g) -v $(pwd)/report:/work -w /work texlive/texlive pdflatex -interaction=nonstopmode Report.tex

echo "===== 4. Hoàn tất! File báo cáo đã được sinh ra tại: report/Report.pdf ====="
ls -lh report/Report.pdf
