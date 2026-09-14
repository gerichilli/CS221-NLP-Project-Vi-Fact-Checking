# 🇻🇳 Vietnamese Fact-Checking (ViFactCheck) — CS221 Course Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![Course](https://img.shields.io/badge/CS221-Natural%20Language%20Processing-green.svg)](https://uit.edu.vn/)

Dự án nghiên cứu và xây dựng hệ thống **Tự động kiểm chứng thông tin tiếng Việt (Automated Vietnamese Fact-Checking)** trên tập dữ liệu benchmark **ViFactCheck**, thuộc khuôn khổ môn học **CS221 — Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)**.

Dự án triển khai và thực nghiệm so sánh toàn diện qua 4 hướng tiếp cận: từ mô hình thống kê cổ điển (**TF-IDF + Logistic Regression**), mô hình ngôn ngữ tiền huấn luyện theo từ (**PhoBERT**), chuỗi thực nghiệm tối ưu chuyên sâu (**BamiBERT SOTA**), đến mô hình sinh ngôn ngữ lớn (**LLM QLoRA Fine-Tuning**).

---

## 📌 1. Bài toán & Dữ liệu (Problem Formulation & Dataset)

### 1.1. Phát biểu bài toán (Task Definition)
Kiểm chứng thông tin được mô hình hóa dưới dạng bài toán **Nhận diện suy luận ngữ nghĩa (Natural Language Inference - NLI)** theo cặp câu:
$$\text{Input} = (\text{Claim}, \text{Evidence}) \longrightarrow \text{Output} \in \{\text{SUPPORTED}, \text{REFUTED}, \text{NEI}\}$$

* **SUPPORTED (Ủng hộ):** Bằng chứng cung cấp đầy đủ dữ liệu xác thực tính đúng đắn của khẳng định.
* **REFUTED (Bác bỏ):** Bằng chứng mâu thuẫn trực tiếp hoặc phủ định tính đúng đắn của khẳng định.
* **NEI (Not Enough Info - Không đủ thông tin):** Bằng chứng không cung cấp đủ căn cứ logic để kết luận khẳng định đúng hay sai.

### 1.2. Thống kê tập dữ liệu ViFactCheck
| Phân chia (Split) | Số lượng mẫu (Samples) | Tỷ lệ (%) | Mục đích |
| :--- | :---: | :---: | :--- |
| **Train** | 10,980 | 78.4% | Huấn luyện mô hình |
| **Dev (Validation)** | 1,577 | 11.3% | Tinh chỉnh siêu tham số, Early Stopping |
| **Test** | 1,447 | 10.3% | Đánh giá độc lập chất lượng mô hình |
| **Tổng cộng** | **14,004** | **100%** | |

---

## 🏆 2. Bảng kết quả tổng hợp (Benchmark Leaderboard)

Dưới đây là kết quả đánh giá thực tế của toàn bộ các mô hình trên cùng một tập kiểm thử độc lập (**Test Set - 1,447 mẫu**):

| Mô hình (Model Architecture) | Hướng tiếp cận (Paradigm) | Test Accuracy | Test Macro-F1 | REFUTED F1 | SUPPORTED F1 | NEI F1 | Vị trí / Đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TF-IDF + Logistic Regression** | Statistical NLP Baseline | 38.63% | 38.11% | 31.77% | 45.20% | 37.37% | Baseline cổ điển |
| **BamiBERT Baseline** | Pre-trained RoBERTa | 81.76% | 81.57% | 75.00% | 82.80% | 86.91% | Baseline Transformer |
| **PhoBERT Base v2** | Word-level Transformer | 84.45% | 84.44% | 81.23% | 87.87% | 84.22% | Strong Transformer |
| 🥇 **BamiBERT (Exp006 - SOTA)** | **Prefix Prompting + Tuning** | **84.87%** | **84.84%** | **81.62%** | **85.14%** | **87.75%** | **Best Transformer** |
| 🚀 **LLM QLoRA (Fine-Tuned)** | Generative Instruction Tuning | **87.01%** | **87.02%** | — | — | — | **SOTA Toàn diện** |

### 💡 Nhận xét học thuật nổi bật:
1. **Sự sụp đổ của mô hình túi từ (Bag-of-Words):** TF-IDF chỉ đạt F1 ~38.11% (gần mức đoán ngẫu nhiên 3 lớp 33.3%). Điều này phản ánh rõ bản chất của Fact-Checking: mô hình bắt buộc phải hiểu ngữ cảnh và quan hệ logic (mâu thuẫn, khẳng định, thiếu hụt) thay vì chỉ đếm tần suất trùng lặp từ vựng.
2. **Đột phá từ Semantic Prefix Prompting (BamiBERT Exp006):** 
   - Thay vì ghép chuỗi thông thường `[CLS] claim [SEP] evidence [SEP]`, kỹ thuật định danh tiền tố `Khẳng định: {claim} Bằng chứng: {evidence}` giúp mô hình phân định rạch ròi vai trò của từng câu.
   - Kết hợp cùng **Weighted Loss (1.3x cho REFUTED)** và **Cosine Annealing Scheduler**, F1 của lớp khó nhất là **REFUTED tăng vọt từ 75.00% lên 81.62% (+6.62%)**, đưa BamiBERT vượt qua PhoBERT để trở thành mô hình Discriminative tốt nhất dự án (**84.84% Macro-F1**).
3. **Sức mạnh khái quát của LLM:** Fine-tuning LLM qua QLoRA đem lại năng lực suy luận ngữ nghĩa vượt trội, đạt đỉnh **87.02% Macro-F1**.

---

## 🔬 3. Chuỗi thực nghiệm tối ưu BamiBERT (Ablation Study)

Để tìm ra cấu hình BamiBERT SOTA, nhóm đã thiết kế chuỗi 7 thực nghiệm có kiểm soát (*ablation study*) được lưu vết tại [`notebooks/models/bamibert/outputs/bamibert/experiments_leaderboard.csv`](file:///notebooks/models/bamibert/outputs/bamibert/experiments_leaderboard.csv):

```
Exp000: Baseline (lr=2e-5, bs=8, linear)                      --> Macro-F1: 81.57% (Refuted: 75.00%)
  │
  ├── Exp001: Warmup 10% + Gradient Accumulation 2 (Eff BS=16) --> Macro-F1: 83.31%
  ├── Exp002: Lower LR (1e-5) chống overfit                     --> Macro-F1: 83.40%
  ├── Exp003: Higher LR (3e-5)                                 --> Macro-F1: 83.43%
  ├── Exp004: Class-Weighted Loss (1.3x REFUTED)                --> Macro-F1: 82.83%
  ├── Exp005: Cosine Annealing + Weight Decay 0.05 + Ep=6       --> Macro-F1: 83.48%
  │
  ├── 🏆 Exp006 (SOTA): Prefix Prompting + Best Hyperparams    --> Macro-F1: 84.84% (Refuted: 81.62% 🔥)
  │
  └── Exp007: Extended 10 Epochs + Early Stopping (Patience=3)  --> Macro-F1: 82.10% (Overfitting)
```

---

## 📂 4. Cấu trúc thư mục dự án (Repository Structure)

```plaintext
Project-NLP-Fact-Checking/
├── README.md                                 # Báo cáo tổng quan dự án
├── requirements.txt                          # Thư viện môi trường Python
├── .gitignore                                # Cấu hình loại bỏ file rác & weights nặng
│
├── data/                                     # Dữ liệu thực nghiệm
│   ├── raw/ViFactCheck_original/             # Dữ liệu gốc (train, dev, test)
│   └── processed/common_cleaned/             # Dữ liệu làm sạch chuẩn hóa chung
│
└── notebooks/
    ├── preprocessing/
    │   └── common_cleaning.ipynb             # Pipeline tiền xử lý văn bản tiếng Việt
    │
    └── models/
        ├── TFIDF_Logistic_Regression/        # 1. Classical Baseline
        │   ├── EDA_TFIDF_Logistic_Regression.ipynb
        │   └── outputs/                      # Báo cáo EDA, ma trận nhầm lẫn, biểu đồ
        │
        ├── Phobert/                          # 2. PhoBERT Base v2
        │   ├── preprocessing_phobert.ipynb   # Tiền xử lý từ & huấn luyện PhoBERT
        │   └── phobert_model/                # Lịch sử train & dự đoán test
        │
        ├── bamibert/                         # 3. BamiBERT Experiments (SOTA)
        │   ├── bamibert_baseline.ipynb       # Huấn luyện Baseline chuẩn mực
        │   ├── bamibert_experiments.ipynb    # 7 thực nghiệm tối ưu & Ablation Study
        │   └── outputs/bamibert/             # Bảng Leaderboard thực nghiệm (.csv)
        │
        └── LLM/                              # 4. Large Language Model (QLoRA)
            ├── LLM_FineTuning.ipynb          # Pipeline tinh chỉnh mô hình sinh
            ├── prompts/                      # Dữ liệu định dạng Prompt (JSONL)
            └── outputs/                      # Báo cáo đánh giá NLI & ma trận lỗi
```

---

## 🚀 5. Hướng dẫn cài đặt & Thực thi (Quickstart)

### 5.1. Khởi tạo môi trường
Yêu cầu: **Python >= 3.10** và khuyến nghị GPU (NVIDIA CUDA hoặc Apple Silicon MPS) khi huấn luyện lại.

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/Project-NLP-Fact-Checking.git
cd Project-NLP-Fact-Checking

# 2. Tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate   # Trên Windows: .venv\Scripts\activate

# 3. Cài đặt các gói phụ thuộc
pip install -U pip
pip install -r requirements.txt
```

### 5.2. Chạy và tái lập kết quả
Mỗi mô hình được đóng gói trọn vẹn trong Jupyter Notebook tương ứng:
* **Mô hình TF-IDF:** Mở và chạy [`notebooks/models/TFIDF_Logistic_Regression/EDA_TFIDF_Logistic_Regression.ipynb`](file:///notebooks/models/TFIDF_Logistic_Regression/EDA_TFIDF_Logistic_Regression.ipynb).
* **Mô hình PhoBERT:** Mở và chạy [`notebooks/models/Phobert/preprocessing_phobert.ipynb`](file:///notebooks/models/Phobert/preprocessing_phobert.ipynb).
* **Mô hình BamiBERT:**
  - Chạy Baseline: [`notebooks/models/bamibert/bamibert_baseline.ipynb`](file:///notebooks/models/bamibert/bamibert_baseline.ipynb).
  - Chạy chuỗi Ablation Study: [`notebooks/models/bamibert/bamibert_experiments.ipynb`](file:///notebooks/models/bamibert/bamibert_experiments.ipynb).
* **Mô hình LLM:** Mở và chạy [`notebooks/models/LLM/LLM_FineTuning.ipynb`](file:///notebooks/models/LLM/LLM_FineTuning.ipynb).

> [!NOTE]
> Tất cả các Notebooks trên GitHub đều **đã được lưu trữ sẵn toàn bộ output, bảng chỉ số và biểu đồ trực quan**, người xem không bắt buộc phải tải lại trọng số để xem kết quả đánh giá.

---

## 📦 6. Lưu ý về Trọng số Mô hình & Hugging Face

Theo chính sách của GitHub (giới hạn file tối đa 100MB), các tệp trọng số nhị phân nặng (`model.safetensors` từ 393MB – 515MB) cùng các thư mục checkpoint huấn luyện tạm thời không được đưa lên repository này (được cấu hình trong [`.gitignore`](file:///Project-NLP-Fact-Checking/.gitignore)).

Trọng số của mô hình xuất sắc nhất (**BamiBERT Exp006 SOTA**) có thể được tải về hoặc gọi trực tiếp từ Hugging Face Hub:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load checkpoint BamiBERT SOTA
model_name = "bamboodata/bamibert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Tải checkpoint đã fine-tune từ Hugging Face Hub (nếu có)
```

---

## 👥 7. Phân công thực hiện (Team Contributions)

| Thành viên | Trách nhiệm chính | Mô hình / Hạng mục đảm nhiệm |
| :--- | :--- | :--- |
| **Thành viên 1** | Khảo sát dữ liệu (EDA), Tiền xử lý cổ điển | TF-IDF + Logistic Regression Baseline, Báo cáo kiểm tra chất lượng dữ liệu |
| **Thành viên 2** | Nghiên cứu Transformer tiếng Việt | Pipeline tách từ VnCoreNLP & Huấn luyện PhoBERT Base v2 |
| **Thành viên 3** | Nghiên cứu SOTA Discriminative & Optimization | Xây dựng BamiBERT Baseline, Thiết kế 7 thực nghiệm Ablation Study, SOTA Prompting |
| **Thành viên 4** | Nghiên cứu Mô hình Ngôn ngữ Lớn (GenAI) | Thiết kế Prompt NLI (Plain/JSON), Fine-Tuning LLM qua QLoRA / PEFT |

---

## 📚 8. Tài liệu tham khảo (References)
1. **ViFactCheck Dataset:** Tập dữ liệu kiểm chứng thông tin tiếng Việt tiêu chuẩn cho bài toán NLI Fact-Checking.
2. **PhoBERT:** Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese*. EMNLP Findings.
3. **BamiBERT:** Bamboo Data Lab (2023). *BamiBERT: A Monolingual RoBERTa Model for Vietnamese Natural Language Processing*.
4. **QLoRA:** Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS.
