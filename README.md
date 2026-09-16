# Vietnamese Fact Checking

## 1. Giới thiệu bài toán

Fact Checking là bài toán xác định một Claim có được hỗ trợ, bị bác bỏ hay chưa đủ bằng chứng dựa trên thông tin liên quan.

Đồ án tập trung vào **Fact Verification tiếng Việt** trên bộ dữ liệu **ViFactCheck**, với ba nhãn:

- `SUPPORTED`
- `REFUTED`
- `NEI` (Not Enough Information)

Mục tiêu của nhóm là xây dựng và đánh giá một pipeline fact-checking, từ xử lý dữ liệu, xây dựng mô hình NLI đến tự động tìm kiếm và xếp hạng lại Evidence.

**Trạng thái:** Đã thực hiện; một số thực nghiệm End-to-End đang được hoàn thiện.

---

## 2. Các công trình liên quan

Nhóm khảo sát các hướng tiếp cận cho bài toán Fact Checking/NLI, bao gồm:

- Classical Machine Learning
- Transformer-based NLI
- Large Language Models
- Information Retrieval
- Information Extraction
- Evidence Ranking

Từ việc khảo sát, nhóm tập trung vào vấn đề: **Evidence có chất lượng ảnh hưởng trực tiếp đến khả năng Fact Verification**. Retrieval dựa trên độ tương đồng từ khóa có thể chọn các câu giống Claim nhưng chưa chắc phù hợp về mặt fact.

Từ đó, nhóm phát triển hướng **Fact-aware Evidence Reranking**, sử dụng thêm các đặc trưng Entity, Number và Date.

**Trạng thái:** Đang hoàn thiện phần Research Gap và Related Work.

---

## 3. Các hướng giải quyết

Nhóm triển khai và so sánh nhiều hướng tiếp cận:

### 3.1. Classical Baseline

- TF-IDF
- Logistic Regression

### 3.2. Transformer NLI

- PhoBERT
- BamiBERT

### 3.3. Large Language Model

- Fine-tuning LLM bằng LoRA/QLoRA

### 3.4. Evidence Retrieval

- BM25
- Top-K Evidence Retrieval

### 3.5. Fact-aware Evidence Processing

- Named Entity Extraction
- Numerical Fact Extraction
- Temporal Information Extraction
- Fact-aware Evidence Reranking

**Trạng thái:** Phần lớn đã thực hiện; IR → IE → Reranking đang được hoàn thiện.

---

## 4. Phương pháp nhóm thực hiện

Pipeline chính:

```text
Raw Dataset
     ↓
Common Preprocessing
     ↓
NLI Baselines
     ↓
Claim + Context
     ↓
Information Retrieval
     ↓
BM25 Top-K Evidence
     ↓
Information Extraction
     ↓
Fact-aware Evidence Reranking
     ↓
Claim + Reranked Evidence
     ↓
BamiBERT / PhoBERT
     ↓
Fact Verification
```

### 4.1. Common Preprocessing

Các bước làm sạch chung:

- Missing value checking
- Cleaning
- Whitespace normalization
- Unicode normalization
- Control/format character handling

### 4.2. Model-specific Preprocessing

Dữ liệu sau common preprocessing được chuẩn bị phù hợp với từng mô hình:

- TF-IDF + Logistic Regression
- PhoBERT
- BamiBERT
- LLM LoRA/QLoRA

### 4.3. Information Retrieval

Input:

```text
Claim + Context
```

Context được chia thành các sentence/evidence candidates.

```text
Claim
  ↓
BM25 scoring
  ↓
Ranking
  ↓
Top-K Evidence
```

### 4.4. Information Extraction

Từ Claim và Retrieved Evidence, nhóm trích xuất:

- Entity
- Number
- Date/Time

### 4.5. Fact-aware Evidence Reranking

Kết hợp:

```text
BM25 Score
+
Fact-level Features
```

để xếp hạng lại Evidence trước khi đưa vào NLI.

**Trạng thái:** IR → IE → Reranking đang thực hiện.

---

## 5. Cài đặt và thực nghiệm

### 5.1. Dataset

Sử dụng **ViFactCheck** với:

```text
Train
Dev
Test
```

Dataset được tổ chức dùng chung cho toàn bộ nhóm.

### 5.2. NLI Baseline

Các mô hình đã được huấn luyện/evaluate với Claim + Evidence:

- TF-IDF + Logistic Regression
- PhoBERT
- BamiBERT
- LLM LoRA/QLoRA

BamiBERT, PhoBERT và LLM đã cho kết quả baseline trên 80% theo các metric được nhóm sử dụng.

### 5.3. End-to-End Fact Verification

#### Experiment A — Gold Evidence → NLI

```text
Claim + Gold Evidence
        ↓
NLI
```

Mục đích: đánh giá verifier khi Evidence được cung cấp chính xác.

**Trạng thái:** Đã thực hiện.

#### Experiment B — BM25 → NLI

```text
Claim + Context
        ↓
BM25
        ↓
Retrieved Evidence
        ↓
NLI
```

Mục đích: đánh giá ảnh hưởng của automatic Evidence Retrieval.

**Trạng thái:** Đang thực hiện.

#### Experiment C — BM25 + IE Reranking → NLI

```text
Claim + Context
        ↓
BM25
        ↓
Top-K Evidence
        ↓
IE
        ↓
Fact-aware Reranking
        ↓
Reranked Evidence
        ↓
NLI
```

Mục đích: đánh giá liệu fact-level reranking có cải thiện hiệu quả Fact Verification hay không.

**Trạng thái:** Đang thực hiện.

### 5.4. Retrieval Evaluation

Các metric:

- Recall@1
- Recall@3
- Recall@5

**Trạng thái:** Đang thực hiện.

---

## 6. Phân tích kết quả

Nhóm sẽ so sánh:

| Experiment | Evidence | Verifier | Trạng thái |
|---|---|---|---|
| A | Gold Evidence | BamiBERT / PhoBERT | Đã thực hiện |
| B | BM25 Evidence | BamiBERT | Đang thực hiện |
| C | BM25 + IE Reranking | BamiBERT | Đang thực hiện |

Phân tích:

- Accuracy
- Macro-F1
- Precision / Recall / F1
- Retrieval Recall@K
- Ảnh hưởng của Evidence Retrieval đến NLI
- Ảnh hưởng của Fact-aware Reranking đến NLI

### Error Analysis

Dự kiến phân tích:

- Retrieval Error
- Entity Mismatch
- Number Mismatch
- Temporal Mismatch
- Negation
- NLI Reasoning Error

**Trạng thái:** Đang thực hiện.

---

## 7. Kết luận

Phần kết luận sẽ tổng hợp:

1. Hiệu quả của các NLI models trên ViFactCheck.
2. Ảnh hưởng của automatic Evidence Retrieval.
3. Hiệu quả của Fact-aware Evidence Reranking.
4. Những trường hợp phương pháp hoạt động tốt/chưa tốt.

**Trạng thái:** Chưa hoàn thiện; chờ kết quả End-to-End.

---

## 8. Hướng phát triển

Các hướng có thể xem xét:

- Dense/Semantic Retrieval
- Deep Learning-based Reranking
- Cải thiện Numerical/Temporal Information Extraction
- LLM-based Data Augmentation
- Thử nghiệm trên dataset/domain khác
- Cải thiện xử lý các trường hợp cần suy luận phức tạp

Các hướng trên là **future work**, không thuộc phạm vi bắt buộc của pipeline hiện tại.

**Trạng thái:** Đã xác định; sẽ chọn lọc trong báo cáo cuối.

---

## Project Status

| Thành phần | Trạng thái |
|---|---|
| Dataset | ✅ Đã thực hiện |
| EDA | ✅ Đã thực hiện |
| Common Preprocessing | ✅ Đã thực hiện |
| TF-IDF + Logistic Regression | ✅ Đã thực hiện |
| PhoBERT | ✅ Đã thực hiện |
| BamiBERT | ✅ Đã thực hiện |
| LLM LoRA/QLoRA | ✅ Đã thực hiện |
| BM25 Retrieval | 🔄 Đang thực hiện |
| Retrieval Evaluation | 🔄 Đang thực hiện |
| Information Extraction | 🔄 Đang thực hiện |
| Fact-aware Reranking | 🔄 Đang thực hiện |
| End-to-End BamiBERT | 🔄 Đang thực hiện |
| Error Analysis | 🔄 Đang thực hiện |
| Conclusion | ⏳ Chờ kết quả cuối |
| Demo | ⏳ Đang xem xét |
