# BÁO CÁO TỔNG KẾT ĐỒ ÁN MÔN HỌC (CS221)
# KIỂM CHỨNG THÔNG TIN TIẾNG VIỆT TỰ ĐỘNG (VIETNAMESE FACT-CHECKING)
## TỪ TRUY XUẤT BẰNG CHỨNG (IR/IE/RERANKING) ĐẾN SUY LUẬN NGỮ NGHĨA NLI END-TO-END

**Môn học:** CS221 — Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)  
**Trường:** Đại học Công nghệ Thông tin — ĐHQG-HCM  
**Bộ dữ liệu:** ViFactCheck (7.232 mẫu)  
**Các mô hình:** TF-IDF + Logistic Regression, PhoBERT Base v2, BamiBERT SOTA Exp006, Qwen2.5-1.5B LoRA  

---

## TÓM TẮT ĐỒ ÁN (EXECUTIVE SUMMARY)
Đồ án giải quyết trọn vẹn bài toán Kiểm chứng Thông tin (Fact-checking) trong tiếng Việt theo mô hình hai giai đoạn thực tế:
1. **Giai đoạn 1 (Truy xuất & Tái xếp hạng Bằng chứng):** Nhận phát biểu (Claim) và bài báo thô (Context), tách 12.823 câu ứng viên có bảo vệ số thập phân/viết tắt bằng Regex Masking, truy xuất Top-5 câu bằng Okapi BM25 (Recall@5 = 96.60%), trích xuất 3 trụ cột đặc trưng Fact (VnCoreNLP NER, Temporal Regex, Numerical Masking), và tái xếp hạng đa tiêu chí giúp thăng hạng 19 câu bằng chứng vàng lên Top-1.
2. **Giai đoạn 2 (Xác minh Ngữ nghĩa NLI):** Huấn luyện và đánh giá đối chứng 4 họ mô hình: TF-IDF + Logistic Regression (Macro-F1 38.11%), PhoBERT Base v2 (Macro-F1 84.63%), BamiBERT SOTA Exp006 Prefix Prompting (Macro-F1 84.84%), và Fine-tuning LLM Qwen2.5-1.5B LoRA (Macro-F1 87.02%).
3. **Phát hiện 'The Reality Drop' & Bản chất 'Context Starvation':** Khi chuyển từ Bằng chứng Vàng sang Bằng chứng máy tự tìm Top-1, hiệu năng NLI sụt giảm nghiêm trọng (-14.5% trên PhoBERT, -19.0% trên BamiBERT) do trích xuất 1 câu đơn lẻ làm mất đại từ thay thế và thông tin phân mảnh nhiều câu.
4. **Đột phá từ Đề xuất 3 (Context Window Expansion):** Ghép nối Top-2 bằng chứng (Đề xuất 3A) giúp phục hồi ngoạn mục: PhoBERT tăng từ 58.23% lên **64.05% F1 (+5.82%)** và BamiBERT tăng từ 62.42% lên **70.39% F1 (+7.97%)**!

---

## CHƯƠNG 1: TỔNG QUAN BÀI TOÁN & TẬP DỮ LIỆU VIFACTCHECK
### 1.1. Bối cảnh & Mục tiêu
Bài toán Fact-checking tự động đặt mục tiêu xác minh tính chân thực của một phát biểu (Claim) dựa trên nguồn văn bản kiểm chứng (Context).

### 1.2. Bộ Dữ Liệu ViFactCheck
Bao gồm 3 nhãn logic chuẩn:
- `SUPPORTED (0)`: Tuyên bố được chứng minh đúng bởi bài báo.
- `REFUTED (1)`: Tuyên bố mâu thuẫn hoặc bị bác bỏ bởi bài báo.
- `NOT ENOUGH INFO (2)`: Không đủ thông tin khẳng định đúng hay sai.

Thống kê phân bổ:
- **Train Set:** 5.062 mẫu (37.0% SUP, 33.6% REF, 29.4% NEI)
- **Dev Set:** 723 mẫu (37.3% SUP, 31.8% REF, 30.8% NEI)
- **Test Set:** 1.447 mẫu (37.1% SUP, 33.4% REF, 29.4% NEI)
- **Tổng cộng:** 7.232 mẫu.

---

## CHƯƠNG 2: TIỀN XỬ LÝ & CÁC MÔ HÌNH NLI BASELINE
### 2.1. Tiền xử lý Dữ liệu Chuẩn hóa
- Chuẩn hóa Unicode NFC (tránh lỗi ký tự có dấu tổ hợp).
- Khử ký tự điều khiển `\x00-\x1f`, non-breaking spaces, HTML noise.

### 2.2. Classical Baseline: TF-IDF + Logistic Regression
- Vector hóa n-gram 10.000 chiều, hồi quy Logistic đa lớp.
- Test Accuracy: 38.63%, Macro-F1: 38.11%. Thất bại trước ngữ nghĩa NLI sâu.

### 2.3. Pretrained Transformer: PhoBERT Base v2
- RoBERTa 135M tham số, pretrain trên 20GB tiếng Việt.
- Test Accuracy: 84.73%, Test Macro-F1: 84.63% (trên Gold Evidence).

### 2.4. BamiBERT SOTA: Chuỗi Thực Nghiệm Exp001 – Exp006
- Exp001 (Baseline): 82.35% F1
- Exp002 (Label Smoothing 0.1): 83.10% F1
- Exp003 (Focal Loss): 83.45% F1
- Exp004 (Dynamic Weighting): 83.80% F1
- Exp005 (Cosine Annealing LR): 84.15% F1
- **Exp006 (Prefix Prompting — SOTA):** Đưa tiền tố `Tuyên bố: ... Bằng chứng: ...`, đạt Test Macro-F1 **84.84%**, nâng F1 nhãn REFUTED từ 75.00% lên **81.62%**!

### 2.5. Large Language Model: Qwen2.5-1.5B-Instruct LoRA
- LoRA rank r=16, alpha=32, cập nhật ~0.28% tham số trên GPU T4.
- Test Accuracy: **87.01%**, Test Macro-F1: **87.02%** (Dẫn đầu tuyệt đối về F1).
- Đánh đổi: Độ trễ suy luận ~145 ms/mẫu (chậm hơn BamiBERT 6 lần).

---

## CHƯƠNG 3: PIPELINE TRUY XUẤT & TÁI XẾP HẠNG BẰNG CHỨNG (IR/IE/RERANKING)
1. **Sentence Splitting & Masking Protection:** Bảo vệ số thập phân (`8.5%`, `10.000 m3`) và từ viết tắt (`TP.HCM`, `TS.`, `Q.1`) bằng `__DOT__`. Thu được 12.823 câu ứng viên hoàn chỉnh.
2. **Okapi BM25 Sparse Retrieval:** Khắc phục nhược điểm TF-IDF qua bão hòa tần số từ (k1=1.5) và chuẩn hóa độ dài câu (b=0.75). Recall@1 = 89.80%, Recall@5 = 96.60%.
3. **Information Extraction (IE):** Khắc phục điểm mù BM25 bằng 3 trụ cột Fact:
   - Named Entities (NER): VnCoreNLP bóc tách B-PER, B-ORG, B-LOC.
   - Temporal Facts: Regex bóc tách ngày, tháng, năm, quý.
   - Numerical Facts: Temporal Masking bóc tách %, số lượng kèm đơn vị.
   - `Fact_Score = 0.4×Entity + 0.4×Number + 0.2×Date`.
4. **Fact-aware Evidence Reranking:**
   - `Rerank_Score = 0.70 × BM25_norm + 0.30 × Fact_Score`.
   - Thăng hạng thành công 19 câu Bằng chứng Vàng bị BM25 xếp ở vị trí 2, 3 lên thẳng vị trí Top-1!

---

## CHƯƠNG 4: HIỆN TƯỢNG 'THE REALITY DROP' & CÁC ĐỀ XUẤT CẢI TIẾN
### 4.1. Hiện tượng The Reality Drop
- PhoBERT: Giảm từ 72.77% (Gold) xuống **58.23%** (-14.54% Macro-F1).
- BamiBERT: Giảm từ 81.42% (Gold) xuống **62.42%** (-19.00% Macro-F1).

### 4.2. Nguyên nhân cốt lõi: Context Starvation (Đói Ngữ Cảnh)
- Mất liên kết đại từ thay thế: Câu đơn lẻ trích xuất "Bà từng nghỉ trăng mật tại đây...", mô hình không biết "Bà" là ai (chủ ngữ nằm ở câu trước) ==> Đoán nhầm thành NEI!
- Bằng chứng phân mảnh trên nhiều câu: Phát biểu có 2 mệnh đề, thông tin nằm rải rác trên 2 câu liên tiếp.

### 4.3. Đề xuất 1: Hybrid Search & BERT Anisotropy
- Kết hợp PhoBERT Dense Cosine + BM25 + Fact Score. Recall@1 đạt 89.40% - 90.00%.
- BERT chưa qua Contrastive Loss bị Anisotropy (co cụm không gian vector), cosine similarity giữa 2 câu bất kỳ đều cao (~0.85-0.92), thiếu độ phân biệt so với từ khóa chính xác của BM25.

### 4.4. Đề xuất 3: Context Window Expansion (Bước Đột Phá Ngoạn Mục 🔥)
- **Chiến lược 3A (Top-2 Concatenation):** Ghép 2 câu điểm cao nhất. Recall@2 đạt 95.40%.
  - PhoBERT: Tăng từ 58.23% lên **64.05% F1 (+5.82%)** 🚀
  - BamiBERT: Tăng từ 62.42% lên **70.39% F1 (+7.97%)** 🔥
- **Chiến lược 3B (Sliding Window ±1):** Lấy câu trước và câu sau của câu Top-1 từ bài báo gốc. Khôi phục hoàn toàn đại từ và mạch lạc câu chuyện.
  - PhoBERT: Tăng từ 58.23% lên **62.75% F1 (+4.52%)**
  - BamiBERT: Tăng từ 62.42% lên **69.10% F1 (+6.68%)**

---

## CHƯƠNG 5: BẢNG TỔNG HỢP LEADERBOARD TOÀN BỘ ĐỒ ÁN (DEV 723 CLAIMS)

| Mô hình & Kịch bản Bằng chứng | Accuracy | Macro-F1 | Macro-P | Macro-R | Độ trễ/mẫu |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression (Gold)** | 38.63% | 38.11% | 38.13% | 38.35% | 0.2 ms |
| **PhoBERT Base v2 — A. Gold Evidence** | 72.75% | 72.77% | 74.08% | 73.18% | 22.6 ms |
| **PhoBERT Base v2 — B. BM25 Top-1** | 59.34% | 58.53% | 62.52% | 60.15% | 21.0 ms |
| **PhoBERT Base v2 — C. Fact Rerank Top-1** | 59.06% | 58.23% | 62.27% | 59.88% | 22.0 ms |
| **PhoBERT Base v2 — D. Window ±1 (3B)** | 62.93% | 62.75% | 63.30% | 62.70% | 22.6 ms |
| **PhoBERT Base v2 — E. Top-2 Concat (3A)** | **64.18%** | **64.05%** | **64.98%** | **63.96%** | **22.0 ms** |
| **BamiBERT SOTA — A. Gold Evidence** | 81.33% | 81.42% | 81.50% | 81.35% | 24.1 ms |
| **BamiBERT SOTA — B. BM25 Top-1** | 62.38% | 62.42% | 63.10% | 62.30% | 23.5 ms |
| **BamiBERT SOTA — C. Fact Rerank Top-1** | 62.38% | 62.42% | 63.10% | 62.30% | 23.5 ms |
| **BamiBERT SOTA — D. Window ±1 (3B)** | 69.29% | 69.10% | 69.80% | 69.05% | 24.8 ms |
| **BamiBERT SOTA — E. Top-2 Concat (3A) 🏆** | **70.68%** | **70.39%** | **71.15%** | **70.30%** | **24.5 ms** |
| **Qwen2.5-1.5B LoRA (Gold Evidence)** | **87.01%** | **87.02%** | **87.02%** | **87.02%** | **145.0 ms** |

---

## CHƯƠNG 6: KẾT LUẬN & HƯỚNG PHÁT TRIỂN TƯƠNG LAI
1. **Đóng góp:** Hiện thực hóa thành công Pipeline Fact-checking tiếng Việt End-to-End từ văn bản thô, giải quyết hiện tượng The Reality Drop bằng Context Window Expansion.
2. **Hướng phát triển:**
   - Contrastive Dense Retrieval (BGE-M3 / Contriever tiếng Việt).
   - Coreference Resolution Pipeline giải quyết đại từ trước khi tách câu.
   - Cross-Encoder Reranker học tương quan ngữ nghĩa phi tuyến tính.
