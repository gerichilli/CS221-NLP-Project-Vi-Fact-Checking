# DÀN Ý & NỘI DUNG SLIDE THUYẾT TRÌNH ĐỒ ÁN CS221
## Đề tài: Kiểm chứng thông tin tiếng Việt (Vietnamese Fact-Checking) Tự động End-to-End

---

### Slide 1: Trang Tiêu Đề
- **Tên đề tài:** KIỂM CHỨNG THÔNG TIN TIẾNG VIỆT TỰ ĐỘNG (VIETNAMESE FACT-CHECKING)
- **Phụ đề:** Từ Truy xuất Bằng chứng (IR/IE/Reranking) đến Suy luận Ngữ nghĩa NLI End-to-End
- **Môn học:** CS221 — Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing)
- **Bộ dữ liệu:** ViFactCheck (7.232 mẫu)
- **Các mô hình:** TF-IDF Baseline, PhoBERT Base v2, BamiBERT SOTA Exp006, Qwen2.5-1.5B LoRA

---

### Slide 2: Đặt Vấn Đề & Bối Cảnh Thực Tế
- **Phòng thí nghiệm (Lab Benchmark):** Cung cấp sẵn cặp (Claim, Gold Evidence). Mô hình chỉ việc đọc hiểu logic. Rất xa rời thực tế!
- **Thực tế đời thực:** Người dùng chỉ đưa Claim + Toàn văn bài báo thô (Context). Không có bằng chứng cắt sẵn!
- **Rào cản:** Transformer giới hạn 256/512 tokens, loãng sự chú ý (Lost in the Middle), chi phí tính toán bậc hai O(N^2).
- **Giải pháp bắt buộc:** Kiến trúc hai giai đoạn: Evidence Retrieval & Reranking + NLI Verification.

---

### Slide 3: Bộ Dữ Liệu Chuẩn ViFactCheck
- **3 Nhãn logic:**
  - `SUPPORTED`: Tuyên bố được chứng minh đúng bởi bằng chứng.
  - `REFUTED`: Tuyên bố mâu thuẫn hoặc bị bác bỏ bởi bằng chứng.
  - `NOT ENOUGH INFO (NEI)`: Không đủ cơ sở khẳng định đúng hay sai.
- **Phân bổ tập mẫu:** Train 5.062 (70%) | Dev 723 (10%) | Test 1.447 (20%).
- **Thách thức:** Đại từ thay thế ("ông", "bà", "vị này"), số thập phân ("8.5%"), từ viết tắt ("TP.HCM", "TS.").

---

### Slide 4: Kiến Trúc Hai Giai Đoạn (Two-Stage Architecture)
- **Giai đoạn 1 (IR/IE/Reranking):** Sàng lọc bài báo 30 câu xuống 1-2 câu chất lượng nhất.
  - Sentence Splitting + Masking Protection -> Okapi BM25 -> VnCoreNLP Fact Extraction -> Fact-aware Reranking -> Context Window Expansion.
- **Giai đoạn 2 (NLI Verification):** Nhận cặp (Claim, Retrieved Evidence) và phán đoán 3 nhãn.

---

### Slide 5: Tiền Xử Lý Dữ Liệu Chuẩn Hóa
- **Common Preprocessing:** Chuẩn hóa Unicode NFC, khử ký tự điều khiển `\x00-\x1f`, khử khoảng trắng đặc thù, khử HTML noise.
- **Model-specific Preprocessing:**
  - TF-IDF: n-gram từ vựng.
  - PhoBERT: BPE tokenizer, max_length=256, longest_first.
  - BamiBERT: Prefix Prompting ("Tuyên bố: ... Bằng chứng: ...").
  - LLM LoRA: ChatML Template cho Qwen.

---

### Slide 6: Các Mô Hình NLI Cơ Sở: TF-IDF vs PhoBERT
- **TF-IDF + Logistic Regression:** Accuracy 38.63%, Macro-F1 38.11% (gần như đoán mò). Bag-of-Words thất bại trước ngữ nghĩa NLI.
- **PhoBERT Base v2:** 135M tham số, pretrain 20GB tiếng Việt. Accuracy 84.73%, Macro-F1 84.63% (trên Gold Evidence). Nắm bắt tốt ngữ nghĩa hai câu.

---

### Slide 7: BamiBERT SOTA (Chuỗi Thực Nghiệm Exp001 – Exp006)
- Chuỗi thực nghiệm: Baseline (82.35%) -> Label Smoothing (83.10%) -> Focal Loss (83.45%) -> Dynamic Weighting (83.80%) -> Cosine LR (84.15%) -> Prefix Prompting (84.84%).
- **Exp006 (Prefix Prompting):** Thêm tiền tố định danh giúp tách bạch hai vế, kéo F1 nhãn khó `REFUTED` từ 75.00% lên 81.62%!

---

### Slide 8: Large Language Model: Qwen2.5-1.5B-Instruct LoRA
- **Cấu hình:** LoRA rank r=16, alpha=32, chỉ cập nhật ~0.28% tham số trên GPU T4.
- **Kết quả:** Test Accuracy 87.01%, Test Macro-F1 87.02% (Dẫn đầu tuyệt đối về F1).
- **Đánh đổi:** Tốc độ suy luận ~145 ms/mẫu (chậm hơn BamiBERT 6 lần).

---

### Slide 9: Pipeline Truy Xuất: Phân Đoạn Câu & Okapi BM25
- **Masking Protection:** Tạm che dấu chấm trong số thập phân ("8.5%") và từ viết tắt ("TP.HCM") bằng `__DOT__` trước khi tách câu. Thu được 12.823 câu ứng viên hoàn hảo.
- **Okapi BM25:** Bão hòa tần số từ (k1=1.5) và chuẩn hóa độ dài câu (b=0.75). Recall@1 = 89.80%, Recall@5 = 96.60%.

---

### Slide 10: Khắc Phục Điểm Mù Của BM25 Bằng Fact Extraction (IE)
- **Điểm mù của BM25:** BM25 chỉ đếm từ trùng lặp. Câu sai năm hoặc sai số tiền vẫn được BM25 chấm điểm cao!
- **3 Trụ cột Fact của nhóm:**
  1. Named Entities (NER): Bóc tách Tên người, Tổ chức, Địa danh qua VnCoreNLP.
  2. Temporal Facts: Regex bắt ngày/tháng/năm, quý, thế kỷ.
  3. Numerical Facts: Temporal Masking bóc tách %, số kèm đơn vị.
- Fact Score = 0.4×Entity + 0.4×Number + 0.2×Date.

---

### Slide 11: Fact-Aware Evidence Reranking (Điểm Mới Của Nhóm 💡)
- **Công thức:** `Rerank_Score = 0.70 × BM25_norm + 0.30 × Fact_Score`.
- **Hiệu quả:** Thăng hạng thành công 19 câu Bằng chứng Vàng bị BM25 xếp ở vị trí 2, 3 lên thẳng Top-1!
- Recall@1 tăng lên 90.00%, MRR đạt 0.9299.

---

### Slide 12: Hiện Tượng "The Reality Drop" Khi Tích Hợp NLI
- Khi chạy NLI trên Bằng chứng máy tự tìm Top-1:
  - PhoBERT: 72.77% -> 58.23% (-14.54% Macro-F1).
  - BamiBERT: 81.42% -> 62.42% (-19.00% Macro-F1).
- Nghịch lý: Tại sao bộ truy xuất Recall@5 đạt 96.60% mà NLI lại tụt dốc nghiêm trọng?

---

### Slide 13: Nguyên Nhân Cốt Lõi: Căn Bệnh "Context Starvation"
- **Nguyên nhân:** Do CHỈ LẤY ĐÚNG 1 CÂU ĐƠN LẺ!
  1. Mất đại từ thay thế: Câu Top-1 là "Bà từng trải qua kỳ nghỉ trăng mật...". NLI không biết "Bà" là ai (chủ ngữ nằm ở câu trước) ==> Đoán nhầm thành NEI!
  2. Bằng chứng phân mảnh nhiều câu: Claim có 2 mệnh đề (năm và địa điểm), mỗi mệnh đề nằm ở 1 câu. Lấy 1 câu sẽ luôn thiếu bằng chứng.

---

### Slide 14: Đề Xuất 1: Hybrid Search & Hiện Tượng BERT Anisotropy
- **Thực nghiệm Hybrid Search:** PhoBERT Dense Cosine + BM25 + Fact Score. Recall@1 đạt 89.40% - 90.00% (chưa tạo ra bước nhảy vọt).
- **Lý thuyết BERT Anisotropy:** BERT chưa qua Contrastive Loss bị co cụm không gian vector vào một hình nón hẹp, cosine similarity giữa 2 câu bất kỳ đều cao (~0.85-0.92), làm mất độ sắc nét phân biệt so với từ khóa chính xác của BM25.

---

### Slide 15: Đề Xuất 3: Mở Rộng Ngữ Cảnh — Bước Đột Phá Ngoạn Mục 🔥
- **Chiến lược 3A (Top-2 Concatenation):** Ghép 2 câu điểm cao nhất. Recall@2 đạt 95.40%.
  - PhoBERT: 58.23% -> 64.05% (+5.82% F1) 🚀
  - BamiBERT: 62.42% -> 70.39% (+7.97% F1) 🔥
- **Chiến lược 3B (Sliding Window ±1):** Lấy câu trước và câu sau của câu Top-1 từ bài báo gốc. Khôi phục hoàn toàn đại từ và mạch lạc câu chuyện.
  - PhoBERT: 58.23% -> 62.75% (+4.52% F1)
  - BamiBERT: 62.42% -> 69.10% (+6.68% F1)

---

### Slide 16: Bảng So Sánh Đối Chiếu Toàn Diện (Dev 723 Claims)
| Kịch bản Thực nghiệm | PhoBERT Macro-F1 | BamiBERT Macro-F1 | BamiBERT Lead |
| :--- | :---: | :---: | :---: |
| 1. Gold Evidence (Lý tưởng) | 72.77% | 81.42% | +8.65% |
| 2. BM25 Top-1 (Từ khóa) | 58.53% | 62.42% | +3.89% |
| 3. Fact Rerank Top-1 | 58.23% | 62.42% | +4.19% |
| 4. Đề xuất 3B (Sliding Window ±1) | 62.75% (+4.52%) | 69.10% (+6.68%) | +6.35% |
| 5. Đề xuất 3A (Top-2 Concat) — SOTA | **64.05% (+5.82%)** | **70.39% (+7.97%)** | **+6.34%** |

---

### Slide 17: Phân Tích Lỗi Định Tính
- Lỗi phủ định ẩn & mơ hồ từ vựng: "chưa phát hiện vi phạm" bị nhầm thành "không vi phạm".
- Lỗi trích dẫn gián tiếp: Gán nhãn sự thật tuyệt đối cho một lời khai chưa kiểm chứng ("theo lời nhân chứng").

---

### Slide 18: Đóng Góp Chính Của Đồ Án
1. Xây dựng hoàn chỉnh Pipeline Fact-checking tiếng Việt End-to-End từ văn bản thô.
2. Đề xuất mới: Fact-aware Reranking & Context Window Expansion (bứt phá +8.0% F1).
3. Đánh giá toàn diện 4 họ mô hình (TF-IDF, PhoBERT, BamiBERT, LLM LoRA).

---

### Slide 19: Hướng Phát Triển Tương Lai
1. Contrastive Dense Retrieval (BGE-M3 / Contriever tiếng Việt) giải quyết triệt để Anisotropy.
2. Coreference Resolution Pipeline giải quyết đại từ trước khi tách câu.
3. Cross-Encoder Reranker học tương quan ngữ nghĩa phi tuyến tính.

---

### Slide 20: Q&A và Lời Cảm Ơn
- Cảm ơn Thầy Cô và các bạn đã lắng nghe!
- Mời Thầy Cô và các bạn đặt câu hỏi.
