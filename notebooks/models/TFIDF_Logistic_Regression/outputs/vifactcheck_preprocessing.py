import re
import unicodedata
import numpy as np
import pandas as pd
from underthesea import word_tokenize
STATEMENT_COL="statement"
EVIDENCE_COL="evidence"
# Tạo biểu thức chính quy nhận diện URL.
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Tạo biểu thức chính quy nhận diện thẻ HTML.
HTML_PATTERN = re.compile(r"<[^>]+>")

# Tạo biểu thức chính quy nhận diện khoảng trắng lặp.
SPACE_PATTERN = re.compile(r"\s+")

# Tạo biểu thức xóa phần lớn dấu câu nhưng giữ dấu chấm, dấu phẩy và phần trăm gắn với dữ kiện số.
PUNCT_PATTERN = re.compile(r"[^\w\s%.,]", flags=re.UNICODE)

# Tạo hàm làm sạch văn bản có tham số để phục vụ thí nghiệm.
LEXICAL_MAP = {"hoà": "hòa", "hoá": "hóa", "thuỷ": "thủy", "xoá": "xóa", "toà": "tòa", "khoẻ": "khỏe"}
LEXICAL_PATTERN = re.compile(r"\b(?:" + "|".join(map(re.escape, LEXICAL_MAP)) + r")\b")

def preprocess_text(text, lowercase=False, remove_punctuation=False, remove_stopwords=False, stopwords=None, normalize_lexical=False):
    # Kiểm tra trường hợp một ô chứa danh sách nhiều đoạn Evidence.
    if isinstance(text, (list, tuple, np.ndarray)):
        # Ghép các đoạn Evidence thành một chuỗi, không tự chia theo dấu câu.
        text = " ".join(str(item) for item in text)
    # Xử lý trường hợp một ô chỉ chứa một giá trị thông thường.
    else:
        # Đổi null thành chuỗi rỗng, còn giá trị hợp lệ thành chuỗi.
        text = "" if pd.isna(text) else str(text)
    # Chuẩn hóa Unicode về NFC để các ký tự tiếng Việt có cùng biểu diễn mã.
    text = unicodedata.normalize("NFC", text)
    # Thay URL bằng khoảng trắng vì URL thường là nhiễu đối với baseline này.
    text = URL_PATTERN.sub(" ", text)
    # Thay thẻ HTML bằng khoảng trắng.
    text = HTML_PATTERN.sub(" ", text)
    # Đưa về chữ thường nếu cấu hình thí nghiệm yêu cầu.
    if lowercase:
        # Chuyển toàn bộ văn bản sang chữ thường.
        text = text.lower()
    if normalize_lexical:
        text = LEXICAL_PATTERN.sub(lambda match: LEXICAL_MAP[match.group(0)], text)
    # Xóa dấu câu nếu cấu hình thí nghiệm yêu cầu.
    if remove_punctuation:
        # Thay phần lớn dấu câu bằng khoảng trắng nhưng vẫn giữ chữ, số, dấu thập phân và phần trăm.
        text = PUNCT_PATTERN.sub(" ", text)
    # Tách từ tiếng Việt và nối từ ghép bằng dấu gạch dưới.
    text = word_tokenize(text, format="text")
    # Xóa stopwords nếu cấu hình thí nghiệm yêu cầu.
    if remove_stopwords:
        # Bảo đảm danh sách stopwords đã được truyền vào.
        if not stopwords:
            raise ValueError("Bật stopwords cần danh sách không rỗng đã được kiểm tra")
        stopwords = set(stopwords)
        protected = {"không", "chưa", "chẳng", "chả", "đừng", "không_phải", "chưa_từng"}
        if protected & {word.lower() for word in stopwords}:
            raise ValueError("Stopword list chứa từ phủ định cần giữ")
        # Chỉ giữ token không nằm trong danh sách stopwords.
        text = " ".join(token for token in text.split() if token not in stopwords)
    # Gộp mọi khoảng trắng lặp thành một khoảng trắng.
    text = SPACE_PATTERN.sub(" ", text).strip()
    # Trả về văn bản đã xử lý.
    return text

def build_model_text(df, config, stopwords=None):
    # Xử lý từng Statement theo cấu hình.
    processed_statement = df[STATEMENT_COL].apply(
        lambda value: preprocess_text(value, stopwords=stopwords, **config)
    )
    # Xử lý từng Evidence theo cùng cấu hình.
    processed_evidence = df[EVIDENCE_COL].apply(
        lambda value: preprocess_text(value, stopwords=stopwords, **config)
    )
    # Ghép Statement và Evidence thành một chuỗi đầu vào theo đúng yêu cầu baseline.
    combined_text = processed_statement + " " + processed_evidence
    # Gộp khoảng trắng và bỏ khoảng trắng hai đầu sau khi ghép.
    combined_text = combined_text.str.replace(r"\s+", " ", regex=True).str.strip()
    # Trả về ba Series để vừa huấn luyện vừa có thể kiểm tra từng phần.
    return processed_statement, processed_evidence, combined_text


