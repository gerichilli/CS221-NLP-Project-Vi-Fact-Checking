"""
HỆ THỐNG XÁC MINH TIN TỨC TIẾNG VIỆT (VIETNAMESE FACT-CHECKING)
Đồ án môn học: CS221 - Xử lý Ngôn ngữ Tự nhiên (NLP)
Đối sánh Đa mô hình: Baseline (Gold Evidence) vs Reranking Pipeline (Retrieved Evidence)
"""

import random
from pathlib import Path
import pandas as pd
import streamlit as st

# ==============================================================================
# CẤU HÌNH TRANG WEB & ÉP LIGHT MODE TUYỆT ĐỐI
# ==============================================================================
st.set_page_config(
    page_title="ViFactCheck — Đối Sánh & Phân Tích Mô Hình",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS ép Light Mode toàn diện, loại bỏ 100% hiện tượng chữ trắng trên nền sáng
st.markdown("""
<style>
    /* 1. Ép màu nền và màu chữ toàn cục */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* 2. Tiêu đề và nhãn */
    h1, h2, h3, h4, h5, h6, label, p, span, div, li, b, i {
        color: #0F172A !important;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A !important;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569 !important;
        margin-bottom: 1.2rem;
    }

    /* 3. Ô Nhập liệu (Textarea, TextInput, Selectbox) - Luôn có chữ đen đậm, nền trắng */
    .stTextArea textarea, .stTextInput input {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    .stTextArea textarea:disabled {
        background-color: #F1F5F9 !important;
        color: #334155 !important;
    }

    /* 4. Thẻ Mô hình (Model Cards) */
    .model-card {
        background-color: #FFFFFF !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .model-card * {
        color: #0F172A !important;
    }

    /* 5. Huy hiệu Trạng thái & Nhãn */
    .status-correct {
        color: #14532D !important;
        background-color: #DCFCE7 !important;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        border: 1px solid #86EFAC;
        display: inline-block;
    }
    .status-wrong {
        color: #7F1D1D !important;
        background-color: #FEE2E2 !important;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        border: 1px solid #FCA5A5;
        display: inline-block;
    }
    .badge-sup {
        background-color: #DCFCE7 !important;
        color: #14532D !important;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        border: 1.5px solid #86EFAC;
        display: inline-block;
    }
    .badge-ref {
        background-color: #FEE2E2 !important;
        color: #7F1D1D !important;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        border: 1.5px solid #FCA5A5;
        display: inline-block;
    }
    .badge-nei {
        background-color: #F1F5F9 !important;
        color: #1E293B !important;
        padding: 4px 14px;
        border-radius: 16px;
        font-weight: 700;
        border: 1.5px solid #CBD5E1;
        display: inline-block;
    }

    /* 6. Thẻ Phân tích Chuyên sâu */
    .analysis-box {
        background-color: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 8px;
        padding: 16px;
        line-height: 1.7;
        margin-top: 8px;
    }
    .analysis-box * {
        color: #0F172A !important;
    }
    .rescue-callout {
        background-color: #ECFDF5 !important;
        border-left: 4px solid #10B981 !important;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        color: #064E3B !important;
    }
    .rescue-callout * {
        color: #064E3B !important;
    }
</style>
""", unsafe_allow_html=True)

# Đường dẫn file dữ liệu tổng hợp
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data/processed/unified_demo_predictions.csv"

@st.cache_data
def load_unified_data():
    if not DATA_PATH.exists():
        st.error(f"Không tìm thấy file: {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    # Chuẩn hóa nhãn thành chữ hoa
    for col in ['gold_label', 'pred_tfidf_gold', 'pred_phobert_gold', 'pred_bamibert_gold', 
                'pred_phobert_rerank', 'pred_bamibert_rerank', 'pred_bamibert_top2', 'pred_bamibert_window']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
    return df

df_all = load_unified_data()

# ==============================================================================
# SIDEBAR: BỘ LỌC & DANH SÁCH CÂU VÍ DỤ TỪ DATASET
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/balance-scale.png", width=60)
    st.markdown("### 🗂️ Chọn Câu Từ Dataset (723 mẫu)")

    # 1. Lọc theo tình huống kinh điển
    scenario_filter = st.selectbox(
        "Tình huống nghiên cứu:",
        [
            "Tất cả các câu mẫu",
            "✨ Ca được giải cứu bởi Đề xuất Mở rộng Ngữ cảnh (Top-2)",
            "📉 Ca The Reality Drop (Baseline ĐÚNG -> Rerank Top-1 SAI)",
            "💡 Ca Semantic Victory (TF-IDF SAI -> BamiBERT ĐÚNG)",
            "💥 Ca Thách Thức (Tất cả mô hình đều sai)"
        ],
        index=1,
        help="Lọc các ca kinh điển minh họa cho các phát hiện khoa học của đồ án."
    )

    # 2. Lọc theo Nhãn thực tế
    label_filter = st.radio(
        "Nhãn thực tế (Ground Truth):",
        ["Tất cả", "SUPPORTED", "REFUTED", "NEI"],
        horizontal=True
    )

    # Áp dụng bộ lọc
    filtered_df = df_all.copy()
    if label_filter != "Tất cả":
        filtered_df = filtered_df[filtered_df["gold_label"] == label_filter]

    if "giải cứu" in scenario_filter:
        filtered_df = filtered_df[
            (filtered_df["pred_bamibert_gold"] == filtered_df["gold_label"]) & 
            (filtered_df["pred_bamibert_rerank"] != filtered_df["gold_label"]) &
            (filtered_df["pred_bamibert_top2"] == filtered_df["gold_label"])
        ]
    elif "The Reality Drop" in scenario_filter:
        filtered_df = filtered_df[
            (filtered_df["pred_bamibert_gold"] == filtered_df["gold_label"]) & 
            (filtered_df["pred_bamibert_rerank"] != filtered_df["gold_label"])
        ]
    elif "Semantic Victory" in scenario_filter:
        filtered_df = filtered_df[
            (filtered_df["pred_tfidf_gold"] != filtered_df["gold_label"]) & 
            (filtered_df["pred_bamibert_gold"] == filtered_df["gold_label"])
        ]
    elif "Thách Thức" in scenario_filter:
        filtered_df = filtered_df[
            (filtered_df["pred_tfidf_gold"] != filtered_df["gold_label"]) & 
            (filtered_df["pred_phobert_gold"] != filtered_df["gold_label"]) &
            (filtered_df["pred_bamibert_gold"] != filtered_df["gold_label"])
        ]

    st.caption(f"Tìm thấy **{len(filtered_df)}** câu phù hợp với tiêu chí lọc.")

    # Tạo danh sách hiển thị
    options = {}
    for idx, row in filtered_df.iterrows():
        short_stmt = (str(row['statement'])[:55] + "...") if len(str(row['statement'])) > 55 else str(row['statement'])
        options[f"[{row['claim_id']}] ({row['gold_label']}) {short_stmt}"] = idx

    # Nút chọn ngẫu nhiên
    if st.button("🎲 Chọn Ngẫu Nhiên 1 Câu", use_container_width=True):
        if len(options) > 0:
            st.session_state["selected_claim_key"] = random.choice(list(options.keys()))

    # Selectbox chọn câu
    selected_key = st.selectbox(
        "Chọn câu cần khảo sát:",
        list(options.keys()),
        key="selected_claim_key" if "selected_claim_key" in st.session_state and st.session_state["selected_claim_key"] in options else None
    )

    st.markdown("---")
    st.caption("👥 **CS221 - NLP** | ViFactCheck Project")

# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================
st.markdown('<div class="main-title">⚖️ ViFactCheck — Khảo Sát Đối Sánh & Phân Tích Dự Đoán</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">So sánh trực diện <b>Baseline (Gold Evidence)</b> vs <b>Reranking Pipeline (Retrieved Evidence)</b> kèm giải thích cơ chế suy luận & phân tích lỗi.</div>', unsafe_allow_html=True)

if not selected_key or len(filtered_df) == 0:
    st.info("Không có câu nào thỏa mãn bộ lọc hiện tại. Vui lòng nới lỏng bộ lọc ở thanh bên trái.")
    st.stop()

selected_row_idx = options[selected_key]
sample = df_all.loc[selected_row_idx]

gold_label = sample["gold_label"]
gold_badge_class = "badge-sup" if gold_label == "SUPPORTED" else ("badge-ref" if gold_label == "REFUTED" else "badge-nei")
gold_icon = "🟢" if gold_label == "SUPPORTED" else ("🔴" if gold_label == "REFUTED" else "⚪")

# Cập nhật session_state để tự động điền vào các ô nhập liệu khi chọn câu mới
if "last_claim_id" not in st.session_state or st.session_state["last_claim_id"] != sample["claim_id"]:
    st.session_state["last_claim_id"] = sample["claim_id"]
    st.session_state["claim_input_val"] = str(sample["statement"])
    st.session_state["gold_input_val"] = str(sample["gold_evidence"])
    st.session_state["rerank_input_val"] = str(sample["reranked_evidence"])

# ==============================================================================
# 1. KHU VỰC CÁC Ô NHẬP LIỆU (TỰ ĐỘNG ĐIỀN KHI CHỌN & CÓ THỂ CHỈNH SỬA)
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-size: 1.15rem; font-weight: 700; color: #1E3A8A;">📝 Dữ liệu Phát biểu & Bằng chứng của Mẫu <code>{sample['claim_id']}</code></span>
        <span>Nhãn Thực Tế: <span class="{gold_badge_class}">{gold_icon} {gold_label}</span></span>
    </div>
    """,
    unsafe_allow_html=True
)

# Ô 1: Tuyên bố (Claim)
user_statement = st.text_area(
    "1. Tuyên bố cần kiểm chứng (Claim / Statement):",
    key="claim_input_val",
    height=80,
    help="Nội dung tuyên bố được tự động điền từ dataset khi bạn chọn câu bên trái. Bạn có thể sửa trực tiếp nếu muốn."
)

col_in_gold, col_in_rerank = st.columns(2)

with col_in_gold:
    # Ô 2: Bằng chứng Vàng (Gold Evidence)
    user_gold = st.text_area(
        "2. Bằng chứng Vàng (Gold Evidence) — [Dành cho Baseline]:",
        key="gold_input_val",
        height=110,
        help="Bằng chứng lý tưởng do chuyên gia gán nhãn trong bài báo."
    )

with col_in_rerank:
    # Ô 3: Bằng chứng Truy xuất (Retrieved Top-1)
    user_rerank = st.text_area(
        "3. Bằng chứng Truy xuất (IR / Fact Rerank Top-1) — [Dành cho Pipeline]:",
        key="rerank_input_val",
        height=110,
        help="Câu đơn lẻ có điểm BM25 + Fact-aware Rerank cao nhất được máy tự động trích xuất từ bài báo."
    )

# Ô 4: Bằng chứng Mở rộng Ngữ cảnh (Top-2 Concatenated)
st.text_area(
    "4. Bằng chứng Mở rộng Ngữ cảnh (Top-2 Concatenated) — [Đề xuất Cải tiến của Đồ án]:",
    value=str(sample["top2_evidence"]),
    height=80,
    disabled=True,
    help="Ghép 2 câu bằng chứng có điểm cao nhất để giải quyết hiện tượng Đói ngữ cảnh (Context Starvation)."
)

st.markdown("---")

# ==============================================================================
# 2. BẢNG ĐỐI SÁNH DỰ ĐOÁN 2 NHÓM MÔ HÌNH (SIDE-BY-SIDE)
# ==============================================================================
st.markdown("### 🥊 Bảng Đối Sánh Dự Đoán Trực Tiếp")

col_base, col_rerank = st.columns(2)

def render_model_result(model_name: str, pred_label: str, true_label: str, input_desc: str):
    is_corr = (pred_label == true_label)
    status_html = '<span class="status-correct">✅ ĐÚNG</span>' if is_corr else '<span class="status-wrong">❌ SAI</span>'
    p_badge = "badge-sup" if pred_label == "SUPPORTED" else ("badge-ref" if pred_label == "REFUTED" else "badge-nei")
    p_icon = "🟢" if pred_label == "SUPPORTED" else ("🔴" if pred_label == "REFUTED" else "⚪")
    
    st.markdown(
        f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #1E3A8A; font-size: 1.0rem;">{model_name}</span>
                {status_html}
            </div>
            <div style="font-size: 0.85rem; color: #64748B; margin: 4px 0;">Đầu vào: <i>{input_desc}</i></div>
            <div style="margin-top: 6px;">
                Dự đoán: <span class="{p_badge}">{p_icon} {pred_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_base:
    st.markdown("#### 🏛️ Nhóm 1: BASELINE (Đọc Bằng chứng Vàng)")
    render_model_result(
        "1. TF-IDF + Logistic Regression",
        sample["pred_tfidf_gold"],
        gold_label,
        "Ô 2: Bằng chứng Vàng (Túi từ - BoW)"
    )
    render_model_result(
        "2. PhoBERT (Base)",
        sample["pred_phobert_gold"],
        gold_label,
        "Ô 2: Bằng chứng Vàng (VnCoreNLP Tokenized)"
    )
    render_model_result(
        "3. BamiBERT (Exp006 SOTA)",
        sample["pred_bamibert_gold"],
        gold_label,
        "Ô 2: Bằng chứng Vàng (Semantic Prefix Prompt)"
    )
    st.caption("ℹ️ *Ghi chú:* Nhóm Baseline phản ánh trần hiệu năng lý tưởng (Upper-bound) khi dữ liệu sạch không nhiễu.")

with col_rerank:
    st.markdown("#### 🔄 Nhóm 2: RERANKING PIPELINE (Đọc Bằng chứng IR)")
    render_model_result(
        "1. PhoBERT (BM25 + Fact Rerank)",
        sample["pred_phobert_rerank"],
        gold_label,
        "Ô 3: Câu Top-1 trích xuất từ bài báo"
    )
    render_model_result(
        "2. BamiBERT (BM25 + Fact Rerank)",
        sample["pred_bamibert_rerank"],
        gold_label,
        "Ô 3: Câu Top-1 trích xuất từ bài báo"
    )
    render_model_result(
        "3. BamiBERT + Top-2 Context Expansion",
        sample["pred_bamibert_top2"],
        gold_label,
        "Ô 4: Ghép 2 câu BM25 + Fact Rerank (Đề xuất Cải tiến)"
    )
    if "pred_bamibert_hybrid_top2" in sample and pd.notna(sample["pred_bamibert_hybrid_top2"]):
        render_model_result(
            "4. BamiBERT (Hybrid SBERT+BM25 + Top-2)",
            sample["pred_bamibert_hybrid_top2"],
            gold_label,
            "Dense SBERT + BM25 + Fact Rerank Top-2 (Module 7.2b)"
        )
    st.caption("ℹ️ *Ghi chú:* Nhóm Reranking phản ánh hiệu năng thực tế khi tích hợp bộ tìm kiếm thông tin từ văn bản bài báo.")

st.markdown("---")

# ==============================================================================
# 3. PHÂN TÍCH CHUYÊN SÂU: TẠI SAO LẠI CÓ DỰ ĐOÁN NHƯ VẬY & PHÂN TÍCH LỖI
# ==============================================================================
st.markdown("### 🧠 Phân Tích Chuyên Sâu: Cơ Chế Dự Đoán & Giải Mã Lỗi Sai")

col_why, col_err = st.columns(2)

with col_why:
    st.markdown("#### 1. Tại sao các mô hình lại dự đoán như vậy?")
    
    # Phân tích cơ chế TFIDF
    tfidf_pred = sample["pred_tfidf_gold"]
    if tfidf_pred == gold_label:
        tfidf_reason = "Khớp được các từ khóa cốt lõi giữa Tuyên bố và Bằng chứng vàng."
    else:
        tfidf_reason = "Mô hình túi từ (Bag-of-Words) bị 'mù' trước cấu trúc ngữ pháp phủ định hoặc mối quan hệ bắc cầu giữa các vế câu."

    # Phân tích cơ chế BamiBERT Gold
    bami_gold_pred = sample["pred_bamibert_gold"]
    if bami_gold_pred == gold_label:
        bami_gold_reason = "Cơ chế Self-Attention đa tầng của BamiBERT kết hợp với tiền tố 'Tuyên bố: ... Bằng chứng: ...' đã kích hoạt trúng mối quan hệ suy luận logic."
    else:
        bami_gold_reason = "Mô hình gặp khó khăn trước các câu phát biểu có nhiều phủ định kép hoặc thực thể chưa từng xuất hiện trong tập huấn luyện."

    # Phân tích cơ chế Reranking
    bami_rerank_pred = sample["pred_bamibert_rerank"]
    is_reality_drop = (bami_gold_pred == gold_label and bami_rerank_pred != gold_label)
    
    if is_reality_drop:
        rerank_reason = f"<b>Hiện tượng The Reality Drop:</b> Câu trích xuất Top-1 <i>('{str(sample['reranked_evidence'])[:75]}...')</i> bị <b>đói ngữ cảnh (Context Starvation)</b>, thiếu mất một vế quan trọng so với Bằng chứng Vàng, khiến mô hình bị nhầm sang <b>{bami_rerank_pred}</b>."
    else:
        rerank_reason = "Câu trích xuất Top-1 đã chứa đủ các từ khóa then chốt và thực thể trùng khớp với Tuyên bố."

    st.markdown(
        f"""
        <div class="analysis-box">
            <b>🔹 Cơ chế của Baseline (TF-IDF vs Transformer):</b>
            <ul style="margin-top: 4px; padding-left: 20px;">
                <li><b>TF-IDF:</b> {tfidf_reason}</li>
                <li><b>BamiBERT Gold:</b> {bami_gold_reason}</li>
            </ul>
            <b>🔹 Cơ chế của Reranking Pipeline:</b>
            <p style="margin-top: 4px;">{rerank_reason}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_err:
    st.markdown("#### 2. Phân tích Lỗi Sai & Giải pháp Giải cứu")
    
    any_wrong = (
        sample["pred_tfidf_gold"] != gold_label or
        sample["pred_phobert_gold"] != gold_label or
        sample["pred_bamibert_gold"] != gold_label or
        sample["pred_phobert_rerank"] != gold_label or
        sample["pred_bamibert_rerank"] != gold_label
    )

    if not any_wrong:
        st.success("🎉 **Ca hoàn hảo:** Toàn bộ các mô hình (cả Baseline lẫn Reranking Pipeline) đều đưa ra phán quyết CHÍNH XÁC!")
    else:
        error_tags = []
        error_details = []

        if sample["pred_tfidf_gold"] != gold_label and sample["pred_bamibert_gold"] == gold_label:
            error_tags.append("Bẫy Từ vựng Cục bộ (Lexical Trap trên TF-IDF)")
            error_details.append("TF-IDF chỉ đếm tần suất từ bề mặt, không hiểu quan hệ entailment/refutation ngữ nghĩa như Transformer.")

        if sample["pred_bamibert_rerank"] != gold_label and sample["pred_bamibert_gold"] == gold_label:
            error_tags.append("Đói Ngữ Cảnh (Context Starvation / The Reality Drop)")
            error_details.append("Câu đơn lẻ trích xuất từ IR bị đứt đoạn, mất liên kết chủ ngữ hoặc thiếu mệnh đề bổ trợ so với Bằng chứng Vàng.")

        if sample["pred_phobert_rerank"] == "NEI" or sample["pred_bamibert_rerank"] == "NEI":
            if gold_label in ["SUPPORTED", "REFUTED"]:
                error_tags.append("Rơi vào bẫy Chưa đủ thông tin (NEI Trap)")
                error_details.append(f"Mô hình không dám khẳng định {gold_label} vì câu trích xuất thiếu dữ liệu kiểm chứng trực diện.")

        is_top2_rescued = (sample["pred_bamibert_rerank"] != gold_label and sample["pred_bamibert_top2"] == gold_label)
        rescue_html = ""
        if is_top2_rescued:
            rescue_html = f"""
            <div class="rescue-callout">
                <b>🚀 Đề xuất Cải tiến Top-2 Expansion đã GIẢI CỨU thành công:</b><br>
                Khi ghép thêm câu thứ 2 từ bài báo:
                <br><i>"{str(sample['top2_evidence'])[:120]}..."</i><br>
                Mô hình BamiBERT đã được bổ sung đầy đủ ngữ cảnh bị thiếu và <b>lật ngược phán quyết từ SAI ({sample['pred_bamibert_rerank']}) sang ĐÚNG ({sample['pred_bamibert_top2']})</b>!
            </div>
            """

        tags_html = " ".join([f"<span class='status-wrong' style='margin-right: 5px; margin-bottom: 4px;'>⚠️ {t}</span>" for t in error_tags])
        details_html = "".join([f"<li style='margin-bottom: 4px;'>{d}</li>" for d in error_details])

        st.markdown(
            f"""
            <div class="analysis-box">
                <b>Phân loại bản chất lỗi:</b><br>
                <div style="margin: 8px 0;">{tags_html}</div>
                <ul style="padding-left: 20px; margin-top: 4px;">{details_html}</ul>
                {rescue_html}
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================================================================
# 4. EXPANDER XEM VĂN BẢN GỐC TOÀN BỘ BÀI BÁO (CONTEXT)
# ==============================================================================
with st.expander("📰 Xem toàn bộ bài báo ngữ cảnh gốc (Context Article)"):
    st.markdown(
        f'<div style="line-height: 1.8; color: #0F172A !important; font-size: 0.95rem; background: #F8FAFC; padding: 14px; border-radius: 6px; border: 1px solid #E2E8F0;">{sample["context"]}</div>', 
        unsafe_allow_html=True
    )

# ==============================================================================
# 5. EXPANDER: BÁO CÁO ĐỐI SÁNH KHOA HỌC (SBERT & KHẢO SÁT TOP-K BamiBERT)
# ==============================================================================
with st.expander("📊 Xem Báo Cáo Đối Sánh Thực Nghiệm: SBERT/Hybrid & Khảo Sát Top-K (1..5)"):
    tab_matrix, tab_h2h, tab_topk, tab_ir = st.tabs([
        "🎯 BamiBERT Toàn Diện (4 Chiến Lược x K=1..5)",
        "🥊 Đối Đầu Trực Diện: BamiBERT vs. PhoBERT",
        "📈 Khảo sát Top-K BamiBERT (K=1..5)", 
        "🔍 Đối sánh IR Tầng 1 (BM25 vs SBERT vs Hybrid)"
    ])

    with tab_h2h:
        st.markdown("#### 🥊 So Sánh Trực Diện: BamiBERT vs. PhoBERT Trên Toàn Bộ 723 Mẫu Dev")
        st.caption("Đối chứng 2 mô hình ngôn ngữ tiếng Việt (cùng kích thước Base) trên các chiến lược truy xuất và Top-K từ 1 đến 5.")
        col_h1, col_h2 = st.columns([1.3, 1])
        with col_h1:
            h2h_chart = PROJECT_ROOT / "notebooks/IR_IE_Reranking/outputs/11_head_to_head_phobert_vs_bamibert_curve.png"
            if h2h_chart.exists():
                st.image(str(h2h_chart), caption="Biểu đồ đối đầu BamiBERT vs PhoBERT", use_container_width=True)
        with col_h2:
            st.markdown(r"""
            **Bảng Đối Sánh Điểm Tối Ưu ($K=2$):**
            | Mô hình | BM25 Baseline | Hybrid + Fact Rerank (Đề xuất) | Chênh lệch ($\Delta$) |
            | :--- | :---: | :---: | :---: |
            | **PhoBERT-base** | 64.62% | **65.67%** | +1.05% |
            | **BamiBERT** | 71.60% | **72.03%** ⭐ | **+6.36%** vs PhoBERT |

            💡 **Nhận xét thực nghiệm:**
            1. **BamiBERT đạt kết quả cao hơn** PhoBERT từ **+2.0% đến +6.4% Macro F1** trên tất cả các kịch bản $K$.
            2. **Điểm tối ưu thực nghiệm:** Cả hai mô hình đều đạt hiệu năng cao nhất tại $K^*=2$ khi kết hợp phương pháp đề xuất **Hybrid + Fact Rerank**.
            """)
    
    with tab_matrix:
        st.markdown("#### 🏆 Đối Sánh Trực Diện 4 Chiến Lược Truy Xuất Đưa Vào BamiBERT ($K=1 \\to 5$)")
        st.caption("Khảo sát toàn diện theo phong cách Figure 4 của bài báo ViFactCheck trên toàn bộ 723 mẫu Dev (20 lượt chạy thực nghiệm).")
        col_m1, col_m2 = st.columns([1.3, 1])
        with col_m1:
            chart_matrix = PROJECT_ROOT / "notebooks/IR_IE_Reranking/outputs/09_end_to_end_model_comparison_chart.png"
            if chart_matrix.exists():
                st.image(str(chart_matrix), caption="Biểu đồ hiệu năng BamiBERT (Macro F1) trên 4 chiến lược truy xuất", use_container_width=True)
        with col_m2:
            matrix_path = PROJECT_ROOT / "notebooks/IR_IE_Reranking/outputs/09_end_to_end_model_results_matrix.csv"
            if matrix_path.exists():
                df_mat = pd.read_csv(matrix_path)
                # Pivot for neat presentation
                df_pivot = df_mat.pivot(index="Method", columns="Top_K", values="Macro_F1")
                st.markdown("**Bảng Macro F1 (%) theo từng $K$:**")
                st.dataframe(df_pivot, use_container_width=True)
            st.success("💡 **Điểm mấu chốt:**\n"
                       "- **Hybrid + Fact Rerank (Đề xuất)** đạt đỉnh cao nhất (**72.03% F1** tại $K=2$).\n"
                       "- **Pure SBERT** thấp hơn đáng kể (~63% - 67%) vì bỏ sót tên riêng, mốc thời gian, số liệu chính xác.\n"
                       "- **Hybrid** vượt trội ở $K=1$ (**61.95%**) và $K=3$ (**70.44%**), chứng minh việc kết hợp từ khóa + ngữ nghĩa giúp câu bằng chứng chặt chẽ hơn.")

    with tab_topk:
        st.markdown("#### 🏆 Khảo Sát Ảnh Hưởng Của Số Lượng Bằng Chứng ($K=1 \\to 5$) Đến BamiBERT")
        st.caption("Thực nghiệm độc lập trên 723 mẫu tập Dev đối chứng hiện tượng Context Starvation vs Attention Distraction.")
        col_img1, col_tbl1 = st.columns([1.2, 1])
        with col_img1:
            chart_topk = PROJECT_ROOT / "notebooks/IR_IE_Reranking/outputs/07_bamibert_topk_survey_curve.png"
            if chart_topk.exists():
                st.image(str(chart_topk), caption="Biểu đồ hiệu năng BamiBERT theo Top-K (Optimal Peak K*=2)", use_container_width=True)
        with col_tbl1:
            metrics_topk_path = PROJECT_ROOT / "notebooks/IR_IE_Reranking/outputs/07_bamibert_topk_survey_metrics.csv"
            if metrics_topk_path.exists():
                st.dataframe(pd.read_csv(metrics_topk_path), use_container_width=True, hide_index=True)
            st.info("💡 **Kết luận:** $K^*=2$ là điểm cân bằng vàng (+11.70% Macro F1), các mức $K \\ge 3$ suy giảm do lẫn câu nhiễu ngoài lề.")

    with tab_ir:
        st.markdown("#### 🎯 So Sánh 3 Chiến Lược Truy Xuất Bằng Chứng Tầng 1 (First-Stage IR)")
        st.caption("Đánh giá độ phủ bằng chứng (Recall@K) và MRR trên 500 mẫu có Ground-truth Evidence.")
        col_img2, col_tbl2 = st.columns([1.2, 1])
        with col_img2:
            chart_ir = PROJECT_ROOT / "notebooks/IR_IE_Reranking/outputs/08_stage1_ir_comparison.png"
            if chart_ir.exists():
                st.image(str(chart_ir), caption="Đối sánh Recall@K giữa BM25 vs SBERT vs Hybrid", use_container_width=True)
        with col_tbl2:
            metrics_ir_path = PROJECT_ROOT / "notebooks/IR_IE_Reranking/outputs/08_stage1_ir_comparison.csv"
            if metrics_ir_path.exists():
                st.dataframe(pd.read_csv(metrics_ir_path), use_container_width=True, hide_index=True)
            st.success("💡 **Kết luận:** Hybrid Search (BM25 + SBERT) đạt Recall@1 cao nhất (**90.00%**) và Recall@5 đạt **97.20%**, kết hợp tối ưu giữa từ khóa và ngữ nghĩa vector.")


