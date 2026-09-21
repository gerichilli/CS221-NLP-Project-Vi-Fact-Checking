"""
HỆ THỐNG XÁC MINH TIN TỨC TIẾNG VIỆT (VIETNAMESE FACT-CHECKING)
Đồ án môn học: CS221 - Xử lý Ngôn ngữ Tự nhiên (NLP)
So Sánh Đa Mô Hình: Baseline (Gold Evidence) vs Reranking Pipeline (Retrieved Evidence)
"""

import difflib
import html
import random
from pathlib import Path
import pandas as pd
import streamlit as st

# ==============================================================================
# CẤU HÌNH TRANG WEB & ÉP LIGHT MODE
# ==============================================================================
st.set_page_config(
    page_title="ViFactCheck — So Sánh & Phân Tích Dự Đoán",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS ép Light Mode toàn diện, loại bỏ chữ trắng trên nền sáng
st.markdown("""
<style>
    /* 1. Nền và màu chữ toàn cục */
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
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569 !important;
        margin-bottom: 1.2rem;
    }

    /* 3. Ô Nhập liệu Claim */
    .stTextArea textarea {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }

    /* 4. Thẻ Bằng Chứng (Sử dụng thẻ p - không dùng text box) */
    .evidence-label {
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .evidence-card {
        border-radius: 8px;
        padding: 12px 14px;
        min-height: 110px;
        max-height: 170px;
        overflow-y: auto;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .evidence-card-gold {
        background-color: #FEFCE8 !important;
        border: 1.5px solid #FEF08A !important;
        border-left: 4px solid #EAB308 !important;
    }
    .evidence-card-ir {
        background-color: #F0F9FF !important;
        border: 1.5px solid #BAE6FD !important;
        border-left: 4px solid #0284C7 !important;
    }
    .evidence-card-top2 {
        background-color: #ECFDF5 !important;
        border: 1.5px solid #A7F3D0 !important;
        border-left: 4px solid #10B981 !important;
    }
    .evidence-p {
        font-size: 0.94rem !important;
        line-height: 1.6 !important;
        color: #0F172A !important;
        margin: 0 !important;
        font-weight: 500 !important;
    }

    /* 5. Thẻ Mô hình (Model Cards) */
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

    /* 6. Huy hiệu Trạng thái & Nhãn */
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

    /* 7. Khung Bài Báo Ngữ Cảnh */
    .context-container {
        line-height: 1.85;
        color: #0F172A !important;
        font-size: 0.96rem;
        background: #FFFFFF;
        padding: 18px;
        border-radius: 8px;
        border: 1.5px solid #E2E8F0;
        max-height: 420px;
        overflow-y: auto;
    }
    .legend-bar {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        font-size: 0.9rem;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Đường dẫn file dữ liệu
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data/processed/unified_demo_predictions.csv"
CANDIDATES_PATH = PROJECT_ROOT / "data/processed/retrieval/evidence_candidates.csv"

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

@st.cache_data
def load_candidates_map():
    if not CANDIDATES_PATH.exists():
        return {}
    cands = pd.read_csv(CANDIDATES_PATH)
    return {cid: group[['sentence_id', 'sentence_text']].to_dict('records') 
            for cid, group in cands.groupby('claim_id')}

df_all = load_unified_data()
candidates_map = load_candidates_map()

# ==============================================================================
# HÀM HỖ TRỢ HIGHLIGHT BÀI BÁO NGỮ CẢNH
# ==============================================================================
def check_overlap(s1, s2):
    """Kiểm tra độ trùng khớp giữa câu trong bài báo và bằng chứng."""
    s1 = str(s1).strip().lower()
    s2 = str(s2).strip().lower()
    if not s1 or not s2 or s1 == 'nan' or s2 == 'nan':
        return False
    if s1 in s2 or s2 in s1:
        return True
    if len(s1) > 25 and (s1[:30] in s2 or s1[-30:] in s2):
        return True
    if len(s2) > 25 and (s2[:30] in s1 or s2[-30:] in s1):
        return True
    matcher = difflib.SequenceMatcher(None, s1, s2)
    match = matcher.find_longest_match(0, len(s1), 0, len(s2))
    if match.size > 35:
        return True
    return False

def render_highlighted_context(claim_id, gold_ev, top2_ev, raw_context):
    """Tô màu bằng chứng vàng và bằng chứng truy xuất trong bài báo ngữ cảnh."""
    sents = candidates_map.get(claim_id, [])
    if not sents:
        # Fallback hiển thị văn bản thường
        return f'<p class="evidence-p">{html.escape(str(raw_context))}</p>'
    
    html_parts = []
    for s in sents:
        stxt = s['sentence_text']
        esc = html.escape(stxt)
        is_g = check_overlap(stxt, gold_ev)
        is_r = check_overlap(stxt, top2_ev)
        
        if is_g and is_r:
            html_parts.append(
                f'<span style="background-color: #DCFCE7; color: #14532D; padding: 2px 6px; border-radius: 4px; font-weight: 600; border: 1px solid #86EFAC;">{esc} <span style="font-size: 0.75rem; background: #16A34A; color: white; padding: 1px 6px; border-radius: 4px; vertical-align: middle;">🟢 Cả hai</span></span>'
            )
        elif is_g:
            html_parts.append(
                f'<span style="background-color: #FEF9C3; color: #713F12; padding: 2px 6px; border-radius: 4px; font-weight: 500; border: 1px solid #FDE047;">{esc} <span style="font-size: 0.75rem; background: #CA8A04; color: white; padding: 1px 6px; border-radius: 4px; vertical-align: middle;">🟡 Gold</span></span>'
            )
        elif is_r:
            html_parts.append(
                f'<span style="background-color: #E0F2FE; color: #075985; padding: 2px 6px; border-radius: 4px; font-weight: 500; border: 1px solid #7DD3FC;">{esc} <span style="font-size: 0.75rem; background: #0284C7; color: white; padding: 1px 6px; border-radius: 4px; vertical-align: middle;">🔵 Retrieval</span></span>'
            )
        else:
            html_parts.append(f'<span>{esc}</span>')
            
    return ' '.join(html_parts)

# ==============================================================================
# SIDEBAR: BỘ LỌC & DANH SÁCH CÂU TỪ DATASET
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/balance-scale.png", width=60)
    st.markdown("### 🗂️ Chọn Câu Từ Dataset (723 mẫu)")

    # 1. Lọc theo tình huống
    scenario_filter = st.selectbox(
        "Tình huống nghiên cứu:",
        [
            "Tất cả các câu mẫu",
            "✨ Ca được giải cứu bởi Đề xuất Mở rộng Ngữ cảnh (Top-2)",
            "📉 Ca Sụt giảm thực tế (Baseline ĐÚNG -> Rerank Top-1 SAI)",
            "💡 Ca Ngữ nghĩa thắng Từ khóa (TF-IDF SAI -> BamiBERT ĐÚNG)",
            "💥 Ca Thách thức (Tất cả mô hình đều sai)"
        ],
        index=1,
        help="Lọc các ca tiêu biểu minh họa cho kết quả thực nghiệm của đồ án."
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
    elif "Sụt giảm thực tế" in scenario_filter:
        filtered_df = filtered_df[
            (filtered_df["pred_bamibert_gold"] == filtered_df["gold_label"]) & 
            (filtered_df["pred_bamibert_rerank"] != filtered_df["gold_label"])
        ]
    elif "Ngữ nghĩa thắng Từ khóa" in scenario_filter:
        filtered_df = filtered_df[
            (filtered_df["pred_tfidf_gold"] != filtered_df["gold_label"]) & 
            (filtered_df["pred_bamibert_gold"] == filtered_df["gold_label"])
        ]
    elif "Thách thức" in scenario_filter:
        filtered_df = filtered_df[
            (filtered_df["pred_tfidf_gold"] != filtered_df["gold_label"]) & 
            (filtered_df["pred_phobert_gold"] != filtered_df["gold_label"]) &
            (filtered_df["pred_bamibert_gold"] != filtered_df["gold_label"])
        ]

    st.caption(f"Tìm thấy **{len(filtered_df)}** câu phù hợp.")

    # Tạo danh sách hiển thị
    options = {}
    for idx, row in filtered_df.iterrows():
        short_stmt = (str(row['statement'])[:55] + "...") if len(str(row['statement'])) > 55 else str(row['statement'])
        options[f"[{row['claim_id']}] ({row['gold_label']}) {short_stmt}"] = idx

    # Nút chọn ngẫu nhiên
    if st.button("🎲 Chọn Ngẫu Nhiên 1 Câu", width='stretch'):
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
st.markdown('<div class="main-title">⚖️ ViFactCheck — Khảo Sát & Phân Tích Dự Đoán</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">So sánh kết quả dự đoán giữa <b>Baseline (Bằng chứng Vàng)</b> và <b>Reranking Pipeline (Bằng chứng Truy xuất)</b> trên 723 mẫu thực tế.</div>', unsafe_allow_html=True)

if not selected_key or len(filtered_df) == 0:
    st.info("Không có câu nào thỏa mãn bộ lọc hiện tại. Vui lòng nới lỏng bộ lọc ở thanh bên trái.")
    st.stop()

selected_row_idx = options[selected_key]
sample = df_all.loc[selected_row_idx]

gold_label = sample["gold_label"]
gold_badge_class = "badge-sup" if gold_label == "SUPPORTED" else ("badge-ref" if gold_label == "REFUTED" else "badge-nei")
gold_icon = "🟢" if gold_label == "SUPPORTED" else ("🔴" if gold_label == "REFUTED" else "⚪")

# ==============================================================================
# 1. TUYÊN BỐ CẦN KIỂM CHỨNG (CLAIM) & 3 LOẠI BẰNG CHỨNG (THẺ P)
# ==============================================================================
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-size: 1.15rem; font-weight: 700; color: #1E3A8A;">📝 Dữ liệu Phát biểu & Bằng chứng của Mẫu <code>{sample['claim_id']}</code></span>
        <span>Nhãn Thực Tế (Ground Truth): <span class="{gold_badge_class}">{gold_icon} {gold_label}</span></span>
    </div>
    """,
    unsafe_allow_html=True
)

# Chỉ có Claim là cần dùng Text Area
user_statement = st.text_area(
    "Tuyên bố cần kiểm chứng (Claim / Statement):",
    value=str(sample["statement"]),
    height=75,
    help="Nội dung tuyên bố từ dataset (có thể xem hoặc sao chép)."
)

# 3 Loại Bằng Chứng Sử Dụng Thẻ <p> (Không dùng text box)
col_gold, col_ir, col_top2 = st.columns(3)

with col_gold:
    st.markdown(
        """
        <div class="evidence-label" style="color: #854D0E;">
            🥇 Bằng chứng Vàng (Gold Evidence)
        </div>
        """,
        unsafe_allow_html=True
    )
    gold_text_clean = html.escape(str(sample["gold_evidence"]))
    st.markdown(
        f"""
        <div class="evidence-card evidence-card-gold">
            <p class="evidence-p">{gold_text_clean}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_ir:
    st.markdown(
        """
        <div class="evidence-label" style="color: #075985;">
            🔍 Bằng chứng Truy xuất (IR / Fact Rerank Top-1)
        </div>
        """,
        unsafe_allow_html=True
    )
    rerank_text_clean = html.escape(str(sample["reranked_evidence"]))
    st.markdown(
        f"""
        <div class="evidence-card evidence-card-ir">
            <p class="evidence-p">{rerank_text_clean}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_top2:
    st.markdown(
        """
        <div class="evidence-label" style="color: #065F46;">
            🚀 Bằng chứng Mở rộng Ngữ cảnh (Top-2 Concatenated)
        </div>
        """,
        unsafe_allow_html=True
    )
    top2_text_clean = html.escape(str(sample["top2_evidence"]))
    st.markdown(
        f"""
        <div class="evidence-card evidence-card-top2">
            <p class="evidence-p">{top2_text_clean}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ==============================================================================
# 2. BẢNG SO SÁNH DỰ ĐOÁN 2 NHÓM MÔ HÌNH (SIDE-BY-SIDE)
# ==============================================================================
st.markdown("### 🥊 Bảng So Sánh Dự Đoán Trực Tiếp")

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
            <div style="margin-top: 6px; display: flex; justify-content: space-between; align-items: center;">
                <span>Dự đoán: <span class="{p_badge}">{p_icon} {pred_label}</span></span>
                <span style="font-size: 0.82rem; color: #64748B;">Thật: <b>{true_label}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_base:
    st.markdown("#### 🏛️ Nhóm 1: BASELINE (Bằng chứng Vàng)")
    render_model_result(
        "1. TF-IDF + Logistic Regression",
        sample["pred_tfidf_gold"],
        gold_label,
        "Bằng chứng Vàng (Mô hình Túi từ - BoW)"
    )
    render_model_result(
        "2. PhoBERT (Base)",
        sample["pred_phobert_gold"],
        gold_label,
        "Bằng chứng Vàng (VnCoreNLP Segmented)"
    )
    render_model_result(
        "3. BamiBERT (Exp006 SOTA)",
        sample["pred_bamibert_gold"],
        gold_label,
        "Bằng chứng Vàng (Semantic Prefix Prompt)"
    )
    st.caption("ℹ️ *Nhóm Baseline:* Phản ánh kết quả khi có sẵn bằng chứng sạch do người chọn sẵn.")

with col_rerank:
    st.markdown("#### 🔄 Nhóm 2: RERANKING PIPELINE (Bằng chứng Truy xuất)")
    render_model_result(
        "1. PhoBERT (BM25 + Fact Rerank Top-1)",
        sample["pred_phobert_rerank"],
        gold_label,
        "Câu Top-1 trích xuất tự động từ bài báo"
    )
    render_model_result(
        "2. BamiBERT (BM25 + Fact Rerank Top-1)",
        sample["pred_bamibert_rerank"],
        gold_label,
        "Câu Top-1 trích xuất tự động từ bài báo"
    )
    render_model_result(
        "3. BamiBERT + Top-2 Context Expansion",
        sample["pred_bamibert_top2"],
        gold_label,
        "Ghép 2 câu Fact Rerank (Đề xuất giải quyết đói ngữ cảnh)"
    )
    if "pred_bamibert_hybrid_top2" in sample and pd.notna(sample["pred_bamibert_hybrid_top2"]):
        render_model_result(
            "4. BamiBERT (Hybrid SBERT+BM25 + Top-2)",
            sample["pred_bamibert_hybrid_top2"],
            gold_label,
            "Dense SBERT + BM25 + Fact Rerank Top-2"
        )
    st.caption("ℹ️ *Nhóm Pipeline:* Phản ánh hiệu năng thực tế khi hệ thống tự tìm bằng chứng từ bài báo.")

st.markdown("---")

# ==============================================================================
# 3. BÀI BÁO NGỮ CẢNH GỐC (HIGHLIGHT GOLD & RETRIEVAL EVIDENCE)
# ==============================================================================
st.markdown("### 📰 Bài Báo Ngữ Cảnh Gốc (Context Article)")

# Thanh chú thích màu highlight
st.markdown(
    """
    <div class="legend-bar">
        <span style="font-weight: 700; color: #1E293B;">🔍 Chú thích vị trí bằng chứng:</span>
        <span style="background-color: #FEF9C3; color: #713F12; border: 1px solid #FDE047; padding: 3px 10px; border-radius: 4px; font-weight: 600;">
            🟡 Bằng chứng Vàng (Gold Evidence)
        </span>
        <span style="background-color: #E0F2FE; color: #075985; border: 1px solid #7DD3FC; padding: 3px 10px; border-radius: 4px; font-weight: 600;">
            🔵 Bằng chứng Truy xuất (Retrieval Evidence)
        </span>
        <span style="background-color: #DCFCE7; color: #14532D; border: 1px solid #86EFAC; padding: 3px 10px; border-radius: 4px; font-weight: 600;">
            🟢 Trùng khớp cả hai (Gold & Retrieval)
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

highlighted_article_html = render_highlighted_context(
    sample["claim_id"],
    sample["gold_evidence"],
    sample["top2_evidence"],
    sample["context"]
)

st.markdown(
    f"""
    <div class="context-container">
        {highlighted_article_html}
    </div>
    """,
    unsafe_allow_html=True
)
