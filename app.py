"""Chat-first Streamlit UI for Vietnam Drug Law RAG Assistant.

Entrypoint: streamlit run app.py
Presentation-only refactor: backend RAG remains generate_with_citation().
"""

from __future__ import annotations

from pathlib import Path
import base64
import html
import json
import re
from typing import Any

import streamlit as st

from src.task10_generation import generate_with_citation
from src.task4_chunking_indexing import chunk_documents, load_documents

st.set_page_config(page_title="Vietnam Drug Law RAG", page_icon="§", layout="wide")

PRIMARY = "#7C3AED"
SUCCESS = "#10B981"


def _esc(value: Any) -> str:
    return html.escape(str(value or ""))


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def icon_svg(name: str, size: int = 20) -> str:
    common = f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"'
    icons = {
        "scale": f'<svg {common}><path d="M12 3v18"/><path d="M5 7h14"/><path d="M6 7l-3 6h6L6 7Z"/><path d="M18 7l-3 6h6l-3-6Z"/><path d="M8 21h8"/></svg>',
        "chat": f'<svg {common}><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/></svg>',
        "doc": f'<svg {common}><path d="M7 3h7l4 4v14H7z"/><path d="M14 3v5h5"/><path d="M9 13h6"/><path d="M9 17h5"/></svg>',
        "news": f'<svg {common}><path d="M4 5h14a2 2 0 0 1 2 2v12H6a2 2 0 0 1-2-2z"/><path d="M8 9h7"/><path d="M8 13h8"/><path d="M8 17h5"/></svg>',
        "chart": f'<svg {common}><path d="M4 19V5"/><path d="M4 19h16"/><path d="M8 16v-5"/><path d="M12 16V8"/><path d="M16 16v-8"/></svg>',
        "search": f'<svg {common}><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg>',
        "guide": f'<svg {common}><path d="M12 3a6 6 0 0 0-4 10.47V17h8v-3.53A6 6 0 0 0 12 3Z"/><path d="M9 21h6"/><path d="M10 17h4"/></svg>',
        "send": f'<svg {common}><path d="m22 2-7 20-4-9-9-4 20-7Z"/><path d="M22 2 11 13"/></svg>',
        "user": f'<svg {common}><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></svg>',
        "ai": f'<svg {common}><rect x="5" y="7" width="14" height="11" rx="3"/><path d="M9 7V4"/><path d="M15 7V4"/><circle cx="10" cy="12" r="1"/><circle cx="14" cy="12" r="1"/><path d="M10 16h4"/></svg>',
        "check": f'<svg {common}><path d="M20 6 9 17l-5-5"/></svg>',
        "shield": f'<svg {common}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-5"/></svg>',
        "refresh": f'<svg {common}><path d="M20 12a8 8 0 1 1-2.34-5.66"/><path d="M20 4v6h-6"/></svg>',
    }
    return icons.get(name, icons["doc"])


def inject_css(dark: bool = False) -> None:
    t = {
        "bg": "#0F1024" if dark else "#FAF8FF",
        "panel": "#17162E" if dark else "#FFFFFF",
        "panel2": "#201B3D" if dark else "#F8FAFC",
        "card": "#1D1B36" if dark else "#FFFFFF",
        "text": "#F8FAFC" if dark else "#111827",
        "muted": "#B8C0D4" if dark else "#64748B",
        "border": "rgba(196,181,253,.28)" if dark else "#E9D5FF",
        "soft": "rgba(124,58,237,.18)" if dark else "#F3E8FF",
        "shadow": "0 20px 50px rgba(0,0,0,.28)" if dark else "0 20px 50px rgba(88,28,135,.08)",
        "input": "#111827" if dark else "#FFFFFF",
    }
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
        :root {{ --primary:{PRIMARY}; --success:{SUCCESS}; --bg:{t['bg']}; --panel:{t['panel']}; --panel2:{t['panel2']}; --card:{t['card']}; --text:{t['text']}; --muted:{t['muted']}; --border:{t['border']}; --soft:{t['soft']}; }}
        html, body, [class*="css"] {{ font-family:'Manrope', sans-serif; }}
        .stApp {{ background: radial-gradient(circle at 76% 6%, rgba(124,58,237,.10), transparent 25%), var(--bg); color:var(--text); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stToolbar"] {{ opacity:.55; }}
        .block-container {{ max-width: 1500px; padding: 1rem 1.1rem 1.8rem; }}
        [data-testid="stSidebar"] {{ background: var(--panel); border-right:1px solid var(--border); box-shadow:8px 0 28px rgba(15,23,42,.04); min-width:250px; max-width:250px; }}
        [data-testid="stSidebar"] * {{ color:var(--text); }}
        h1,h2,h3,h4,p,span,div,label {{ color:var(--text); }}
        .app-title {{ font-size:28px; font-weight:850; letter-spacing:-.04em; margin:0; }}
        .muted {{ color:var(--muted); }}
        .brand {{ display:flex; align-items:center; gap:12px; padding:10px 0 18px; }}
        .brand-mark, .bubble {{ display:flex; align-items:center; justify-content:center; color:var(--primary); background:var(--soft); border-radius:14px; }}
        .brand-mark {{ width:44px; height:44px; color:white; background:linear-gradient(135deg,#A78BFA,#7C3AED); box-shadow:0 12px 24px rgba(124,58,237,.22); }}
        .brand-name {{ font-size:21px; font-weight:850; line-height:1; }}
        .brand-sub {{ color:var(--primary); font-size:12px; font-weight:850; margin-top:4px; }}
        div[data-testid="stRadio"] label {{ padding:9px 12px; border-radius:14px; transition:background .18s ease, color .18s ease; }}
        div[data-testid="stRadio"] label:hover {{ background:var(--soft); cursor:pointer; }}
        .notice {{ margin-top:22px; padding:16px; border-radius:18px; background:linear-gradient(150deg,var(--soft),var(--panel)); border:1px solid var(--border); }}
        .notice-title {{ display:flex; align-items:center; gap:9px; font-weight:850; color:var(--primary); margin-bottom:8px; }}
        .notice-copy {{ color:var(--muted); font-size:12.5px; line-height:1.65; }}
        .chat-header {{ display:flex; justify-content:space-between; gap:20px; align-items:center; padding:22px 24px; background:var(--panel); border:1px solid var(--border); border-radius:24px; box-shadow:{t['shadow']}; margin-bottom:18px; }}
        .chat-header-copy {{ color:var(--muted); line-height:1.58; margin-top:8px; max-width:760px; font-size:15px; }}
        .header-actions {{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
        .status-pill {{ padding:8px 12px; border-radius:999px; background:var(--soft); color:var(--primary); font-weight:800; font-size:12px; white-space:nowrap; }}
        .question-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:16px 0 18px; }}
        div.stButton > button {{ min-height:64px; border-radius:16px; border:1px solid var(--border); background:var(--panel); color:var(--text); font-weight:700; font-size:13.5px; text-align:left; white-space:normal; box-shadow:0 10px 24px rgba(88,28,135,.055); transition:border-color .18s ease, background .18s ease, color .18s ease; }}
        div.stButton > button:hover {{ border-color:var(--primary); background:var(--soft); color:var(--primary); }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--panel); border:1px solid var(--border); border-radius:24px; box-shadow:{t['shadow']}; padding:10px; }}
        div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {{ gap:.25rem; }}
        .chat-empty {{ padding:34px 16px; text-align:center; color:var(--muted); }}
        .msg {{ display:flex; gap:12px; margin:14px 0; align-items:flex-start; }}
        .msg.user {{ justify-content:flex-end; }}
        .msg.user .msg-content {{ background:linear-gradient(135deg,#7C3AED,#9F67FF); color:white; border:0; max-width:min(76%, 760px); overflow-wrap:anywhere; }}
        .msg.assistant .msg-content {{ background:var(--card); border:1px solid var(--border); max-width:min(86%, 860px); overflow-wrap:anywhere; }}
        .msg-content {{ border-radius:20px; padding:15px 17px; box-shadow:0 12px 28px rgba(88,28,135,.06); }}
        .msg-content, .msg-content * {{ color:inherit; }}
        .avatar {{ width:38px; height:38px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:0 0 auto; color:white; background:linear-gradient(135deg,#A78BFA,#7C3AED); }}
        .avatar svg {{ width:19px; height:19px; }}
        .msg-meta {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; color:var(--muted); font-size:12px; font-weight:700; }}
        .answer-badge {{ display:inline-flex; align-items:center; gap:5px; padding:4px 9px; border-radius:999px; background:rgba(16,185,129,.14); color:var(--success); font-weight:850; font-size:12px; }}
        .answer-text {{ font-size:16px; line-height:1.82; color:var(--text); letter-spacing:-.01em; font-weight:500; }}
        .term-chip {{ display:inline; padding:1px 4px 2px; border-radius:7px; background:{'rgba(167,139,250,.18)' if dark else 'rgba(124,58,237,.09)'}; color:{'#EDE9FE' if dark else '#5B21B6'}; font-weight:750; box-decoration-break:clone; -webkit-box-decoration-break:clone; }}
        .confidence {{ margin-top:12px; display:flex; align-items:center; gap:8px; color:var(--muted); font-size:13px; }}
        .green-dot {{ width:8px; height:8px; border-radius:50%; background:var(--success); }}
        div[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:16px; background:var(--panel); margin:8px 0 16px 50px; box-shadow:none; max-width:calc(100% - 60px); }}
        div[data-testid="stExpander"] details summary {{ color:var(--muted); font-weight:800; font-size:13px; }}
        .source-card {{ background:var(--panel2); border:1px solid var(--border); border-radius:16px; padding:12px; min-height:96px; }}
        .source-actions {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
        .source-link {{ display:inline-flex; align-items:center; justify-content:center; border-radius:999px; padding:7px 10px; background:var(--soft); color:var(--primary) !important; font-size:12px; font-weight:850; text-decoration:none !important; }}
        .source-card:hover {{ border-color:#C4B5FD; }}
        .doc-icon {{ width:32px; height:32px; border-radius:11px; background:var(--soft); color:var(--primary); display:flex; align-items:center; justify-content:center; font-weight:850; margin-bottom:8px; }}
        .doc-title {{ font-size:13px; font-weight:850; line-height:1.35; color:var(--text); margin-bottom:5px; }}
        .doc-meta {{ font-size:12px; line-height:1.4; color:var(--muted); }}
        .input-shell {{ margin-top:14px; padding:10px; border-radius:22px; background:var(--panel); border:1.5px solid rgba(124,58,237,.46); box-shadow:0 12px 32px rgba(124,58,237,.10); }}
        .input-shell input {{ background:{t['input']} !important; color:var(--text) !important; border:0 !important; font-size:15px !important; }}
        .input-shell button {{ min-height:46px !important; border-radius:999px !important; background:linear-gradient(135deg,#8B5CF6,#7C3AED) !important; color:white !important; text-align:center !important; }}
        .right-card {{ background:var(--panel); border:1px solid var(--border); border-radius:22px; padding:16px; margin-bottom:14px; box-shadow:{t['shadow']}; }}
        .right-title {{ color:var(--primary); font-weight:850; font-size:16px; margin-bottom:13px; }}
        .metric-row {{ display:flex; justify-content:space-between; align-items:center; gap:12px; padding:10px 11px; border-radius:14px; background:var(--panel2); border:1px solid var(--border); margin-bottom:8px; }}
        .metric-left {{ display:flex; align-items:center; gap:9px; font-size:13px; font-weight:800; }}
        .small-icon {{ width:30px; height:30px; border-radius:11px; display:flex; align-items:center; justify-content:center; background:var(--soft); color:var(--primary); }}
        .metric-value {{ font-size:20px; font-weight:900; text-align:right; }}
        .metric-caption {{ font-size:11px; color:var(--muted); text-align:right; }}
        .chip-wrap {{ display:flex; flex-wrap:wrap; gap:8px; }}
        .keyword-chip {{ padding:7px 11px; border-radius:999px; background:var(--soft); color:{'#EDE9FE' if dark else '#4C1D95'}; font-size:12px; font-weight:800; }}
        .history-item {{ padding:10px 11px; border-radius:13px; background:var(--panel2); margin-bottom:8px; font-size:12px; display:flex; justify-content:space-between; gap:8px; }}

        .explorer-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:16px; }}
        .explorer-card {{ position:relative; overflow:hidden; background:var(--panel); border:1px solid var(--border); border-radius:24px; padding:18px; box-shadow:0 18px 42px rgba(88,28,135,.07); }}
        .explorer-card::before {{ content:""; position:absolute; inset:0 0 auto 0; height:4px; background:linear-gradient(90deg,#7C3AED,#C4B5FD); }}
        .explorer-top {{ display:flex; justify-content:space-between; gap:14px; align-items:flex-start; margin-bottom:12px; }}
        .explorer-kicker {{ display:inline-flex; align-items:center; gap:7px; padding:6px 10px; border-radius:999px; background:var(--soft); color:var(--primary); font-size:12px; font-weight:850; }}
        .explorer-title {{ margin:10px 0 8px; color:var(--text); font-size:22px; line-height:1.22; letter-spacing:-.025em; font-weight:900; }}
        .explorer-summary {{ color:var(--text); font-size:14.5px; line-height:1.72; margin:0 0 14px; }}
        .explorer-meta {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
        .meta-chip {{ display:inline-flex; align-items:center; gap:6px; padding:6px 9px; border-radius:999px; background:var(--panel2); border:1px solid var(--border); color:var(--muted); font-size:12px; font-weight:750; }}
        .article-card {{ background:var(--panel); border:1px solid var(--border); border-radius:22px; padding:16px; margin:0 0 14px; box-shadow:0 14px 36px rgba(88,28,135,.065); }}
        .article-title {{ color:var(--text); font-size:20px; line-height:1.3; letter-spacing:-.02em; font-weight:900; margin:9px 0 8px; }}
        .article-lede {{ color:var(--text); line-height:1.68; font-size:14.5px; margin:10px 0 12px; }}
        .article-actions {{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin-top:12px; }}
        .external-link {{ color:var(--primary); font-weight:850; text-decoration:none; }}
        .duplicate-note {{ color:var(--muted); background:var(--panel2); border:1px dashed var(--border); border-radius:14px; padding:10px 12px; font-size:13px; }}
        @media (max-width: 900px) {{ .explorer-grid {{ grid-template-columns:1fr; }} }}
        .history-time {{ color:var(--muted); white-space:nowrap; }}
        @media (max-width: 1200px) {{ .question-row,.source-grid {{ grid-template-columns:1fr 1fr; }} }}
        @media (max-width: 780px) {{ .chat-header {{ flex-direction:column; align-items:flex-start; }} .question-row,.source-grid {{ grid-template-columns:1fr; }} .msg.user .msg-content,.msg.assistant .msg-content {{ max-width:100%; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Sidebar ---------------------------------------------------------------------
def render_sidebar() -> tuple[str, bool]:
    st.sidebar.markdown(
        f"""
        <div class="brand">
            <div class="brand-mark">{icon_svg('scale', 22)}</div>
            <div><div class="brand-name">Legal RAG</div><div class="brand-sub">Drug Law Assistant</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "",
        ["Chat", "Văn bản pháp luật", "Tin tức", "Đánh giá", "Từ khóa nổi bật", "Hướng dẫn sử dụng"],
        label_visibility="collapsed",
    )
    dark = st.sidebar.toggle("Chế độ tối", value=st.session_state.get("dark_mode", False))
    st.session_state.dark_mode = dark
    st.sidebar.markdown(
        f"""
        <div class="notice">
            <div class="notice-title"><span class="small-icon">{icon_svg('shield', 15)}</span>Lưu ý quan trọng</div>
            <div class="notice-copy">Kết quả chỉ mang tính tham khảo. Vui lòng đối chiếu văn bản gốc hoặc liên hệ cơ quan chức năng.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return page, dark


# Main chat UI ----------------------------------------------------------------
def render_hero() -> None:
    st.markdown(
        f"""
        <div class="chat-header">
            <div>
                <h1 class="app-title">Trợ lý pháp luật về ma túy Việt Nam</h1>
                <div class="chat-header-copy">Hỏi đáp theo phong cách chatbot, truy xuất từ văn bản pháp luật và tin tức đã chuẩn hóa. Mỗi câu trả lời đi kèm nguồn tham khảo để bạn đối chiếu nhanh.</div>
            </div>
            <div class="header-actions">
                <span class="status-pill">RAG pipeline</span>
                <span class="status-pill">Citation-ready</span>
                <span class="status-pill">Legal-tech</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggested_questions() -> None:
    st.markdown("<div style='font-weight:850;margin:6px 0 10px'>Gợi ý câu hỏi</div>", unsafe_allow_html=True)
    questions = [
        "Chiêu thức giấu 91kg ma túy từ Campuchia liên quan hành vi gì?",
        "Vụ Công Trí liên quan điều luật nào?",
        "Đường dây vận chuyển 117kg ma túy của Tuấn cọp bị xử lý ra sao?",
        "Thời hạn cai nghiện ma túy theo luật mới nhất được nói ở nguồn nào?",
    ]
    cols = st.columns(4)
    for idx, (col, question) in enumerate(zip(cols, questions)):
        with col:
            if st.button(question, key=f"suggest_{idx}", use_container_width=True):
                st.session_state.pending_question = question


def confidence(score: float, count: int) -> tuple[str, float]:
    if score >= 0.07 and count >= 2:
        return "Cao", 0.92
    if score >= 0.03 or count >= 1:
        return "Trung bình", 0.74
    return "Thấp", 0.48


def _highlight_terms(answer: str) -> str:
    safe = _esc(answer)
    # Keep highlights subtle and sparse so the answer reads like normal prose.
    terms = [
        "B? lu?t H?nh s? 2015",
        "Lu?t Ph?ng, ch?ng ma t?y 2021",
        "Ngh? ??nh 57/2022",
        "Ngh? ??nh 105/2021",
        "Ketamine",
    ]
    for term in terms:
        safe = re.sub(
            re.escape(term),
            f"<span class='term-chip'>{_esc(term)}</span>",
            safe,
            count=1,
            flags=re.IGNORECASE,
        )
    return safe.replace("\n", "<br>")

def _news_url_for_source(source: str) -> str:
    stem = Path(source or "").stem
    path = Path("data/landing/news") / f"{stem}.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload.get("url") or "")
    except Exception:
        return ""

def _source_target(md: dict[str, Any]) -> tuple[str, Path | None]:
    source = str(md.get("source") or "")
    if md.get("type") == "news":
        return _news_url_for_source(source), None
    return "", _find_legal_pdf(source)

def _default_source_cards() -> list[dict]:
    return [
        {"metadata": {"title": "Chiêu thức giấu 91kg ma túy", "source": "Tuổi Trẻ, 2025", "chunk_index": "article_003", "type": "news"}, "score": 0.92},
        {"metadata": {"title": "Bộ luật Hình sự 2015", "source": "Các tội phạm về ma túy", "chunk_index": "Điều 248-251", "type": "legal"}, "score": 0.88},
        {"metadata": {"title": "Luật Phòng, chống ma túy 2021", "source": "Quản lý và phòng chống ma túy", "chunk_index": "Luật 2021", "type": "legal"}, "score": 0.82},
    ]


def render_source_cards(sources: list[dict] | None) -> None:
    """Show compact, per-answer citations with direct source actions."""
    cards = (sources or [])[:3] or _default_source_cards()
    with st.expander(f"Nguồn tham khảo ({len(cards)}) - bấm để xem chi tiết", expanded=False):
        st.caption("Các nguồn này thuộc riêng câu trả lời ngay phía trên. Bấm mở bài gốc hoặc tải PDF để đối chiếu.")
        cols = st.columns(len(cards))
        for idx, (col, src) in enumerate(zip(cols, cards), 1):
            md = src.get("metadata", {})
            title = md.get("title") or md.get("source") or _default_source_cards()[idx - 1]["metadata"]["title"]
            source = md.get("source", "Nguồn pháp luật")
            chunk = md.get("chunk_index", "Trích đoạn")
            is_news = md.get("type") == "news"
            kind = "NEWS" if is_news else "LAW"
            source_url, pdf_path = _source_target(md)
            with col:
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="doc-icon">{kind}</div>
                        <div class="doc-title">{_esc(_clip(title, 78))}</div>
                        <div class="doc-meta">{_esc(_clip(source, 70))}</div>
                        <div style="margin-top:9px"><span class="term-chip">Chunk { _esc(chunk) }</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if source_url:
                    st.markdown(f"<a class='source-link' href='{_esc(source_url)}' target='_blank'>Mở bài gốc</a>", unsafe_allow_html=True)
                elif pdf_path:
                    st.download_button(
                        "Tải / mở PDF gốc",
                        data=pdf_path.read_bytes(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        key=f"source_pdf_{idx}_{Path(str(source)).stem}",
                        use_container_width=True,
                    )
                else:
                    st.caption("Nguồn nội bộ trong corpus RAG")

def render_chat_message(role: str, content: str, sources: list[dict] | None = None, score: float = 0.0) -> None:
    if role == "user":
        st.markdown(
            f"""
            <div class="msg user">
                <div class="msg-content"><div style="font-weight:800;margin-bottom:4px">Bạn</div>{_esc(content)}</div>
                <div class="avatar">{icon_svg('user', 18)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    label, value = confidence(score, len(sources or []))
    st.markdown(
        f"""
        <div class="msg assistant">
            <div class="avatar">{icon_svg('ai', 18)}</div>
            <div class="msg-content">
                <div class="msg-meta"><span class="answer-badge">{icon_svg('check', 13)} Trả lời</span><span>10:30 AM</span></div>
                <div class="answer-text">{_highlight_terms(content)}</div>
                <div class="confidence"><span class="green-dot"></span> Độ tin cậy: {label} ({value:.2f})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_source_cards(sources)


# Right panel -----------------------------------------------------------------
def render_right_panel() -> None:
    docs = load_documents()
    chunks = chunk_documents(docs) if docs else []
    legal_count = len([d for d in docs if d.get("metadata", {}).get("type") == "legal"])
    news_count = len([d for d in docs if d.get("metadata", {}).get("type") == "news"])
    latest_crawl = "2026-06-08"
    try:
        news_json = sorted(Path("data/landing/news").glob("article_*.json"))
        if news_json:
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in news_json]
            news_count = len(payloads)
            crawl_dates = [str(item.get("crawl_date", "")) for item in payloads if isinstance(item, dict) and item.get("crawl_date")]
            latest_crawl = max(crawl_dates) if crawl_dates else latest_crawl
    except Exception:
        pass
    metrics = [
        ("doc", "Văn bản pháp luật", legal_count or 5, "văn bản"),
        ("news", "Tin tức", news_count or 20, "bài báo"),
        ("chart", "Tổng chunks", f"{len(chunks):,}", "đoạn văn bản"),
        ("refresh", "Cập nhật dữ liệu", latest_crawl, "08:01 AM"),
    ]
    rows = "".join(
        f"<div class='metric-row'><div class='metric-left'><span class='small-icon'>{icon_svg(icon, 15)}</span>{label}</div><div><div class='metric-value'>{value}</div><div class='metric-caption'>{caption}</div></div></div>"
        for icon, label, value, caption in metrics
    )
    st.markdown(f"<div class='right-card'><div class='right-title'>Nguồn dữ liệu</div>{rows}</div>", unsafe_allow_html=True)
    chips = ["Ketamine", "Ma túy", "Vận chuyển", "Tàng trữ", "Mua bán", "Cần sa", "MDMA", "Cai nghiện", "Xử phạt", "Tiền chất"]
    st.markdown("<div class='right-card'><div class='right-title'>Từ khóa nổi bật</div><div class='chip-wrap'>" + "".join(f"<span class='keyword-chip'>{_esc(c)}</span>" for c in chips) + "</div></div>", unsafe_allow_html=True)
    user_turns = [m.get("content", "") for m in st.session_state.get("messages", []) if m.get("role") == "user"][-5:][::-1]
    history = [(q, "vừa hỏi") for q in user_turns] or [("Chưa có câu hỏi nào trong phiên này", "")]
    st.markdown("<div class='right-card'><div class='right-title'>Lịch sử trò chuyện</div>" + "".join(f"<div class='history-item'><span>{_esc(_clip(q, 54))}</span><span class='history-time'>{_esc(t)}</span></div>" for q, t in history) + "</div>", unsafe_allow_html=True)


# State and pages -------------------------------------------------------------
def _ensure_state() -> None:
    st.session_state.setdefault("messages", [])


def _needs_context(question: str) -> bool:
    q = question.lower().strip()
    markers = ["đó", "này", "vậy", "trên", "hành vi đó", "vụ này", "người đó", "đối tượng đó", "thế nào", "ra sao"]
    return len(q.split()) <= 8 or any(marker in q for marker in markers)


def _build_rag_query(question: str) -> str:
    previous_user_turns = [m["content"] for m in st.session_state.messages if m.get("role") == "user"][-3:]
    if previous_user_turns and _needs_context(question):
        context = " | ".join(previous_user_turns)
        return (
            f"Câu hỏi cần trả lời: {question}\n"
            f"Ngữ cảnh hội thoại trước để hiểu đại từ/chủ thể: {context}\n"
            "Chỉ trả lời câu hỏi cần trả lời, không trả lời lại các câu hỏi cũ."
        )
    return question


def _submit_question(question: str) -> None:
    rag_query = _build_rag_query(question)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Đang tìm kiếm tài liệu, rerank và tạo câu trả lời..."):
        result = generate_with_citation(rag_query, top_k=5)
    st.session_state.messages.append({"role": "assistant", "content": result.get("answer", "Không tìm thấy đủ thông tin."), "sources": result.get("sources", [])})


def render_chat_area() -> None:
    _ensure_state()
    st.caption("Khung hội thoại lưu lịch sử trong phiên làm việc hiện tại. Kéo thanh cuộn để xem lại các câu trước.")

    # Use Streamlit's native scrollable container; HTML wrappers cannot reliably
    # contain Streamlit components and can make messages visually leak outside.
    with st.container(height=560, border=True):
        if not st.session_state.messages:
            st.markdown(
                """
                <div class='chat-empty'>
                    <div class='answer-badge'>Sẵn sàng tra cứu</div>
                    <div style='font-weight:900;font-size:22px;margin:12px 0 6px'>Bắt đầu cuộc trò chuyện pháp lý</div>
                    <div>Đặt câu hỏi về luật, nghị định hoặc tin tức ma túy. Các câu trả lời sau sẽ được lưu lại trong khung chat này.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.messages:
                sources = msg.get("sources", [])
                score = float(sources[0].get("score", 0.0)) if sources else 0.0
                render_chat_message(msg["role"], msg["content"], sources, score)

    st.markdown("<div class='input-shell'>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        input_col, send_col = st.columns([8, 1])
        question = input_col.text_input("", placeholder="Nhập câu hỏi về luật ma túy hoặc tin tức...", label_visibility="collapsed")
        sent = send_col.form_submit_button("Gửi", use_container_width=True)
        st.caption("Nhấn Enter để gửi - Shift + Enter để xuống dòng")
    st.markdown("</div>", unsafe_allow_html=True)
    if sent and question.strip():
        _submit_question(question.strip())
        st.rerun()

def chat_page() -> None:
    center, right = st.columns([3.35, 1], gap="large")
    with center:
        render_hero()
        render_suggested_questions()
        pending = st.session_state.pop("pending_question", None)
        if pending:
            _submit_question(pending)
        render_chat_area()
    with right:
        render_right_panel()


def _clean_markdown_text(text: str) -> str:
    """Create a compact preview by removing repeated headings/paragraphs from source files."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#+\s*", "", line).strip()
        key = re.sub(r"\W+", " ", line.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(line)
    return "\n\n".join(cleaned)

def _text_similarity(a: str, b: str) -> float:
    a_words = set(re.findall(r"\w+", (a or "").lower()))
    b_words = set(re.findall(r"\w+", (b or "").lower()))
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(len(a_words | b_words), 1)

def _render_page_header(title: str, copy: str, badge: str) -> None:
    st.markdown(
        f"""
        <div class='chat-header'>
            <div>
                <div class='explorer-kicker'>{_esc(badge)}</div>
                <h1 class='app-title'>{_esc(title)}</h1>
                <div class='chat-header-copy'>{_esc(copy)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _find_legal_pdf(source_name: str) -> Path | None:
    stem = Path(source_name or "").stem
    legal_dir = Path("data/landing/legal")
    candidates = [legal_dir / f"{stem}.pdf", legal_dir / f"{stem}.pdf.pdf"]
    candidates.extend(sorted(legal_dir.glob(f"{stem}*.pdf*")))
    return next((path for path in candidates if path.exists()), None)

def _render_pdf_tools(pdf_path: Path | None, title: str, key_prefix: str) -> None:
    if not pdf_path:
        st.warning("Chưa tìm thấy file PDF gốc tương ứng trong data/landing/legal.")
        return

    pdf_bytes = pdf_path.read_bytes()
    left, right = st.columns([1, 1])
    with left:
        st.download_button(
            "Tải PDF gốc",
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
            key=f"download_{key_prefix}",
            use_container_width=True,
        )
    with right:
        st.caption(f"File: {pdf_path.name}")

    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    st.markdown(
        f"""
        <iframe
            src="data:application/pdf;base64,{encoded}#toolbar=1&navpanes=0"
            width="100%"
            height="640"
            style="border:1px solid var(--border); border-radius:18px; background:white;"
            title="{_esc(title)}">
        </iframe>
        """,
        unsafe_allow_html=True,
    )

def _render_legal_page() -> None:
    _render_page_header(
        "Văn bản pháp luật",
        "Các văn bản nền tảng dùng để đối chiếu câu trả lời. Nội dung được rút gọn, khử lặp và có bản PDF gốc để xem trực tiếp.",
        "Legal corpus",
    )
    docs = [d for d in load_documents() if d.get("metadata", {}).get("type") == "legal"]
    if not docs:
        st.info("Chưa tìm thấy văn bản pháp luật trong data/standardized/legal.")
        return

    for idx, doc in enumerate(docs, 1):
        md = doc.get("metadata", {})
        pdf_path = _find_legal_pdf(md.get("source", ""))
        cleaned = _clean_markdown_text(doc.get("content", ""))
        parts = [part for part in cleaned.split("\n\n") if part.strip()]
        card_title = md.get("title") or (parts[0] if parts else md.get("source", "Văn bản pháp luật"))
        summary = next((part for part in parts if part != card_title), cleaned)
        pdf_label = "Có PDF gốc" if pdf_path else "Chưa có PDF"
        st.markdown(
            f"""
            <div class='explorer-card'>
                <div class='explorer-top'>
                    <span class='explorer-kicker'>{icon_svg('doc', 14)} Văn bản #{idx}</span>
                    <span class='meta-chip'>{_esc(md.get('source', 'local markdown'))}</span>
                </div>
                <div class='explorer-title'>{_esc(card_title)}</div>
                <p class='explorer-summary'>{_esc(_clip(summary, 360))}</p>
                <div class='explorer-meta'>
                    <span class='meta-chip'>Loại: pháp luật</span>
                    <span class='meta-chip'>Đã khử lặp nội dung</span>
                    <span class='meta-chip'>{len(parts)} đoạn duy nhất</span>
                    <span class='meta-chip'>{_esc(pdf_label)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Xem PDF gốc - {card_title}", expanded=False):
            _render_pdf_tools(pdf_path, card_title, f"legal_{idx}")
        with st.expander(f"Xem nội dung đã làm sạch - {card_title}", expanded=False):
            st.markdown(cleaned)

def _load_news_articles() -> list[dict[str, Any]]:
    articles: list[dict[str, Any]] = []
    for path in sorted(Path("data/landing/news").glob("article_*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(item, dict):
                item["_file"] = path.name
                articles.append(item)
        except Exception:
            continue
    return articles

def _render_news_page() -> None:
    articles = _load_news_articles()
    _render_page_header(
        "Tin tức",
        f"{len(articles)} bài viết nguồn đã crawl. Giao diện chỉ hiển thị nội dung bài viết khi khác phần tóm tắt để tránh lặp.",
        "News corpus",
    )
    if not articles:
        st.info("Chưa tìm thấy article_*.json trong data/landing/news.")
        return

    for idx, article in enumerate(articles, 1):
        article_title = str(article.get("title") or f"Bài viết {idx}")
        summary = str(article.get("summary") or "")
        content = str(article.get("content") or "")
        same = summary.strip() == content.strip() or _text_similarity(summary, content) > 0.92
        publisher = article.get("publisher") or "Nguồn tin"
        date = article.get("published_date") or "Không rõ ngày"
        topic = article.get("topic") or "ma túy"
        url = article.get("url") or ""
        duplicate_note = "<div class='duplicate-note'>Nội dung bài viết trong dữ liệu hiện tại trùng hoặc gần trùng với phần tóm tắt, nên app chỉ hiển thị một lần để tránh rối mắt.</div>" if same else ""
        source_link = f"<a class='external-link' href='{_esc(url)}' target='_blank'>Mở bài gốc</a>" if url else ""
        st.markdown(
            f"""
            <div class='article-card'>
                <div class='explorer-top'>
                    <span class='explorer-kicker'>{icon_svg('news', 14)} Bài viết #{idx}</span>
                    <span class='meta-chip'>{_esc(article.get('_file', 'article.json'))}</span>
                </div>
                <div class='article-title'>{_esc(article_title)}</div>
                <div class='explorer-meta'>
                    <span class='meta-chip'>{_esc(publisher)}</span>
                    <span class='meta-chip'>{_esc(date)}</span>
                    <span class='meta-chip'>{_esc(topic)}</span>
                </div>
                <p class='article-lede'>{_esc(_clip(summary or content, 430))}</p>
                {duplicate_note}
                <div class='article-actions'>
                    <span class='meta-chip'>Nguồn RAG: news</span>
                    {source_link}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if content and not same:
            with st.expander(f"Xem nội dung đầy đủ - {article_title}", expanded=False):
                st.write(content)

def explorer_page(doc_type: str, title: str) -> None:
    if doc_type == "legal":
        _render_legal_page()
    elif doc_type == "news":
        _render_news_page()
    else:
        _render_page_header(title, "Duyệt nguồn dữ liệu đã chuẩn hóa dùng cho RAG.", "Corpus")

def evaluation_page() -> None:
    st.markdown("<div class='chat-header'><div><h1 class='app-title'>Đánh giá RAG</h1><div class='chat-header-copy'>Theo dõi faithfulness, relevance, context recall và precision.</div></div></div>", unsafe_allow_html=True)
    path = Path("group_project/evaluation/results.md")
    st.markdown(path.read_text(encoding="utf-8") if path.exists() else "Chưa có results.md. Chạy evaluation pipeline trước.")
    dataset = Path("group_project/evaluation/golden_dataset.json")
    if dataset.exists():
        st.metric("Golden Q&A pairs", len(json.loads(dataset.read_text(encoding="utf-8"))))


def keywords_page() -> None:
    st.markdown("<div class='chat-header'><div><h1 class='app-title'>Từ khóa nổi bật</h1><div class='chat-header-copy'>Các chủ đề xuất hiện nhiều trong legal/news corpus.</div></div></div>", unsafe_allow_html=True)
    docs = load_documents()
    text = " ".join(d["content"].lower() for d in docs)
    keywords = ["ketamine", "ma túy", "vận chuyển", "tàng trữ", "mua bán", "cần sa", "mdma", "cai nghiện", "xử phạt", "tiền chất"]
    st.bar_chart({"Số lần xuất hiện": {kw: text.count(kw) for kw in keywords}})


def guide_page() -> None:
    st.markdown("""
    <div class="chat-header"><div><h1 class="app-title">Hướng dẫn sử dụng</h1><div class="chat-header-copy">Cách đặt câu hỏi để nhận câu trả lời có nguồn tham khảo rõ ràng.</div></div></div>
    <div class="right-card">
    <b>Gợi ý sử dụng</b><br><br>
    1. Hỏi rõ hành vi hoặc chất cần tra cứu.<br>
    2. Khi hỏi vụ việc thực tế, nêu tên vụ việc hoặc đối tượng trong tin tức.<br>
    3. Luôn mở phần nguồn tham khảo để đối chiếu văn bản gốc.<br>
    4. Kết quả chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp.
    </div>
    """, unsafe_allow_html=True)


def main() -> None:
    page, dark = render_sidebar()
    inject_css(dark)
    if page == "Chat":
        chat_page()
    elif page == "Văn bản pháp luật":
        explorer_page("legal", "Văn bản pháp luật")
    elif page == "Tin tức":
        explorer_page("news", "Tin tức")
    elif page == "Đánh giá":
        evaluation_page()
    elif page == "Từ khóa nổi bật":
        keywords_page()
    else:
        guide_page()


if __name__ == "__main__":
    main()
