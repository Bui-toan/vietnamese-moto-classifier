"""
Theme và CSS tùy chỉnh cho ứng dụng Streamlit (app/ui/theme.py)
Tuân thủ phong cách giao diện mục 7.1 của kế hoạch v4:
- Nền tối xanh navy (#0B1220, #131C2E).
- Màu nhấn xanh ngọc teal (#14B8A6), cam cảnh báo (#F59E0B).
- Thẻ bo góc đổ bóng nhẹ, tiêu đề gradient.
"""

CUSTOM_CSS = """
<style>
/* Font và tiêu đề gradient */
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #14B8A6 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #94A3B8;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

/* Thẻ kết quả card bo góc */
.result-card {
    background-color: #131C2E;
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid #1E293B;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    margin-bottom: 1rem;
}

/* Badge trạng thái */
.badge-pass {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid #10B981;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

.badge-reject {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid #F59E0B;
    padding: 0.25rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

.badge-model {
    background-color: rgba(20, 184, 166, 0.2);
    color: #14B8A6;
    border: 1px solid #14B8A6;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: bold;
}

/* Thanh xác suất tùy chỉnh */
.prob-container {
    margin-bottom: 0.6rem;
}
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: #E2E8F0;
    margin-bottom: 0.2rem;
}
.prob-bar-bg {
    width: 100%;
    background-color: #1E293B;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #14B8A6, #0EA5E9);
    border-radius: 6px;
}
</style>
"""


def apply_theme():
    """Nhúng CSS vào Streamlit."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
