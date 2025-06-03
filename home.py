import streamlit as st

# 設置頁面配置
st.set_page_config(
    page_title="叫我香香的穴位大師",
    page_icon="💆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 使用 HTML 和 CSS 來美化頁面
st.markdown("""
    <style>
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 80vh;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 40px;
        text-align: center;
    }
    .disclaimer {
        font-size: 20px;
        color: #666;
        margin-bottom: 40px;
        text-align: center;
        max-width: 600px;
        line-height: 1.5;
    }
    </style>
    <div class="main">
        <div class="title">叫我香香的穴位大師</div>
        <div class="disclaimer">
            本平台提供之資訊僅供參考，無法取代專業醫療建議。如有身體不適或疑似重大疾病，請儘速就醫，以確保健康安全。
        </div>
    </div>
""", unsafe_allow_html=True)

# 創建確認按鈕
if st.button("我了解", use_container_width=True):
    # 設置 session state 表示用戶已確認
    st.session_state['disclaimer_accepted'] = True
    # 跳轉到主頁面
    st.switch_page("pages/website.py") 