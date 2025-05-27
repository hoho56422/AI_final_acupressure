import streamlit as st

# 設置頁面配置，隱藏側邊欄
st.set_page_config(
    page_title="芳療穴位治療",
    page_icon="💆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <h1 style='text-align: center;'>芳療穴位治療</h1>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_greeting = True

if st.session_state.show_greeting and len(st.session_state.messages) == 0:
    st.markdown("""
        <div style='text-align: center; font-size: 24px; margin-top: 100px;'>
            你今天還好嗎？
        </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if st.button("查看穴位", key=f"camera_btn_{len(st.session_state.messages)}"):
                st.switch_page("pages/camera.py")

if prompt := st.chat_input("請輸入您的症狀"):
    st.session_state.show_greeting = False
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": "我了解您的症狀，需要查看穴位嗎？"})
    with st.chat_message("assistant"):
        st.markdown("我了解您的症狀，需要查看穴位嗎？")
        if st.button("查看穴位", key=f"camera_btn_{len(st.session_state.messages)}_reply"):
            st.switch_page("pages/camera.py")
