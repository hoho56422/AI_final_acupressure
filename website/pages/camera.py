import streamlit as st
import cv2
import os

os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

st.markdown("""
    <h1 style='text-align: center;'>鏡頭畫面</h1>
""", unsafe_allow_html=True)

def init_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("無法開啟攝像頭，請確保已授予攝像頭訪問權限。")
        return None
    return cap

camera_placeholder = st.empty()

if 'camera' not in st.session_state:
    st.session_state.camera = init_camera()

if st.button("返回聊天頁面"):
    if 'camera' in st.session_state and st.session_state.camera is not None:
        st.session_state.camera.release()
        del st.session_state.camera
    st.switch_page("website.py")

if 'camera' in st.session_state and st.session_state.camera is not None:
    try:
        while True:
            ret, frame = st.session_state.camera.read()
            if ret:
                # mirror mode
                frame = cv2.flip(frame, 1)
                
                # 将BGR转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                camera_placeholder.image(frame_rgb, channels="RGB")
            else:
                st.error("無法讀取畫面")
                break
                
    except Exception as e:
        st.error(f"發生錯誤：{str(e)}")
    finally:
        if 'camera' in st.session_state and st.session_state.camera is not None:
            st.session_state.camera.release() 