import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import json
import os

# 設置頁面配置，隱藏側邊欄
st.set_page_config(
    page_title="叫我香香的穴位大師",
    page_icon="💆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 檢查用戶是否已接受免責聲明
if 'disclaimer_accepted' not in st.session_state:
    st.switch_page("home.py")

# 獲取當前文件的絕對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "trained_model")

# 載入模型和標籤映射
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    with open(os.path.join(model_path, "id2label.json"), "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    return model, tokenizer, label_mapping

# 載入芳療和穴位資料
@st.cache_resource
def load_remedies():
    with open("ramedy.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 預測症狀函數
def predict_symptom(text: str):
    model, tokenizer, id2label = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()
    return id2label[str(pred)]

st.markdown("""
    <h1 style='text-align: center;'>叫我香香的穴位大師</h1>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_greeting = True

if st.session_state.show_greeting and len(st.session_state.messages) == 0:
    st.markdown("""
        <div style='text-align: center; font-size: 24px; margin-top: 100px;'>
            請描述您的症狀，我將為您提供芳療和穴位建議
        </div>
    """, unsafe_allow_html=True)

# 顯示歷史訊息和按鈕
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "has_remedy" in message and message["has_remedy"]:
            # 使用唯一的 key，並傳遞症狀資訊
            if st.button("查看穴位", key=f"camera_btn_{i}"):
                # 設置對應的症狀
                st.session_state['current_symptom'] = message["symptom"]
                st.switch_page("pages/camera.py")

if prompt := st.chat_input("請輸入您的症狀"):
    st.session_state.show_greeting = False
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 使用模型預測症狀
    predicted_symptom = predict_symptom(prompt)
    remedies = load_remedies()
    
    # 獲取對應的芳療和穴位建議
    if predicted_symptom in remedies:
        remedy = remedies[predicted_symptom]
        # 將芳療和穴位建議中的 ~ 替換為 ～
        aromatherapy = [item.replace("~", "～") for item in remedy['芳療']]
        acupoints = [item.replace("~", "～") for item in remedy['穴位']]
        
        response = f"""根據您的描述，您可能有{predicted_symptom}的症狀。

**芳療建議：**

{chr(10).join(['• ' + item + chr(10) for item in aromatherapy])}

**穴位建議：**

{chr(10).join(['• ' + item + chr(10) for item in acupoints])}

需要查看相關穴位嗎？"""
        
        # 加入訊息到 session state，包含症狀資訊和是否有治療方案的標記
        message_data = {
            "role": "assistant", 
            "content": response, 
            "has_remedy": True, 
            "symptom": predicted_symptom
        }
        st.session_state.messages.append(message_data)
        st.session_state['current_symptom'] = predicted_symptom
        
        with st.chat_message("assistant"):
            st.markdown(response)
            # 使用唯一的 key 避免重複
            if st.button("查看穴位", key=f"camera_btn_{len(st.session_state.messages)-1}_new"):
                st.session_state['current_symptom'] = predicted_symptom
                st.switch_page("pages/camera.py")
    else:
        response = f"不好意思，我沒有辦法判斷您的症狀，請問可以描述得更清楚一點嗎？"
        
        # 無法判斷症狀時，不包含 has_remedy 標記
        message_data = {
            "role": "assistant", 
            "content": response, 
            "has_remedy": False
        }
        st.session_state.messages.append(message_data)
        
        with st.chat_message("assistant"):
            st.markdown(response)
            # 無法判斷症狀時不顯示按鈕