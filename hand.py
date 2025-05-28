import cv2
import mediapipe as mp

# 初始化 MediaPipe Hands 模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,     # 即時模式
    max_num_hands=1,             # 只偵測一隻手
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 開啟攝影機
cap = cv2.VideoCapture(1)  

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("無法讀取畫面")
        break

    # 翻轉畫面（自拍鏡像效果）
    frame = cv2.flip(frame, 1)

    # 轉成 RGB 並送進 MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # 取拇指MCP（2）和食指MCP（5）
            point_thumb = hand_landmarks.landmark[2]
            point_index = hand_landmarks.landmark[5]

            # 計算合谷穴位置
            x = int((point_thumb.x + point_index.x) / 2 * w)
            y = int((point_thumb.y + point_index.y) / 2 * h)

            # 畫出手部骨架
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 標記合谷穴
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, '合谷', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 顯示結果
    cv2.imshow('HeGu Acupoint - 合谷穴即時偵測', frame)
    
    # 按q離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 結束攝影機與視窗
cap.release()
cv2.destroyAllWindows()