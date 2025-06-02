import cv2
import math
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def put_chinese_text(img, text, position, font_size=32, color=(0, 0, 255), font_path=None):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"

    if font_path is None:
        font_path = "/System/Library/Fonts/STHeiti Light.ttc"  # macOS
        # font_path = "C:/Windows/Fonts/msjh.ttc"
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color[::-1])  

    # 轉回OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 初始化 MediaPipe 模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=2,
                                enable_segmentation=False,
                                refine_face_landmarks=True,
                                min_detection_confidence=0.7,
                                min_tracking_confidence=0.7)

# 畫畫
mp_drawing = mp.solutions.drawing_utils

# 相機
cap = cv2.VideoCapture(0)

# 共用座標轉換
def get_point(lm, shape, index):
    h, w = shape[:2]
    return int(lm[index].x * w), int(lm[index].y * h)



# ==== 手部穴位 ====
def hegu(frame, lm, shape):
    x1, y1 = get_point(lm, shape, 2)  # 食指根部
    x2, y2 = get_point(lm, shape, 5)  # 大拇指根部

    # 位置為中點
    x, y = (x1 + x2) // 2, (y1 + y2) // 2

    # 往大拇指偏移一小段
    offset_ratio = -0.2 # 靠近5值
    dx = int((x2 - x1) * offset_ratio)
    dy = int((y2 - y1) * offset_ratio)
    x += dx
    y += dy

    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'hegu', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    


def taiyuan(frame, lm, shape):
    x, y = get_point(lm, shape, 0)  # 手腕
    x5, _ = get_point(lm, shape, 5)  # 食指
    x17, _ = get_point(lm, shape, 17)  # 小指

    # 判斷左右手
    if x5 < x17:
        offset = -40  # 右
    else:
        offset = 40  # 左

    cx = x + offset
    cy = y + 30  # 向下

    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'taiyuan', (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def neiguan(frame, lm, shape):
    x0, y0 = get_point(lm, shape, 0)  
    x9, y9 = get_point(lm, shape, 9)  

    dx = x9 - x0
    dy = y9 - y0

    length = math.sqrt(dx**2 + dy**2)
    unit_dx = dx / length
    unit_dy = dy / length

    # 延伸
    extend_length = -130
    x = int(x0 + unit_dx * extend_length)
    y = int(y0 + unit_dy * extend_length)

    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'neiguan', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def lieque(frame, lm, shape):
    x0, y0 = get_point(lm, shape, 0)   
    x5, y5 = get_point(lm, shape, 5)   
    x17, _ = get_point(lm, shape, 17)  

    # 手腕到食指
    dx = x5 - x0
    dy = y5 - y0
    length = math.sqrt(dx**2 + dy**2)
    unit_dx = dx / length
    unit_dy = dy / length

    along_length = -50  # 偏移
    perp_offset = -70   # 垂直手臂

    # 垂直
    if x5 < x17:
        # 右
        perp_dx = -unit_dy
        perp_dy = unit_dx
    else:
        # 左
        perp_dx = unit_dy
        perp_dy = -unit_dx

    cx = int(x0 + unit_dx * along_length + perp_dx * perp_offset)
    cy = int(y0 + unit_dy * along_length + perp_dy * perp_offset)

    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'lieque', (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def shenmen(frame, lm, shape):
    x0, y0 = get_point(lm, shape, 0)     # 手腕中心點
    x5, y5 = get_point(lm, shape, 5)     # 食指
    x17, y17 = get_point(lm, shape, 17)  # 小指

    # 手腕的左右
    dx = x17 - x5
    dy = y17 - y5
    length = math.sqrt(dx**2 + dy**2)
    unit_dx = dx / length
    unit_dy = dy / length

    # 往小指偏移
    offset = 80
    x = int(x0 + unit_dx * offset)
    y = int(y0 + unit_dy * offset)

    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'shenmen', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def waiguan(frame, lm, shape):
    x0, y0 = get_point(lm, shape, 0)  
    x9, y9 = get_point(lm, shape, 9)  

    dx = x9 - x0
    dy = y9 - y0

    length = math.sqrt(dx**2 + dy**2)
    unit_dx = dx / length
    unit_dy = dy / length

    # 延伸
    extend_length = -130
    x = int(x0 + unit_dx * extend_length)
    y = int(y0 + unit_dy * extend_length)

    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'waiguan', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def houxi(frame, lm, shape):
    x0, y0 = get_point(lm, shape, 0)     # 手腕
    x5, _ = get_point(lm, shape, 5)      # 食指
    x17, y17 = get_point(lm, shape, 17)  # 小指

    # 判斷左右
    is_right_hand = x5 < x17  

    # 方向
    dx = x0 - x17
    dy = y0 - y17
    length = math.sqrt(dx**2 + dy**2)
    unit_dx = dx / length
    unit_dy = dy / length

    # 垂直
    if is_right_hand:
        perp_dx = unit_dy
        perp_dy = -unit_dx
    else:
        perp_dx = -unit_dy
        perp_dy = unit_dx

    # 往手腕延伸
    offset_along = 30  # 手腕
    offset_side = 20   # 外緣

    x = int(x17 + unit_dx * offset_along + perp_dx * offset_side)
    y = int(y17 + unit_dy * offset_along + perp_dy * offset_side)

    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'houxi', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def taibai(frame, lm, shape):
    x1, y1 = get_point(lm, shape, 2)  
   
    cv2.circle(frame, (x1, y1+25), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'taibai', (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def taichong(frame, lm, shape):
    x1, y1 = get_point(lm, shape, 5)  # 左腳內踝
    x2, y2 = get_point(lm, shape, 9)  # 左腳大拇趾

    # 中點
    x, y = (x1 + x2) // 2, (y1 + y2) // 2

    x0, y0 = get_point(lm, shape, 0)

    # 中點向0
    dx, dy = x0 - x, y0 - y

    # 正規化
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    factor = 50  # 移動
    x += int(dx / length * factor)
    y += int(dy / length * factor)

    # 畫圖
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'taichong', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def gongsun(frame, lm, shape):
    x1, y1 = get_point(lm, shape, 2)  
   
    cv2.circle(frame, (x1, y1+40), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'gongsun', (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def hanjian(frame, lm, shape):
    x1, y1 = get_point(lm, shape, 5)  # 左腳內踝
    x2, y2 = get_point(lm, shape, 9)  # 左腳大拇趾

    # 中點
    x, y = (x1 + x2) // 2, (y1 + y2) // 2

    x0, y0 = get_point(lm, shape, 0)

    # 中點向0
    dx, dy = x0 - x, y0 - y

    # 正規化
    length = max((dx**2 + dy**2) ** 0.5, 1e-6)
    factor = -10  # 移動
    x += int(dx / length * factor)
    y += int(dy / length * factor)

    # 畫圖
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'hanjian', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def xiaxi(frame, lm, shape):
    x1, y1 = get_point(lm, shape, 13)  # 左腳內踝
    x2, y2 = get_point(lm, shape, 17)  # 左腳大拇趾

    # 中點
    x, y = (x1 + x2) // 2, (y1 + y2) // 2
    # 畫圖
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'xiaxi', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def zutonggu(frame, lm, shape):
    x, y = get_point(lm, shape, 17)  

    frame_center_x = frame.shape[1] // 2  

    # 判斷左右
    if x < frame_center_x:
        x -= 25  
    else:
        x += 25  

    # 畫圖
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'zutonggu', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# ==== 腳部穴位（支援左右） ====
def taixi(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('taixi (l)', mp_holistic.PoseLandmark.LEFT_ANKLE),
                               ('taixi (r)', mp_holistic.PoseLandmark.RIGHT_ANKLE)]:
        lm = landmarks[lm_id]
        x, y = int(lm.x * w + 10), int(lm.y * h)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def kunlun(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('kunlun (l)', mp_holistic.PoseLandmark.LEFT_HEEL),
                               ('kunlun (r)', mp_holistic.PoseLandmark.RIGHT_HEEL)]:
        lm = landmarks[lm_id]
        x, y = int(lm.x * w + 5), int(lm.y * h + 10)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def qiuxu(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('qiuxu (l)', mp_holistic.PoseLandmark.LEFT_ANKLE),
                               ('qiuxu (r)', mp_holistic.PoseLandmark.RIGHT_ANKLE)]:
        lm = landmarks[lm_id]
        x, y = int(lm.x * w + 20), int(lm.y * h - 10)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def xuehai(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('xuehai (l)', mp_holistic.PoseLandmark.LEFT_KNEE),
                               ('xuehai (r)', mp_holistic.PoseLandmark.RIGHT_KNEE)]:
        lm = landmarks[lm_id]
        x, y = int(lm.x * w - 30), int(lm.y * h - 30)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def zhaohai(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('zhaohai (l)', mp_holistic.PoseLandmark.LEFT_ANKLE),
                               ('zhaohai (r)', mp_holistic.PoseLandmark.RIGHT_ANKLE)]:
        lm = landmarks[lm_id]
        x, y = int(lm.x * w - 10), int(lm.y * h + 15)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def chengshan(frame, landmarks, shape):
    h, w = shape[:2]
    for label, k_id, a_id in [
        ('chengshan (l)', mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE),
        ('chengshan (r)', mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE)]:
        knee = landmarks[k_id]
        ankle = landmarks[a_id]
        x = int((knee.x + ankle.x) / 2 * w)
        y = int((knee.y + ankle.y) / 2 * h + 20)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def shaohai(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('shaohai (l)', mp_holistic.PoseLandmark.LEFT_ELBOW),
                               ('shaohai (r)', mp_holistic.PoseLandmark.RIGHT_ELBOW)]:
        lm = landmarks[lm_id]
        x, y = int(lm.x * w - 15), int(lm.y * h)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def zusanli(frame, landmarks, shape):
    h, w = shape[:2]
    for label, k_id, a_id in [
        ('zusanli (l)', mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE),
        ('zusanli (r)', mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE)]:
        knee = landmarks[k_id]
        ankle = landmarks[a_id]
        x = int((knee.x * 0.7 + ankle.x * 0.3) * w)
        y = int((knee.y * 0.7 + ankle.y * 0.3) * h + 10)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def sanyinjiao(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('sanyinjiao (l)', mp_holistic.PoseLandmark.LEFT_ANKLE),
                               ('sanyinjiao (r)', mp_holistic.PoseLandmark.RIGHT_ANKLE)]:
        lm = landmarks[lm_id]
        x = int(lm.x * w)
        y = int(lm.y * h - 50)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def yinlingquan(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('yinlingquan (l)', mp_holistic.PoseLandmark.LEFT_KNEE),
                               ('yinlingquan (r)', mp_holistic.PoseLandmark.RIGHT_KNEE)]:
        lm = landmarks[lm_id]
        x = int(lm.x * w - 20)
        y = int(lm.y * h + 10)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def yanglingquan(frame, landmarks, shape):
    h, w = shape[:2]
    for label, lm_id in [('yanglingquan (l)', mp_holistic.PoseLandmark.LEFT_KNEE),
                               ('yanglingquan (r)', mp_holistic.PoseLandmark.RIGHT_KNEE)]:
        lm = landmarks[lm_id]
        x = int(lm.x * w + 20)
        y = int(lm.y * h + 10)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def weizhong(frame, landmarks, shape):
    h, w = shape[:2]
    for label, k_id, a_id in [
        ('weizhong (l)', mp_holistic.PoseLandmark.LEFT_KNEE, mp_holistic.PoseLandmark.LEFT_ANKLE),
        ('weizhong (r)', mp_holistic.PoseLandmark.RIGHT_KNEE, mp_holistic.PoseLandmark.RIGHT_ANKLE)]:
        knee = landmarks[k_id]
        ankle = landmarks[a_id]
        x = int((knee.x + ankle.x) / 2 * w)
        y = int((knee.y + ankle.y) / 2 * h - 10)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def yingxiang(frame, landmarks, shape):
    h, w = shape[:2]
    nose = landmarks[mp_holistic.PoseLandmark.NOSE]

    nose_x = int(nose.x * w)
    nose_y = int(nose.y * h)

    offset_x = 30  # 左右偏移
    offset_y = 10  # 垂直

    # 左迎香
    x_left = nose_x - offset_x
    y_left = nose_y + offset_y
    cv2.circle(frame, (x_left, y_left), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'yingxiang (l)', (x_left + 10, y_left - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 右迎香（
    x_right = nose_x + offset_x
    y_right = nose_y + offset_y
    cv2.circle(frame, (x_right, y_right), 10, (0, 0, 255), -1)
    cv2.putText(frame, 'yingxiang (r)', (x_right + 10, y_right - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def quchi(frame, landmarks, shape):
    h, w = shape[:2]
    for name, idx in [('quchi (l)', mp_holistic.PoseLandmark.LEFT_ELBOW),
                      ('quchi (r)', mp_holistic.PoseLandmark.RIGHT_ELBOW)]:
        lm = landmarks[idx]
        x, y = int(lm.x * w + 10), int(lm.y * h - 10)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def shaohai(frame, landmarks, shape):
    h, w = shape[:2]
    for name, idx, x_offset in [
        ('shaohai (l)', mp_holistic.PoseLandmark.LEFT_ELBOW, -10),
        ('shaohai (r)', mp_holistic.PoseLandmark.RIGHT_ELBOW, 10)
    ]:
        lm = landmarks[idx]
        x = int(lm.x * w + x_offset)
        y = int(lm.y * h - 10)  
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(frame, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


# 穴位對應表
hand_map = {
    "合谷": hegu,
    "太淵": taiyuan,
    "內關": neiguan,
    "列缺": lieque,
    "神門": shenmen,
    "外關": waiguan,
    "後谿": houxi,
    "太白": taibai,
    "太衝": taichong,
    "公孫": gongsun,
    "行間": hanjian,
    "俠溪": xiaxi,
    "足通谷": zutonggu
}

body_map = {
    "太溪": taixi,
    "崑崙": kunlun,
    "丘墟": qiuxu,
    "血海": xuehai,
    "照海": zhaohai,
    "承山": chengshan,
    "少海": shaohai,
    "足三里": zusanli,
    "三陰交": sanyinjiao,
    "陰陵泉": yinlingquan,
    "陽陵泉": yanglingquan,
    "委中": weizhong,
    "迎香": yingxiang,
    "曲池": quchi,
    "少海": shaohai
}

# 輸入
choice = input("請輸入欲辨識的穴位：").strip()
if choice not in hand_map and choice not in body_map:
    print("未偵測到指定穴位，預設為『合谷』")
    choice = "合谷"

# main
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    shape = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手
    hand_results = hands.process(rgb)

    # 身體
    holistic_results = holistic.process(rgb)

    # ---------- 手部穴位處理 ----------
    if hand_results.multi_hand_landmarks and choice in hand_map:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            acupoint_function = hand_map[choice]
            acupoint_function(frame, hand_landmarks.landmark, shape)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---------- 身體穴位處理 ----------
    # 身體穴位處理
    if holistic_results.pose_landmarks and choice in body_map:
        landmarks = holistic_results.pose_landmarks.landmark
        body_function = body_map[choice]
        body_function(frame, landmarks, shape)
        mp_drawing.draw_landmarks(frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # 顯示結果
    cv2.imshow("穴位偵測", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 收尾：關閉資源
hands.close()
holistic.close()
cap.release()
cv2.destroyAllWindows()
