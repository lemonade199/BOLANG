import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
import tempfile

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ================================
# Fungsi bantu
# ================================
def get_finger_states(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    for tip in tips:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
    return fingers

def classify_letter(fingers, user_name):
    if fingers == [1, 0, 0, 0, 1]:
        return f"Nama saya {user_name}"
    elif fingers == [1, 1, 1, 0, 0]:
        return "Saya"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Halo"
    elif fingers == [1, 0, 1, 0, 1]:
        return "Semuanya"
    elif fingers == [1, 0, 1, 0, 0]:
        return "Nama"
    else:
        return "-"

# ================================
# Generator kamera + suara
# ================================
def generate_frames(user_name="Orang Hebat", lang_code='id'):
    cap = cv2.VideoCapture(0)
    last_text = "-"

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        text = "-"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = get_finger_states(hand_landmarks)
                text = classify_letter(fingers, user_name)

        if text != last_text and text != "-":
            try:
                tts = gTTS(text=text, lang=lang_code)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    temp_path = fp.name
                tts.save(temp_path)
                os.system(f'start /min wmplayer "{temp_path}"')
                last_text = text
            except Exception as e:
                print("TTS Error:", e)

        cv2.putText(frame, f"{text}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
