from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import pygame
import os
import time
from PIL import Image
import io

# ======================================================
# 1. Setup Flask
# ======================================================
app = Flask(__name__)
CORS(app)  # biar Laravel (port 8000) bisa kirim request ke Flask (port 5000)

# ======================================================
# 2. Inisialisasi MediaPipe
# ======================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# ======================================================
# 3. Fungsi bantu deteksi jari
# ======================================================
def get_finger_states(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    thumb_base = hand_landmarks.landmark[2]
    thumb_up = thumb_tip.x > thumb_base.x

    tips = [4, 8, 12, 16, 20]
    fingers = []
    for i, tip in enumerate(tips):
        if i == 0:
            fingers.append(1 if thumb_up else 0)
        else:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
    return fingers

def classify_letter(fingers, user_name="Orang Hebat"):
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

# ======================================================
# 4. Fungsi Suara (gTTS + pygame)
# ======================================================
def speak(text):
    """Fungsi untuk mengucapkan teks menggunakan gTTS dan pygame."""
    try:
        if text == "-":
            return
        tts = gTTS(text=text, lang='id')
        filename = "temp.mp3"
        tts.save(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        os.remove(filename)
    except Exception as e:
        print(f"⚠️ Error TTS: {e}")

# ======================================================
# 5. Route API: menerima frame dari Laravel
# ======================================================
@app.route('/detect', methods=['POST'])
def detect():
    try:
        file = request.files['frame']
        img = Image.open(file.stream).convert('RGB')
        frame = np.array(img)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        text = "-"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = get_finger_states(hand_landmarks)
                text = classify_letter(fingers)
                speak(text)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================================================
# 6. Jalankan server Flask
# ======================================================
if __name__ == '__main__':
    print("🚀 Server Flask berjalan di http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000)
