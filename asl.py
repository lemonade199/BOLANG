import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import pygame
import os
import time

# ================================
# 1. Inisialisasi suara (TTS) - Menggunakan gTTS + pygame (lebih stabil)
# ================================

def speak(text):
    """Fungsi untuk mengucapkan teks menggunakan gTTS dan pygame (lebih stabil)."""
    try:
        tts = gTTS(text=text, lang='id')
        filename = "temp.mp3"
        tts.save(filename)
        
        # Inisialisasi pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        # Tunggu sampai suara selesai
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
        os.remove(filename)
        time.sleep(0.3)
    except Exception as e:
        print(f"Error TTS: {e}")

# ================================
# 2. Input nama pengguna
# ================================

user_name = input("Masukkan nama kamu: ").strip()
if not user_name:
    user_name = "Orang Hebat"

# ================================
# 3. Inisialisasi Mediapipe Hands
# ================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3
)

# ================================
# 4. Fungsi bantu deteksi jari
# ================================

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

def classify_letter(fingers):
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
# 5. Jalankan kamera dan tampilkan hasil
# ================================

cap = cv2.VideoCapture(0)
print("\n📸 Kamera aktif — tekan Q untuk keluar\n")

last_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    text = "-"
    fingers = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_states(hand_landmarks)
            text = classify_letter(fingers)

    if fingers:
        print(f"Pola jari: {fingers} -> {text}")

    # Ucapkan kata hanya jika berubah dan bukan "-"
    if text != last_text and text != "-":
        speak(text)
        last_text = text

    cv2.putText(frame, f"Teks: {text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    if fingers:
        cv2.putText(frame, f"Pola: {fingers}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow("🤟 Pengenalan Bahasa Isyarat + Suara", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Kamera dimatikan.")
