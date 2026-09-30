import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gesture_utils import GESTURE_TEXT, classify_gesture
from tts_utils import get_current_text, speak

STABLE_THRESHOLD = 8
COOLDOWN_SECONDS = 2.5
MODEL_PATH = "models/hand_landmarker.task"

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hand(frame, hand_landmarks, w, h):
    for conn in HAND_CONNECTIONS:
        p0 = hand_landmarks[conn.start]
        p1 = hand_landmarks[conn.end]
        cv2.line(frame, (int(p0.x * w), int(p0.y * h)),
                 (int(p1.x * w), int(p1.y * h)), (0, 255, 0), 2)
    for lm in hand_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Kamera tidak terdeteksi atau sedang dipakai aplikasi lain.")
        print("Cek Settings > Privacy > Camera di Windows, lalu jalankan ulang.")
        return

    try:
        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as exc:
        print(f"ERROR: Gagal memuat model hand_landmarker: {exc}")
        cap.release()
        return

    print("Arahkan tangan ke kamera. Tekan 'q' pada window untuk keluar.")

    stable_gesture = None
    stable_count = 0
    triggered = False
    last_trigger_gesture = None
    last_trigger_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Gagal membaca frame dari kamera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            h, w = frame.shape[:2]
            gesture = None
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    draw_hand(frame, hand_landmarks, w, h)
                    gesture = classify_gesture(hand_landmarks)

            now = time.time()
            if gesture == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = gesture
                stable_count = 1 if gesture is not None else 0
                triggered = False

            if (stable_count >= STABLE_THRESHOLD and not triggered
                    and stable_gesture is not None
                    and (stable_gesture != last_trigger_gesture
                         or now - last_trigger_time >= COOLDOWN_SECONDS)):
                sentence = GESTURE_TEXT.get(stable_gesture)
                if sentence:
                    speak(sentence)
                    triggered = True
                    last_trigger_gesture = stable_gesture
                    last_trigger_time = now
                    print(f"[TTS] Gesture '{stable_gesture}' -> mengucapkan: {sentence}")

            status = stable_gesture or "-"
            cv2.putText(frame, f"Gesture: {status} | stable {stable_count}/{STABLE_THRESHOLD}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            spoken = get_current_text()
            if spoken:
                (tw, th), _ = cv2.getTextSize(spoken, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                bx = max((w - tw) // 2 - 10, 5)
                by = h - 30 - th
                cv2.rectangle(frame, (bx - 5, by - th - 8),
                              (bx + tw + 25, h - 8), (0, 0, 0), -1)
                cv2.putText(frame, spoken, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 255), 2)

            cv2.imshow("Gesture to Voice", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")