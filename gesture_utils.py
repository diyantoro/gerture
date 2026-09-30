"""Klasifikasi gesture tangan dari 21 landmark MediaPipe (handedness-independent)."""
import math

GESTURE_TEXT = {
    "Thumbs Up": "Halo everyone",
    "Peace": "Perkenalkan nama saya Diyantoro",
    "Open Palm": "Iwak tempek",
}

# Indeks landmark jari (sesuai urutan MediaPipe HandLandmark)
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

ANGLE_COS_THRESHOLD = 0.5  # sudut < 60 derajat => ibu jari lurus/terbuka


def _is_thumb_extended(landmarks):
    mcp = landmarks[THUMB_MCP]
    ip = landmarks[THUMB_IP]
    tip = landmarks[THUMB_TIP]
    ux, uy = ip.x - mcp.x, ip.y - mcp.y
    vx, vy = tip.x - ip.x, tip.y - ip.y
    len_u = math.hypot(ux, uy)
    len_v = math.hypot(vx, vy)
    if len_u == 0.0 or len_v == 0.0:
        return False
    cosine = (ux * vx + uy * vy) / (len_u * len_v)
    return cosine > ANGLE_COS_THRESHOLD


def _is_finger_extended(tip_idx, pip_idx, landmarks):
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def get_fingers_extended(landmarks):
    thumb = _is_thumb_extended(landmarks)
    index = _is_finger_extended(INDEX_TIP, INDEX_PIP, landmarks)
    middle = _is_finger_extended(MIDDLE_TIP, MIDDLE_PIP, landmarks)
    ring = _is_finger_extended(RING_TIP, RING_PIP, landmarks)
    pinky = _is_finger_extended(PINKY_TIP, PINKY_PIP, landmarks)
    return thumb, index, middle, ring, pinky


def classify_gesture(landmarks):
    thumb, index, middle, ring, pinky = get_fingers_extended(landmarks)
    if thumb and index and middle and ring and pinky:
        return "Open Palm"
    if thumb and not (index or middle or ring or pinky):
        return "Thumbs Up"
    if index and middle and not (ring or pinky):
        return "Peace"
    return None