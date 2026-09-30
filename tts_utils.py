"""Text-to-speech: edge-tts (suara neural natural) dengan fallback pyttsx3."""
import asyncio
import hashlib
import os
import tempfile
import threading

import edge_tts
import pyttsx3

_speak_lock = threading.Lock()
_current_text = None
_current_text_lock = threading.Lock()

VOICE = "id-ID-ArdiNeural"
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "gerture_tts_cache")

FALLBACK_VOLUME = 1.0
FALLBACK_RATE = 160
PREFERRED_VOICE_KEYWORDS = ("david", "george", "daniel", "harry", "guy")


def _cache_path(text):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    digest = hashlib.md5(f"{VOICE}:{text}".encode()).hexdigest()
    return os.path.join(_CACHE_DIR, digest + ".mp3")


def _play_with_edge(text):
    import sounddevice as sd
    import soundfile as sf

    path = _cache_path(text)
    if not os.path.exists(path):
        asyncio.run(edge_tts.Communicate(text, VOICE).save(path))
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()


def _play_with_pyttsx3(text):
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("volume", FALLBACK_VOLUME)
        engine.setProperty("rate", FALLBACK_RATE)
        for voice in engine.getProperty("voices"):
            identifier = (voice.id or "").lower()
            name = (voice.name or "").lower()
            if any(keyword in identifier or keyword in name
                   for keyword in PREFERRED_VOICE_KEYWORDS):
                engine.setProperty("voice", voice.id)
                break
        engine.say(text)
        engine.runAndWait()
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


def _set_current_text(text):
    global _current_text
    with _current_text_lock:
        _current_text = text


def get_current_text():
    with _current_text_lock:
        return _current_text


def _speak_worker(text):
    _set_current_text(text)
    try:
        with _speak_lock:
            _play_with_edge(text)
    except Exception as exc:
        print(f"[TTS] edge-tts gagal ({exc}), fallback ke suara sistem.")
        try:
            with _speak_lock:
                _play_with_pyttsx3(text)
        except Exception as fallback_exc:
            print(f"[TTS] Fallback juga gagal: {fallback_exc}")
    finally:
        _set_current_text(None)


def speak(text):
    if not text:
        return
    threading.Thread(target=_speak_worker, args=(text,), daemon=True).start()