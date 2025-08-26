import sounddevice as sd
import soundfile as sf
import whisper
import tempfile
import os
import pyttsx3
from ollama import chat  # latest Ollama SDK

# ===============================
# 1️⃣ Load models
# ===============================

# Whisper STT
stt_model = whisper.load_model("base")

# pyttsx3 TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# ===============================
# 2️⃣ Functions
# ===============================

def record_audio(seconds=5, samplerate=16000, device=None):
    """
    Records audio from mic and saves to temporary WAV file
    """
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32', device=device)
    sd.wait()
    print("✅ Recording finished.")
    
    tmpfile = tempfile.mktemp(suffix=".wav")
    sf.write(tmpfile, audio, samplerate)
    return tmpfile

def speech_to_text(audio_file):
    """
    Transcribes audio to text using Whisper
    """
    result = stt_model.transcribe(audio_file, fp16=False, language="en")
    return result["text"]

def chat_with_ollama(prompt):
    """
    Sends user text to Phi Mini via Ollama and returns AI reply
    """
    response = chat(model="phi3:mini", messages=[{"role": "user", "content": prompt}])
    return response.message.content  # ✅ latest SDK correct way

# ===============================
# 3️⃣ Main chat loop
# ===============================

MIC_DEVICE_INDEX = 1  # change to your AMD mic index

print("💬 Voice AI chat started! Press Ctrl+C to stop.\n")

while True:
    try:
        # 1️⃣ Record
        audio_file = record_audio(5, device=MIC_DEVICE_INDEX)
        user_text = speech_to_text(audio_file)
        os.remove(audio_file)
        
        if user_text.strip() == "":
            print("⚠️ Didn't catch that, try again...")
            continue
        
        print("📝 You:", user_text)
        
        # 2️⃣ AI Reply
        reply = chat_with_ollama(user_text)
        print("🤖 AI:", reply)
        
        # 3️⃣ Speak reply
        speak_text(reply)
    
    except KeyboardInterrupt:
        print("\n🛑 Chat ended.")
        break
    except Exception as e:
        print("❌ Error:", e)
        continue
