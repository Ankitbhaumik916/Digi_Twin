import whisper
model = whisper.load_model("base")
result = model.transcribe("your_audio.wav")
print(result["text"])
