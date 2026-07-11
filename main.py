import os
import base64
import io
import pandas as pd
from fastapi import FastAPI, Request
from groq import Groq

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.post("/analyze")
async def analyze_audio(request: Request):
    data = await request.json()
    audio_base64 = data["audio_base64"]
    
    # 1. Decode base64 to bytes
    audio_bytes = base64.b64decode(audio_base64)
    
    # 2. Transcribe using Groq
    # We save to a temporary file-like object for the API
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    
    transcription = client.audio.transcriptions.create(
        file=(audio_file.name, audio_file.getvalue()),
        model="whisper-large-v3-turbo",
    )
    
    text = transcription.text
    words = text.split()
    word_lengths = [len(w) for w in words] if words else [0]
    
    # 3. Create DataFrame for statistics
    df = pd.DataFrame({"word_lengths": word_lengths})
    
    # 4. Return required JSON structure
    return {
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "mean": df.mean().to_dict(),
        "std": df.std().fillna(0).to_dict(),
        "variance": df.var().fillna(0).to_dict(),
        "min": df.min().to_dict(),
        "max": df.max().to_dict(),
        "median": df.median().to_dict(),
        "mode": df["word_lengths"].mode().iloc[0] if not df.empty else 0,
        "range": (df.max() - df.min()).to_dict(),
        "allowed_values": {"word_lengths": df["word_lengths"].unique().tolist()},
        "value_range": {"word_lengths": [float(df["word_lengths"].min()), float(df["word_lengths"].max())]},
        "correlation": df.corr().fillna(0).values.tolist()
    }
