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
    try:
        data = await request.json()
        audio_base64 = data.get("audio_base64")
        
        audio_bytes = base64.b64decode(audio_base64)
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_file.getvalue()),
            model="whisper-large-v3-turbo",
        )
        
        text = transcription.text.strip()
        words = text.split()

        # 1. EMPTY CASE: Explicitly return {} for empty objects
        if not words:
            return {
                "rows": 0, "columns": [], "mean": {}, "std": {},
                "variance": {}, "min": {}, "max": {}, "median": {},
                "mode": {}, "range": {}, "allowed_values": {}, 
                "value_range": {}, "correlation": []
            }

        # 2. DATA CASE: Return {"word_lengths": [...]}
        # We explicitly cast to list/int to avoid numpy types that JSON serializers hate
        word_lens = [int(len(w)) for w in words]
        df = pd.DataFrame({"word_lengths": word_lens})
        
        return {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "mean": {k: float(v) for k, v in df.mean().to_dict().items()},
            "std": {k: float(v) for k, v in df.std().fillna(0).to_dict().items()},
            "variance": {k: float(v) for k, v in df.var().fillna(0).to_dict().items()},
            "min": {k: int(v) for k, v in df.min().to_dict().items()},
            "max": {k: int(v) for k, v in df.max().to_dict().items()},
            "median": {k: float(v) for k, v in df.median().to_dict().items()},
            "mode": {k: int(v) for k, v in df.mode().iloc[0].to_dict().items()},
            "range": {k: int(v) for k, v in (df.max() - df.min()).to_dict().items()},
            "allowed_values": {"word_lengths": [int(x) for x in df["word_lengths"].unique()]},
            "value_range": {"word_lengths": [int(df["word_lengths"].min()), int(df["word_lengths"].max())]},
            "correlation": df.corr().fillna(0).values.tolist()
        }

    except Exception as e:
        # Fallback to empty structure
        return {
            "rows": 0, "columns": [], "mean": {}, "std": {},
            "variance": {}, "min": {}, "max": {}, "median": {},
            "mode": {}, "range": {}, "allowed_values": {},
            "value_range": {}, "correlation": []
        }
