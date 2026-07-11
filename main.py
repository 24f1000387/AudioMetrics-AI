import os
import base64
import io
import pandas as pd
import logging
from fastapi import FastAPI, Request, HTTPException
from groq import Groq

# Set up logging for Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Ensure API Key is loaded
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    logger.error("GROQ_API_KEY is not set!")
    
client = Groq(api_key=api_key)

@app.post("/analyze")
async def analyze_audio(request: Request):
    try:
        data = await request.json()
        audio_base64 = data.get("audio_base64")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="No audio_base64 provided")

        # Decode base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Transcribe
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", audio_file.getvalue()),
            model="whisper-large-v3-turbo",
        )
        
        # Logic for statistics
        text = transcription.text
        words = text.split()
        word_lengths = [len(w) for w in words] if words else [0]
        df = pd.DataFrame({"word_lengths": word_lengths})
        
        # Prepare response
        return {
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "mean": df.mean().to_dict(),
            "std": df.std().fillna(0).to_dict(),
            "variance": df.var().fillna(0).to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict(),
            "median": df.median().to_dict(),
            "mode": int(df["word_lengths"].mode().iloc[0]) if not df.empty else 0,
            "range": (df.max() - df.min()).to_dict(),
            "allowed_values": {"word_lengths": df["word_lengths"].unique().tolist()},
            "value_range": {"word_lengths": [float(df["word_lengths"].min()), float(df["word_lengths"].max())]},
            "correlation": df.corr().fillna(0).values.tolist()
        }
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
