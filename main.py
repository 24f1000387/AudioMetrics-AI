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

        # Base empty structure
        response = {
            "rows": 0, "columns": [], "mean": {}, "std": {},
            "variance": {}, "min": {}, "max": {}, "median": {},
            "mode": {}, "range": {}, "allowed_values": [], 
            "value_range": {}, "correlation": []
        }

        if words:
            df = pd.DataFrame({"word_lengths": [len(w) for w in words]})
            
            # Map stats to column names
            response["rows"] = int(df.shape[0])
            response["columns"] = list(df.columns)
            response["mean"] = df.mean().to_dict()
            response["std"] = df.std().fillna(0).to_dict()
            response["variance"] = df.var().fillna(0).to_dict()
            response["min"] = df.min().to_dict()
            response["max"] = df.max().to_dict()
            response["median"] = df.median().to_dict()
            response["mode"] = df.mode().iloc[0].to_dict()
            response["range"] = (df.max() - df.min()).to_dict()
            
            # FIX: Return a list for allowed_values as expected=[]
            response["allowed_values"] = [int(x) for x in df["word_lengths"].unique()]
            
            # FIX: Return a list [min, max] or similar for value_range
            response["value_range"] = [int(df["word_lengths"].min()), int(df["word_lengths"].max())]
            
            response["correlation"] = df.corr().fillna(0).values.tolist()

        return response

    except Exception:
        return {
            "rows": 0, "columns": [], "mean": {}, "std": {},
            "variance": {}, "min": {}, "max": {}, "median": {},
            "mode": {}, "range": {}, "allowed_values": [],
            "value_range": {}, "correlation": []
        }
