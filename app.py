from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from typing import Dict, Any
import shutil
import traceback
import os
import uuid

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment, effects
from basic_pitch.inference import predict

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount static folder for plots ---
plots_dir = os.path.join("static", "plots")
os.makedirs(plots_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


def midi_to_note_name(midi_number: int) -> str:
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"


@app.get("/ping")
async def ping() -> Dict[str, str]:
    return {"message": "pong"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_path = None
    try:
        # 1) save incoming file
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # 2) normalize with pydub
        audio = AudioSegment.from_file(temp_path)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio = effects.normalize(audio)
        audio.export(temp_path, format="wav")

        # 3) Basic Pitch note detection
        model_output = predict(temp_path)
        if not (isinstance(model_output, (list, tuple)) and len(model_output) >= 3):
            raise ValueError("Unexpected prediction output format")
        note_events = model_output[2]

        # filter & format notes
        detected_notes = []
        for start, end, pitch, confidence, _ in note_events:
            dur = end - start
            m = int(pitch)
            if confidence >= 0.6 and dur >= 0.05:
                detected_notes.append({
                    "start_time": float(start),
                    "end_time": float(end),
                    "midi_note": m,
                    "note_name": midi_to_note_name(m),
                    "confidence": float(confidence),
                })

        # 4) Compute librosa features
        y, sr = librosa.load(temp_path, sr=22050)
        duration = float(len(y) / sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        zc = float(np.mean(librosa.zero_crossings(y, pad=False)))
        spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_bw   = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rolloff   = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_str = float(np.mean(onset_env))
        # for a single “pitch” we’ll take the mean of all non-zero piptrack values:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        ave_pitch = float(np.mean(pitches[pitches > 0])) if np.any(pitches>0) else 0.0

        metrics = {
            "tempo": float(tempo),
            "duration": duration,
            "sample_rate": sr,
            "zero_crossings": zc,
            "spectral_centroid": spec_cent,
            "spectral_bandwidth": spec_bw,
            "spectral_rolloff": rolloff,
            "onset_strength": onset_str,
            "pitch": ave_pitch,
            # include the raw notes for sheet-music generation
            "notes": detected_notes,
        }

        # 5) create a waveform plot
        vis_filename = f"{uuid.uuid4().hex}.png"
        vis_path = os.path.join(plots_dir, vis_filename)
        plt.figure(figsize=(8,2))
        plt.plot(y, linewidth=0.3)
        plt.title("Waveform")
        plt.tight_layout()
        plt.savefig(vis_path)
        plt.close()

        # 6) return JSON exactly matching your client’s expectations:
        return {
            "results": {
                "results": metrics
            },
            "visualization": f"/static/plots/{vis_filename}"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
