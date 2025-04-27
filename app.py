from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tempfile import NamedTemporaryFile
from typing import Dict, Any, List
import shutil
import traceback
import os
import openai
from pydub import AudioSegment, effects
from basic_pitch.inference import predict

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def midi_to_note_name(midi_number: int) -> str:
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = notes[midi_number % 12]
    return f"{note}{octave}"


@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    temp_path = None
    try:
        # Save file
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Process audio
        audio = AudioSegment.from_file(temp_path)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio = effects.normalize(audio)
        audio.export(temp_path, format="wav")

        # Predict
        model_output = predict(temp_path)
        if not isinstance(model_output, (tuple, list)) or len(model_output) < 3:
            raise ValueError(
                f"Unexpected prediction output format: {model_output}")

        note_events = model_output[2]

        raw_notes = []
        for note_tuple in note_events:
            start_time, end_time, pitch, confidence, _ = note_tuple
            duration = end_time - start_time
            midi = int(pitch)

            if confidence >= 0.6 and duration >= 0.05:
                raw_notes.append({
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "midi_note": midi,
                    "note_name": midi_to_note_name(midi),
                    "confidence": float(confidence),
                })

        return {"notes": raw_notes}

    except Exception as e:
        print("❌ Error during analysis:", e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
