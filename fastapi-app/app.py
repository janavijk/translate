from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import os

app = FastAPI()

model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")


def transcribe_audio(file_path: str) -> str:
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    global model
    if model is None:
        return JSONResponse(content={"error": "Model loading..."}, status_code=503)
    file_path = f"uploads/{file.filename}"
    os.makedirs('uploads', exist_ok=True)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    transcription = transcribe_audio(file_path)
    return {"transcription": transcription}
