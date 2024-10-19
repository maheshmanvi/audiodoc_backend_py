from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import whisper
import os
import shutil

app = FastAPI()

# Transcription function
def transcribe_audio(audio_file_path: str, language: str = None):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file_path, language=language)
    return result['segments']

@app.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("en", description="Language for transcription, default is English"),
    timestamp: bool = Form(False, description="If true, returns SRT format with timestamps")
):
    try:
        # Save the uploaded file temporarily
        audio_file_path = f"temp_{file.filename}"
        with open(audio_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Call the transcription function
        transcription = transcribe_audio(audio_file_path, language=language)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Remove the temporary audio file after transcribing
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

    # Prepare the plain text transcript
    plain_text = "".join([segment['text'] for segment in transcription])

    # Prepare the SRT content (if requested)
    srt_content = ""
    if timestamp:
        srt_content = ""
        for i, segment in enumerate(transcription, 1):
            start = convert_seconds_to_srt_time(segment['start'])
            end = convert_seconds_to_srt_time(segment['end'])
            text = segment['text']
            srt_content += f"{i}\n{start} --> {end}\n{text}\n\n"

    # Return both formats in the response
    return {
        "transcript": plain_text,
        "srt_format": srt_content if timestamp else None
    }

# Helper function to convert seconds to SRT time format (HH:MM:SS,MS)
def convert_seconds_to_srt_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"
