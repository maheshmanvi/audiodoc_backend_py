from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import whisperx
import torch
import os
import shutil
import gc
from dotenv import load_dotenv
from langdetect import detect

# Load environment variables from .env file
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

app = FastAPI()

# Function to process audio with diarization
def process_audio_with_diarization(audio_file_path, hf_token, min_speakers=0, max_speakers=5, batch_size=16):
    # Check if CUDA is available and use GPU if possible, otherwise fallback to CPU
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
        print("CUDA is available, using GPU.")
    else:
        device = "cpu"
        compute_type = "int8"
        print("CUDA is not available, using CPU.")

    # 1. Load WhisperX model for transcription
    model = whisperx.load_model("large-v2", device, task="transcribe", compute_type=compute_type, language="en")

    # 2. Load audio and transcribe with word-level timestamps
    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # Print the raw transcript text
    # print("Raw Transcript from Model:")
    # print(" ".join([segment['text'] for segment in result["segments"]]))  # printing the transcript

    # Check if there are any transcribed segments
    if not result["segments"]:
        print("No active speech found in audio.")
        return {
            "transcript": "No active speech found",
            "srt_format": "No active speech found"
        }

    # Optional: Free up memory
    del model
    gc.collect()

    # 3. Align the transcription with precise word-level timestamps
    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # Optional: Free up memory after alignment
    del align_model
    gc.collect()

    # 4. Diarization (identifying speakers)
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # 5. Assign speaker labels to the transcription
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Detect language of each segment and update accordingly (English or Kannada)
    for segment in result["segments"]:
        detected_language = detect(segment['text'])  # Detect language of the text
        if detected_language == "kn":  # If Kannada detected
            segment['text'] = segment['text']  # Keep as Kannada (no change)
        elif detected_language == "en":  # If English detected
            segment['text'] = segment['text']  # Keep as English (no change)

    # Generate SRT with speaker labels
    srt_content = generate_srt(result["segments"], diarize_segments)

    return {
        "transcript": " ".join([segment['text'] for segment in result["segments"]]),
        "srt_format": srt_content
    }



def generate_srt(segments, diarize_segments, overlap_threshold=0.2, small_gap=0.1):
    srt = []
    speaker_mapping = {}  # Map to track unique speaker labels and assign numbers
    speaker_counter = 1
    prev_end_time = 0.0  # Keeps track of the previous end time to prevent overlap

    for idx, segment in enumerate(segments):
        start_time = segment['start']
        end_time = segment['end']
        speaker_id = segment['speaker']

        # Check if the speaker has been assigned a number, if not, assign a new one
        if speaker_id not in speaker_mapping:
            speaker_mapping[speaker_id] = f"Speaker {speaker_counter}"
            speaker_counter += 1

        speaker = speaker_mapping[speaker_id]
        text = segment['text']

        # Check for overlap with the previous segment
        if start_time < prev_end_time:
            # If overlap is too short, allow overlap and adjust timings
            if prev_end_time - start_time < overlap_threshold:
                start_time = prev_end_time  # Start where the previous segment ends
            else:
                # Otherwise, add a small gap between segments
                start_time = prev_end_time + small_gap

        # Ensure timestamps don't overlap beyond threshold
        prev_end_time = end_time

        # Format start and end times for SRT
        start_str = format_timestamp(start_time)
        end_str = format_timestamp(end_time)

        srt.append(f"{idx + 1}")
        srt.append(f"{start_str} --> {end_str}")
        srt.append(f"{speaker}: {text}")
        srt.append("")

    # print("Transcript after creating srt content:")
    # print("\n".join(srt))  # printing the transcript

    return "\n".join(srt)


# Helper function to format timestamps for SRT
def format_timestamp(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    time_str = f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02},{millis:03}"
    return time_str

# FastAPI endpoint to handle file upload and transcription with diarization
@app.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("en", description="Language for transcription, default is English"),
    min_speakers: int = Form(0, description="Minimum number of speakers for diarization"),
    max_speakers: int = Form(5, description="Maximum number of speakers for diarization")
):
    try:
        audio_file_path = f"temp_{file.filename}"
        with open(audio_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Use the Hugging Face token from the .env file
        hf_token = HUGGING_FACE_TOKEN

        # Call the WhisperX-based function for transcription and diarization
        result = process_audio_with_diarization(
            audio_file_path,
            hf_token,
            # language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

    return {
        "transcript": result["transcript"],
        "srt_format": result["srt_format"] if Form("True") else None
    }
