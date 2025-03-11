from fastapi import FastAPI, HTTPException, Query
import os
import shutil
import logging

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Hugging Face token (ensure this is loaded from your environment)
HUGGING_FACE_TOKEN = "your_hugging_face_token"  # Replace with your actual token

# Function placeholder for audio processing
def process_audio_with_diarization(audio_file_path, hf_token, language, min_speakers, max_speakers):
    # Placeholder for actual audio processing logic
    return {
        "transcript": "Transcription text here",  # Replace with actual transcription result
        "srt_format": "SRT format text here"  # Replace with actual SRT format result
    }

@app.get("/transcribe/")
async def transcribe(
    language: str = Query("en", description="Language for transcription, default is English"),
    min_speakers: int = Query(2, description="Minimum number of speakers for diarization"),
    max_speakers: int = Query(5, description="Maximum number of speakers for diarization"),
):
    logger.info("Request Received")

    # Define the audio file name
    audio_file_name = "harvard.wav"

    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)

    # Create the full path to the audio file in the same directory
    audio_file_path = os.path.join(current_dir, audio_file_name)

    logger.info(f"File Name is: {audio_file_name}")
    logger.info(f"Audio File Path: {audio_file_path}")

    try:
        # Instead of copying a file, we are directly using the audio file path
        # Use the Hugging Face token from the .env file
        hf_token = HUGGING_FACE_TOKEN

        # Call the WhisperX-based function for transcription and diarization
        result = process_audio_with_diarization(
            audio_file_path,
            hf_token,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

    except Exception as e:
        logger.error(f"Error processing file {audio_file_name}: {str(e)}")  # Log the error
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    # No need for a finally block to remove the file since we are not creating a temp file
    return {
        "transcript": result["transcript"],
        "srt_format": result["srt_format"] if result.get("srt_format") else None
    }
