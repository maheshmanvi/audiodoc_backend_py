import httpx

# Define the API endpoint
url = "http://127.0.0.1:8001/transcribe/"

# Path to the audio file you want to test
#audio_file_path = "harvard.wav"  # Change this to your actual audio file name
audio_file_path = "D:/harvard.wav"
# Prepare the payload
with open(audio_file_path, "rb") as audio_file:
    files = {"file": audio_file}
    data = {
        "language": "en",  # Change this as needed
        "min_speakers": 2,
        "max_speakers": 5
    }

    # Make the request
    response = httpx.post(url, files=files, data=data, timeout=300)

# Print the response
print(response.json())
