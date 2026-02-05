import base64
import requests

# 1. Convert MP3 → Base64
with open("test.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# 2. Prepare request
payload = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": audio_base64
}

headers = {
    "Content-Type": "application/json",
    "x-api-key": "sk_test_123456789"
}

# 3. Call API
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    json=payload,
    headers=headers
)

print(response.json())
