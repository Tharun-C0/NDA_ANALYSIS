import os, sys
import dotenv
from google import genai

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

test_models = [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite"
]

print("Testing candidate models for ensemble...")
working_models = []
for m in test_models:
    try:
        res = client.models.generate_content(
            model=m,
            contents='Return valid JSON: {"status": "ok"}'
        )
        print(f"SUCCESS: {m}")
        working_models.append(m)
    except Exception as e:
        print(f"FAILED {m}: {e}")

print(f"\nWorking models ({len(working_models)}): {working_models}")
