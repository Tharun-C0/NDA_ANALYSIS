import os
import time
import traceback
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    print("google-genai is not installed")
    exit(1)

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is missing in .env")
        exit(1)
        
    client = genai.Client(api_key=api_key, http_options={'timeout': 60000})
    
    target_model = "models/gemini-3.6-flash"

        
    print(f"model name: {target_model}")
    print("Calling Gemini...")
    
    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=target_model,
            contents="Reply with exactly: GEMINI_TEST_OK"
        )
        end_time = time.time()
        print(f"response: {response.text.strip()}")
        print(f"total request time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        end_time = time.time()
        print(f"FAILED after {end_time - start_time:.2f} seconds")
        traceback.print_exc()

if __name__ == "__main__":
    main()
