import os
import sys
sys.path.append(r"f:\NDA")
from scripts.llm_auto_review import init_gemini

client = init_gemini()
for m in client.models.list():
    print(m.name)
