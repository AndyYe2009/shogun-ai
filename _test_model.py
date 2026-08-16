from capture import capture_screenshot
from screen_parser import parse_to_gamestate
from state import state_to_text
from ollama_client import ollama_chat, DECISION_SYSTEM_PROMPT
import json

img = capture_screenshot()
gs = parse_to_gamestate(img, turn_number=1)
txt = state_to_text(gs)
print(f"State: {len(txt)} chars")

for i in range(3):
    resp = ollama_chat("minicpm-v:8b", txt, system=DECISION_SYSTEM_PROMPT, max_tokens=128, timeout=30)
    print(f"\nTest {i+1}: LEN={len(resp)}")
    text = resp.strip()
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        d = json.loads(text)
        print(f"  VALID: {json.dumps(d)}")
    except Exception as e:
        print(f"  INVALID: [{text[:150]}]")
