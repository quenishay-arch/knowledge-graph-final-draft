import re
import json

with open("data/extracted_text/raw_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Flexible unit detection
pattern = r"(Unit\s*\d+.*|UNIT\s*\d+.*|Module\s*\d+.*|Chapter\s*\d+.*)"

parts = re.split(pattern, text)

print("TOTAL PARTS:", len(parts))

chunks = []

current_title = None

for part in parts:

    if re.match(pattern, part):
        current_title = part.strip()

    else:
        if current_title and part.strip():

            chunks.append({
                "unit_title": current_title,
                "content": part.strip()
            })

print("TOTAL CHUNKS:", len(chunks))

with open("data/chunks/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Chunking complete.")
