import re
with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r'final_description = "\n  "\.join\(description_parts\)', r'final_description = "\\n".join(description_parts)', text)
text = re.sub(r'final_description = "\r\n  "\.join\(description_parts\)', r'final_description = "\\n".join(description_parts)', text)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(text)
