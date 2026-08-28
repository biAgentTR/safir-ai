import re
with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('final_description = "\n".join(description_parts) if description_parts else "Analiz sonucsuz."\n".join(description_parts) if description_parts else "Analiz sonucsuz."', 'final_description = "\\n".join(description_parts) if description_parts else "Analiz sonucsuz."')

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(content)
