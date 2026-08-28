with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "final_description =" in line:
        lines[i] = '        final_description = "\\n".join(description_parts) if description_parts else "Analiz sonucsuz."\n'
        if i+1 < len(lines) and '".join(description_parts)' in lines[i+1]:
            lines[i+1] = ""

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
