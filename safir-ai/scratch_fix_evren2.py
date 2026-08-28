with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if line.strip().startswith("final_description ="):
        new_lines.append('        final_description = "\\n".join(description_parts) if description_parts else "Analiz sonucsuz."\n')
        skip = True
    elif skip and line.strip() == "":
        continue
    elif skip and line.strip().startswith("return VLMResponse("):
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
