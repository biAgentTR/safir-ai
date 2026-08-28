with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    evren = f.read()

evren = evren.replace(
    """                if "_provenance" not in shifted:\\n                    shifted["_provenance"] = {}\\n                shifted["_provenance"]["normalized_relative_start_sec"] = shifted.get("normalized_relative_start_sec")""",
    """                if "_provenance" not in shifted:
                    shifted["_provenance"] = {}
                shifted["_provenance"]["normalized_relative_start_sec"] = shifted.get("normalized_relative_start_sec")"""
)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(evren)
