import re

with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

helpers = """def generate_model_call_id(chunk_id: str) -> str:
    return f"{chunk_id}:vlm"

def generate_observation_id(chunk_id: str, event_index: int) -> str:
    return f"{chunk_id}:observation:{event_index:06d}"

"""

if "def generate_model_call_id" not in content:
    # insert after imports
    content = content.replace("import logging", "import logging\n" + helpers)

old_provenance = """                # C1B: Uygulama sahipli provenance (Modelin ciktisindan bagimsiz)
                if getattr(chunk, 'context', None) is not None:
                    shifted["_provenance"] = {
                        "source_analysis_id": chunk.context.analysis_id,
                        "source_video_id": chunk.context.video_id,
                        "source_chunk_id": chunk.chunk_id,
                        "source_model_call_id": f"{chunk.chunk_id}:vlm",
                        "source_observation_id": f"{chunk.chunk_id}:observation:{event_index:06d}",
                        "relative_start_sec": shifted.get("start_time"),
                        "relative_end_sec": shifted.get("end_time"),
                    }"""

new_provenance = """                # C1B: Uygulama sahipli provenance (Modelin ciktisindan bagimsiz)
                if getattr(chunk, 'context', None) is not None:
                    trusted_provenance = {
                        "source_analysis_id": chunk.context.analysis_id,
                        "source_video_id": chunk.context.video_id,
                        "source_chunk_id": chunk.chunk_id,
                        "source_model_call_id": generate_model_call_id(chunk.chunk_id),
                        "source_observation_id": generate_observation_id(chunk.chunk_id, event_index),
                        "relative_start_sec": shifted.get("start_time"),
                        "relative_end_sec": shifted.get("end_time"),
                    }
                    shifted.pop("_provenance", None)
                    shifted["_provenance"] = trusted_provenance"""

content = content.replace(old_provenance, new_provenance)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Applied Gate 1 modifications to evren_vlm.py")
