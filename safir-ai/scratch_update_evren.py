import re

with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add _provenance dict in _analyze_video_chunks
old_loop = '''            for event in response.structured_events:
                shifted = dict(event)
                for key in ("start_time", "end_time"):
                    value = shifted.get(key)
                    if isinstance(value, (int, float)):
                        shifted[key] = value + chunk.start_offset_sec
                merged_events.append(shifted)'''

new_loop = '''            for event_index, event in enumerate(response.structured_events):
                shifted = dict(event)
                
                # C1B: Uygulama sahipli provenance (Modelin ciktisindan bagimsiz)
                if getattr(chunk, 'context', None) is not None:
                    shifted["_provenance"] = {
                        "source_analysis_id": chunk.context.analysis_id,
                        "source_video_id": chunk.context.video_id,
                        "source_chunk_id": chunk.chunk_id,
                        "source_model_call_id": f"{chunk.chunk_id}:vlm",
                        "source_observation_id": f"{chunk.chunk_id}:observation:{event_index:06d}",
                        "relative_start_sec": shifted.get("start_time"),
                        "relative_end_sec": shifted.get("end_time"),
                    }
                
                for key in ("start_time", "end_time"):
                    value = shifted.get(key)
                    if isinstance(value, (int, float)):
                        shifted[key] = value + chunk.start_offset_sec
                merged_events.append(shifted)'''

content = content.replace(old_loop, new_loop)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated evren_vlm.py with _provenance")
