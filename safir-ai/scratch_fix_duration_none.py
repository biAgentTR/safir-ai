import re

with open("src/vlm/time_normalizer.py", "r", encoding="utf-8") as f:
    content = f.read()

# Change signature
old_sig = """def normalize_observation_time(
    observation: VLMSceneObservation,
    chunk_start_offset_sec: float,
    chunk_duration_sec: float
) -> NormalizedObservationTime:"""

new_sig = """from typing import Optional
def normalize_observation_time(
    observation: VLMSceneObservation,
    chunk_start_offset_sec: float,
    chunk_duration_sec: Optional[float]
) -> NormalizedObservationTime:"""

content = content.replace(old_sig, new_sig)

# Change exceed logic
old_exceed = """    if result.normalized_relative_start_sec > chunk_duration_sec:
        result.time_status = "invalid"
        result.adjustment_reasons.append(f"start ({result.normalized_relative_start_sec}) exceeds chunk duration ({chunk_duration_sec})")
        
    if result.normalized_relative_end_sec > chunk_duration_sec:
        if result.normalized_relative_end_sec <= chunk_duration_sec + 1.0:
            result.normalized_relative_end_sec = chunk_duration_sec
            result.was_adjusted = True
            result.adjustment_reasons.append("end clamped to chunk duration")
        else:
            result.time_status = "invalid"
            result.adjustment_reasons.append(f"end ({result.normalized_relative_end_sec}) exceeds chunk duration ({chunk_duration_sec})")"""

new_exceed = """    if chunk_duration_sec is not None:
        if result.normalized_relative_start_sec > chunk_duration_sec:
            result.time_status = "invalid"
            result.adjustment_reasons.append(f"start ({result.normalized_relative_start_sec}) exceeds chunk duration ({chunk_duration_sec})")
            
        if result.normalized_relative_end_sec > chunk_duration_sec:
            if result.normalized_relative_end_sec <= chunk_duration_sec + 1.0:
                result.normalized_relative_end_sec = chunk_duration_sec
                result.was_adjusted = True
                result.adjustment_reasons.append("end clamped to chunk duration")
            else:
                result.time_status = "invalid"
                result.adjustment_reasons.append(f"end ({result.normalized_relative_end_sec}) exceeds chunk duration ({chunk_duration_sec})")"""

content = content.replace(old_exceed, new_exceed)

with open("src/vlm/time_normalizer.py", "w", encoding="utf-8") as f:
    f.write(content)

# And update base_vlm.py to use None
with open("src/vlm/base_vlm.py", "r", encoding="utf-8") as f:
    base = f.read()

old_base_norm = """norm = normalize_observation_time(obs, 0.0, float('inf'))"""
new_base_norm = """norm = normalize_observation_time(obs, 0.0, None)"""
base = base.replace(old_base_norm, new_base_norm)

with open("src/vlm/base_vlm.py", "w", encoding="utf-8") as f:
    f.write(base)

# And evren_vlm.py too!
with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    evren = f.read()
    
# chunk_duration_sec: float = 0.0 -> Optional[float] = None
evren = re.sub(
    r"chunk_start_offset_sec:\s*float\s*=\s*0\.0,\s*chunk_duration_sec:\s*float\s*=\s*0\.0,",
    "chunk_start_offset_sec: float = 0.0,\n        chunk_duration_sec: Optional[float] = None,",
    evren
)

evren = evren.replace("""chunk_duration_sec=float('inf')""", """chunk_duration_sec=None""")

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(evren)

