import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Update caller stage_context
old_call = '''        current_call_events = _select_current_call_events(temporal_events, latest_timestamp, detected_events)'''

new_call = '''        current_model_call_ids = set()
        current_chunk_ids = set()
        current_analysis_id = None
        current_evidence_ids = set()
        is_vlm_direct = False
        
        for d in (detected_events or []):
            if d.source_model_call_id:
                current_model_call_ids.add(d.source_model_call_id)
            if d.source_chunk_id:
                current_chunk_ids.add(d.source_chunk_id)
            if d.source_analysis_id:
                current_analysis_id = d.source_analysis_id
            if d.evidence_ids:
                current_evidence_ids.update(d.evidence_ids)
            if d.source_observation_id:
                is_vlm_direct = True
                
        current_call_events = _select_current_call_events(
            temporal_events=temporal_events,
            latest_timestamp=latest_timestamp,
            current_model_call_ids=current_model_call_ids,
            current_chunk_ids=current_chunk_ids,
            current_analysis_id=current_analysis_id,
            current_evidence_ids=current_evidence_ids,
            is_vlm_direct=is_vlm_direct
        )'''

content = content.replace(old_call, new_call)

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated stage_context caller")
