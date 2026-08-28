import re

with open("src/event_analysis/event_engine.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add _provenance reading in _detect_from_structured
old_append = '''            detections.append(
                DetectedEvent(
                    event_name=event_name,
                    event_type=event_type,
                    description=evidence,
                    timestamp=start_time,
                    end_timestamp=end_time,
                    confidence=confidence,
                    matched_keywords=keywords,
                    source_model=engine_input.source_model,
                    vlm_event_id=str(item.get("event_id")) if item.get("event_id") is not None else None,
                    evidence_ids=evidence_ids,
                    risk_hint=self._coerce_risk_hint(item.get("risk_score")),
                )
            )'''

new_append = '''            prov = item.get("_provenance", {})
            detections.append(
                DetectedEvent(
                    event_name=event_name,
                    event_type=event_type,
                    description=evidence,
                    timestamp=start_time,
                    end_timestamp=end_time,
                    confidence=confidence,
                    matched_keywords=keywords,
                    source_model=engine_input.source_model,
                    vlm_event_id=str(item.get("event_id")) if item.get("event_id") is not None else None,
                    evidence_ids=evidence_ids,
                    risk_hint=self._coerce_risk_hint(item.get("risk_score")),
                    source_analysis_id=prov.get("source_analysis_id"),
                    source_video_id=prov.get("source_video_id"),
                    source_chunk_id=prov.get("source_chunk_id"),
                    source_model_call_id=prov.get("source_model_call_id"),
                    source_observation_id=prov.get("source_observation_id"),
                    relative_start_sec=prov.get("relative_start_sec"),
                    relative_end_sec=prov.get("relative_end_sec"),
                )
            )'''

content = content.replace(old_append, new_append)

with open("src/event_analysis/event_engine.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated event_engine.py with provenance parsing")
