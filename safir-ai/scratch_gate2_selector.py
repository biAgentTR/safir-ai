import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

old_func = '''def _select_current_call_events(
    temporal_events: List[TemporalEvent],
    latest_timestamp: float,
    detected_events: Optional[List[DetectedEvent]] = None,
) -> List[TemporalEvent]:'''

new_func = '''def _select_current_call_events(
    temporal_events: List[TemporalEvent],
    latest_timestamp: float,
    current_model_call_ids: set[str],
    current_chunk_ids: set[str],
    current_analysis_id: Optional[str],
    current_evidence_ids: set[str],
    is_vlm_direct: bool = False,
) -> List[TemporalEvent]:'''

content = content.replace(old_func, new_func)

old_body = '''    current_call_evidence_ids = {eid for d in (detected_events or []) for eid in d.evidence_ids}

    def _belongs_to_current_call(te: TemporalEvent) -> bool:
        if current_call_evidence_ids and te.evidence_ids:
            return bool(set(te.evidence_ids) & current_call_evidence_ids)
        return abs(te.end_timestamp - latest_timestamp) <= _CURRENT_CALL_TIMESTAMP_TOLERANCE

    current_call_events = [te for te in temporal_events if _belongs_to_current_call(te)]
    current_call_events.sort(key=lambda te: te.confidence, reverse=True)
    return current_call_events'''

new_body = '''    def _belongs_to_current_call(te: TemporalEvent) -> bool:
        # 1. Model Call ID kesisimi (Kesin)
        if current_model_call_ids and te.source_model_call_ids:
            if set(te.source_model_call_ids) & current_model_call_ids:
                return True
                
        # 2. Chunk ID kesisimi
        if current_chunk_ids and te.source_chunk_ids:
            if set(te.source_chunk_ids) & current_chunk_ids:
                return True
                
        # 3. Evidence ID kesisimi (Eski kare modu veya fallback)
        if current_evidence_ids and te.evidence_ids:
            if set(te.evidence_ids) & current_evidence_ids:
                return True
                
        # 4. Geriye uyumluluk (Provenance YOKSA ve vlm_direct degilse)
        # vlm_direct akisi kesinlikle timestamp fallback'e dusmemeli
        if not is_vlm_direct and not te.source_model_call_ids and not te.source_chunk_ids:
            return abs(te.end_timestamp - latest_timestamp) <= _CURRENT_CALL_TIMESTAMP_TOLERANCE
            
        return False

    current_call_events = [te for te in temporal_events if _belongs_to_current_call(te)]
    
    # Siralama sorumlulugu: Selector yalnizca uyelik secer, gizli siralamayi iptal ediyoruz
    # Siralama veya kronolojik order'i baska yerde yapacagiz veya burada saf kronolojik dondurecegiz.
    # Prompt: "Timeline kronolojik siralanmali. Selector icinde gizli confidence sort bulunmamali."
    current_call_events.sort(key=lambda te: te.start_timestamp)
    return current_call_events'''

content = content.replace(old_body, new_body)

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated _select_current_call_events")
