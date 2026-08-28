import re

with open("src/vlm/base_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add TimeNormalizer logic in _post_chat_completion
old_parse = """        # Modelin urettigi makine-okunur EVENTS_JSON blogunu ayristir; insan-okur
        # aciklamadan ayir (bos ise EventEngine anahtar-kelime fallback'ine duser).
        description, structured_events = parse_structured_events(raw_content)

        latency_ms = (time.perf_counter() - started_at) * 1000"""

new_parse = """        # Yeni Typed Parser kullanimi (Legacy Adapter destekli)
        from src.vlm.parser import parse_vlm_response
        from src.vlm.time_normalizer import normalize_observation_time
        from src.vlm.schemas import VLMAnalysisStatus
        
        description, chunk_res = parse_vlm_response(raw_content)
        
        structured_events = []
        has_invalid = False
        all_invalid = True
        
        if chunk_res.report and chunk_res.report.observations:
            for obs in chunk_res.report.observations:
                # EvrenFrames veya diger frame-tabanli cagrilar tum videoyu tek bir 'chunk' olarak 
                # (veya cercevelerin gercek video zamanlarini) kabul ettigi icin offset 0.
                norm = normalize_observation_time(obs, 0.0, float('inf'))
                
                if norm.time_status == "invalid":
                    has_invalid = True
                    chunk_res.analysis_status = VLMAnalysisStatus.PARTIAL
                    continue
                    
                all_invalid = False
                
                structured_events.append({
                    "event_name": obs.observed_label,
                    "confidence": obs.confidence,
                    "start_time": norm.global_start_sec,
                    "end_time": norm.global_end_sec,
                    "evidence_ids": obs.evidence,
                    "normalized_relative_start_sec": norm.normalized_relative_start_sec,
                    "normalized_relative_end_sec": norm.normalized_relative_end_sec,
                    "was_adjusted": norm.was_adjusted,
                    "adjustment_reasons": norm.adjustment_reasons,
                    "time_status": norm.time_status,
                    "time_base": norm.time_base,
                })
        
        if chunk_res.report and chunk_res.report.observations and all_invalid:
            chunk_res.analysis_status = VLMAnalysisStatus.PARTIAL
            chunk_res.parse_status = "all_times_invalid"

        latency_ms = (time.perf_counter() - started_at) * 1000"""

content = content.replace(old_parse, new_parse)

# Need to update VLMResponse constructor call
old_vlm_response = """        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=image_count,
            latency_ms=latency_ms,
            structured_events=structured_events,
        )"""

new_vlm_response = """        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=image_count,
            latency_ms=latency_ms,
            structured_events=structured_events,
            chunk_analysis_result=chunk_res,
        )"""

content = content.replace(old_vlm_response, new_vlm_response)

with open("src/vlm/base_vlm.py", "w", encoding="utf-8") as f:
    f.write(content)
