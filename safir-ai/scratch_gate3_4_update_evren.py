import re

with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add import for parser
import_str = "from src.vlm.parser import parse_vlm_response"
if import_str not in content:
    content = content.replace("from src.vlm.base_vlm import (", f"{import_str}\nfrom src.vlm.base_vlm import (")

# Modify _send_single_video parsing logic
old_parse = """        raise_if_empty_content(raw_content, video_path, data)
        description, structured_events = parse_structured_events(raw_content)
        latency_ms = (time.perf_counter() - started_at) * 1000
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=0,
            latency_ms=latency_ms,
            structured_events=structured_events,
        )"""

new_parse = """        raise_if_empty_content(raw_content, video_path, data)
        # Yeni Typed Parser kullanimi (Legacy Adapter destekli)
        description, chunk_res = parse_vlm_response(raw_content)
        
        # Geriye uyumluluk: VLMResponse structured_events degeri hala dictionary olarak doldurulur, 
        # ta ki EventEngine tamamen Typed nesnelere (VLMObservationReport) gecene kadar.
        structured_events = []
        if chunk_res.report and chunk_res.report.observations:
            for obs in chunk_res.report.observations:
                structured_events.append({
                    "event_name": obs.observed_label,
                    "confidence": obs.confidence,
                    "start_time": obs.relative_start_sec,
                    "end_time": obs.relative_end_sec,
                    "evidence_ids": obs.evidence
                })
                
        latency_ms = (time.perf_counter() - started_at) * 1000
        return VLMResponse(
            description=description,
            model_name=self.model_name,
            frame_count=0,
            latency_ms=latency_ms,
            structured_events=structured_events,
            chunk_analysis_result=chunk_res,
        )"""

content = content.replace(old_parse, new_parse)

# We also need to fix _analyze_video_chunks where parse_structured_events is used.
old_chunk_parse = """            raise_if_empty_content(raw_content, chunk.path, data)
            description, structured_events = parse_structured_events(raw_content)"""

new_chunk_parse = """            raise_if_empty_content(raw_content, chunk.path, data)
            description, chunk_res = parse_vlm_response(raw_content)
            structured_events = []
            if chunk_res.report and chunk_res.report.observations:
                for obs in chunk_res.report.observations:
                    structured_events.append({
                        "event_name": obs.observed_label,
                        "confidence": obs.confidence,
                        "start_time": obs.relative_start_sec,
                        "end_time": obs.relative_end_sec,
                        "evidence_ids": obs.evidence
                    })
            # Gelecekte ChunkAnalysisResult (chunk_res) toplanip VLMResponse'a set edilebilir.
            """

content = content.replace(old_chunk_parse, new_chunk_parse)

with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(content)
