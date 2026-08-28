import re

# Update evren_vlm.py to use TimeNormalizer and guided JSON
with open("src/vlm/evren_vlm.py", "r", encoding="utf-8") as f:
    evren = f.read()

# Add response_format injection in _send_single_video
old_payload = """        payload = {
            "model": self._endpoint.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    ],
                }
            ],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
        }"""

new_payload = """        payload = {
            "model": self._endpoint.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    ],
                }
            ],
            "max_tokens": self._endpoint.max_new_tokens,
            "temperature": self._endpoint.temperature,
        }
        if getattr(self._endpoint, "provider", "") != "gemini":
            from src.vlm.schemas import VLMObservationReport
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "VLMObservationReport",
                    "schema": VLMObservationReport.model_json_schema(),
                    "strict": True
                }
            }"""

evren = evren.replace(old_payload, new_payload)

# Replace structured_events loop in _send_single_video to run TimeNormalizer
old_structured_events = """        structured_events = []
        if chunk_res.report and chunk_res.report.observations:
            for obs in chunk_res.report.observations:
                structured_events.append({
                    "event_name": obs.observed_label,
                    "confidence": obs.confidence,
                    "start_time": obs.relative_start_sec,
                    "end_time": obs.relative_end_sec,
                    "evidence_ids": obs.evidence
                })"""

new_structured_events = """        from src.vlm.time_normalizer import normalize_observation_time
        structured_events = []
        if chunk_res.report and chunk_res.report.observations:
            for obs in chunk_res.report.observations:
                # We will apply TimeNormalizer here if no chunks (meaning single video)
                # But wait, we need chunk_start_offset_sec and chunk_duration_sec here!
                pass # We will do it in _send_single_video's signature
"""

# Let's completely rewrite the _send_single_video method to take chunk offsets
# We'll use re.sub for the signature
evren = re.sub(
    r"def _send_single_video\(\s*self,\s*video_path:\s*str,\s*prompt:\s*str,\s*on_progress:\s*Optional\[VlmProgressCallback\]\s*=\s*None,\s*chunk_index:\s*int\s*=\s*1,\s*total_chunks:\s*int\s*=\s*1,\s*range_label:\s*Optional\[str\]\s*=\s*None,\s*\)\s*->\s*VLMResponse:",
    """def _send_single_video(
        self,
        video_path: str,
        prompt: str,
        on_progress: Optional[VlmProgressCallback] = None,
        chunk_index: int = 1,
        total_chunks: int = 1,
        range_label: Optional[str] = None,
        chunk_start_offset_sec: float = 0.0,
        chunk_duration_sec: float = 0.0,
    ) -> VLMResponse:""",
    evren
)

# And replace the structured events assembly
evren = evren.replace(old_structured_events, """        from src.vlm.time_normalizer import normalize_observation_time
        from src.vlm.schemas import VLMAnalysisStatus
        structured_events = []
        has_invalid = False
        all_invalid = True
        
        if chunk_res.report and chunk_res.report.observations:
            for obs in chunk_res.report.observations:
                norm = normalize_observation_time(obs, chunk_start_offset_sec, chunk_duration_sec)
                
                # Check for invalid
                if norm.time_status == "invalid":
                    has_invalid = True
                    # Partial
                    chunk_res.analysis_status = VLMAnalysisStatus.PARTIAL
                    # We might skip it or keep it with None. Let's skip invalid time.
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
""")

# Now update analyze_video to pass the correct duration
old_analyze_video_1 = """            response = self._send_single_video(video_source, prompt, on_progress, 1, 1)"""
# Wait, there are multiple matches
evren = evren.replace(
    """        if chunk_duration_sec <= 0:
            response = self._send_single_video(video_source, prompt, on_progress, 1, 1)""",
    """        if chunk_duration_sec <= 0:
            # For single video, we don't have a video length easily without ffprobe, we pass a large number.
            response = self._send_single_video(video_source, prompt, on_progress, 1, 1, chunk_start_offset_sec=0.0, chunk_duration_sec=float('inf'))"""
)

evren = evren.replace(
    """        if len(chunks) == 1:
            response = self._send_single_video(video_source, prompt, on_progress, 1, 1)""",
    """        if len(chunks) == 1:
            response = self._send_single_video(video_source, prompt, on_progress, 1, 1, chunk_start_offset_sec=0.0, chunk_duration_sec=chunks[0].duration_sec)"""
)

# And in _analyze_video_chunks:
old_send_chunk = """                response = self._send_single_video(
                    chunk.path,
                    prompt,
                    on_progress=on_progress,
                    chunk_index=chunk.index + 1,
                    total_chunks=len(chunks),
                    range_label=label,
                )"""

new_send_chunk = """                response = self._send_single_video(
                    chunk.path,
                    prompt,
                    on_progress=on_progress,
                    chunk_index=chunk.index + 1,
                    total_chunks=len(chunks),
                    range_label=label,
                    chunk_start_offset_sec=chunk.start_offset_sec,
                    chunk_duration_sec=chunk.end_offset_sec - chunk.start_offset_sec,
                )"""

evren = evren.replace(old_send_chunk, new_send_chunk)

# And in _analyze_video_chunks, REMOVE the offset addition!
old_offset_loop = """                for key in ("start_time", "end_time"):
                    value = shifted.get(key)
                    if isinstance(value, (int, float)):
                        shifted[key] = value + chunk.start_offset_sec"""

new_offset_loop = """                # offset eklemesi KALDIRILDI - TimeNormalizer _send_single_video icinde halleder.
                # Sadece provenance id'leri ekliyoruz.
                shifted["_provenance"]["normalized_relative_start_sec"] = shifted.get("normalized_relative_start_sec")
                shifted["_provenance"]["normalized_relative_end_sec"] = shifted.get("normalized_relative_end_sec")
                shifted["_provenance"]["was_adjusted"] = shifted.get("was_adjusted")
                shifted["_provenance"]["adjustment_reasons"] = shifted.get("adjustment_reasons")
                shifted["_provenance"]["time_status"] = shifted.get("time_status")
                shifted["_provenance"]["time_base"] = shifted.get("time_base")
                """
evren = evren.replace(old_offset_loop, new_offset_loop)

# Also fix the relative_start_sec in trusted_provenance:
old_trusted = """                        "relative_start_sec": shifted.get("start_time"),
                        "relative_end_sec": shifted.get("end_time"),"""

new_trusted = """                        "relative_start_sec": shifted.get("normalized_relative_start_sec"),
                        "relative_end_sec": shifted.get("normalized_relative_end_sec"),"""

evren = evren.replace(old_trusted, new_trusted)


with open("src/vlm/evren_vlm.py", "w", encoding="utf-8") as f:
    f.write(evren)

