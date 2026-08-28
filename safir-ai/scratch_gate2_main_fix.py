import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update build_report signature
old_build_report_sig = '''    def build_report(
        self,
        *,
        video_source: str,
        sampler,
        evidence_frames: List[EvidenceFrame],
        vlm_response: VLMResponse,
        context,
        decision,
        escalation,
        temporal_events,
        rule_matches,
        latest_timestamp: float,
        detected_events: Optional[List[DetectedEvent]] = None,
        risk_provenance: Optional[RiskProvenance] = None,
        precomputed_evidence_frames: Optional[List[EvidenceFrameOut]] = None,
    ) -> SafirReport:'''

new_build_report_sig = '''    def build_report(
        self,
        *,
        video_source: str,
        sampler,
        evidence_frames: List[EvidenceFrame],
        vlm_response: VLMResponse,
        context,
        decision,
        escalation,
        temporal_events,
        rule_matches,
        latest_timestamp: float,
        detected_events: Optional[List[DetectedEvent]] = None,
        risk_provenance: Optional[RiskProvenance] = None,
        precomputed_evidence_frames: Optional[List[EvidenceFrameOut]] = None,
        analysis_mode: str = "vlm_direct",
    ) -> SafirReport:'''

content = content.replace(old_build_report_sig, new_build_report_sig)

# 2. Update the logic inside build_report
old_build_report_body = '''        current_model_call_ids = set()
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

new_build_report_body = '''        current_model_call_ids = set()
        current_chunk_ids = set()
        current_analysis_ids = set()
        current_evidence_ids = set()
        
        for d in (detected_events or []):
            if d.source_model_call_id:
                current_model_call_ids.add(d.source_model_call_id)
            if d.source_chunk_id:
                current_chunk_ids.add(d.source_chunk_id)
            if d.source_analysis_id:
                current_analysis_ids.add(d.source_analysis_id)
            if d.evidence_ids:
                current_evidence_ids.update(d.evidence_ids)
                
        if len(current_analysis_ids) > 1:
            import logging
            logging.getLogger(__name__).error(f"Security/Invariant violation: Multiple analysis IDs detected in a single pipeline call: {current_analysis_ids}")
            # Controlled failure: Drop mixed events to prevent contamination.
            detected_events = []
            current_analysis_ids.clear()
            
        final_analysis_id = current_analysis_ids.pop() if current_analysis_ids else None
        
        current_call_events = _select_current_call_events(
            temporal_events=temporal_events,
            latest_timestamp=latest_timestamp,
            current_model_call_ids=current_model_call_ids,
            current_chunk_ids=current_chunk_ids,
            current_analysis_id=final_analysis_id,
            current_evidence_ids=current_evidence_ids,
            allow_legacy_timestamp_fallback=False
        )'''

content = content.replace(old_build_report_body, new_build_report_body)

# 3. Update the call in SafirPipeline.run
old_run_call = '''        report = self.build_report(
            video_source=video_source,
            sampler=self._sampler,
            evidence_frames=evidence_frames,
            vlm_response=vlm_response,
            context=context_info,
            decision=decision,
            escalation=escalation,
            temporal_events=temporal_events,
            rule_matches=rule_matches,
            latest_timestamp=latest_timestamp,
            detected_events=detected_events,
            risk_provenance=risk_provenance,
            precomputed_evidence_frames=precomputed_evidence_frames,
        )'''

new_run_call = '''        report = self.build_report(
            video_source=video_source,
            sampler=self._sampler,
            evidence_frames=evidence_frames,
            vlm_response=vlm_response,
            context=context_info,
            decision=decision,
            escalation=escalation,
            temporal_events=temporal_events,
            rule_matches=rule_matches,
            latest_timestamp=latest_timestamp,
            detected_events=detected_events,
            risk_provenance=risk_provenance,
            precomputed_evidence_frames=precomputed_evidence_frames,
            analysis_mode=analysis_mode,
        )'''

content = content.replace(old_run_call, new_run_call)

# 4. Remove allow_legacy_timestamp_fallback parameter mismatch if left from my first hack
content = content.replace("allow_legacy_timestamp_fallback: bool = False", "allow_legacy_timestamp_fallback: bool = False") # It should be already correct in _select_current_call_events signature

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)
