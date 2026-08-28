import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update _select_current_call_events signature and body
old_select = '''def _select_current_call_events(
    temporal_events: List[TemporalEvent],
    latest_timestamp: float,
    current_model_call_ids: set[str],
    current_chunk_ids: set[str],
    current_analysis_id: Optional[str],
    current_evidence_ids: set[str],
    is_vlm_direct: bool = False,
) -> List[TemporalEvent]:
    """Bu pipeline cagrisinda uretilen/guncellenen TUM `TemporalEvent`leri secer.'''

new_select = '''def _select_current_call_events(
    temporal_events: List[TemporalEvent],
    latest_timestamp: float,
    current_model_call_ids: set[str],
    current_chunk_ids: set[str],
    current_analysis_id: Optional[str],
    current_evidence_ids: set[str],
    allow_legacy_timestamp_fallback: bool = False,
) -> List[TemporalEvent]:
    """Bu pipeline cagrisinda uretilen/guncellenen TUM `TemporalEvent`leri secer.'''

content = content.replace(old_select, new_select)

old_belongs = '''    def _belongs_to_current_call(te: TemporalEvent) -> bool:
        # 1. Model Call ID kesisimi (Kesin, birincil)
        if current_model_call_ids and te.source_model_call_ids:
            if set(te.source_model_call_ids) & current_model_call_ids:
                return True
                
        # 2. Chunk ID kesisimi (Model Call fail olursa fallback)
        if current_chunk_ids and te.source_chunk_ids:
            if set(te.source_chunk_ids) & current_chunk_ids:
                return True
                
        # 3. Evidence ID kesisimi (Eski kare modu veya fallback)
        if current_evidence_ids and te.evidence_ids:
            if set(te.evidence_ids) & current_evidence_ids:
                return True
                
        # 4. Geriye uyumluluk (Provenance YOKSA ve vlm_direct degilse)
        # vlm_direct akisi kesinlikle timestamp fallback'e dusmez.
        if not is_vlm_direct and not te.source_model_call_ids and not te.source_chunk_ids:
            return abs(te.end_timestamp - latest_timestamp) <= _CURRENT_CALL_TIMESTAMP_TOLERANCE
            
        return False'''

new_belongs = '''    def _belongs_to_current_call(te: TemporalEvent) -> bool:
        # Guvenlik Kontrolu: Eger olay baska bir analysis ID'ye aitse KESINLIKLE reddet!
        if te.source_analysis_ids:
            if not current_analysis_id:
                return False
            if current_analysis_id not in te.source_analysis_ids:
                return False

        # 1. Model Call ID kesisimi (Kesin, birincil)
        if current_model_call_ids and te.source_model_call_ids:
            if set(te.source_model_call_ids) & current_model_call_ids:
                return True
                
        # 2. Chunk ID kesisimi (Model Call fail olursa fallback)
        if current_chunk_ids and te.source_chunk_ids:
            if set(te.source_chunk_ids) & current_chunk_ids:
                return True
                
        # 3. Evidence ID kesisimi (Eski kare modu veya fallback)
        if current_evidence_ids and te.evidence_ids:
            if set(te.evidence_ids) & current_evidence_ids:
                return True
                
        # 4. Geriye uyumluluk (Yalnizca opt-in)
        # Sifir provenance, sifir evidence durumu
        if allow_legacy_timestamp_fallback and not te.source_model_call_ids and not te.source_chunk_ids and not te.evidence_ids:
            import logging
            logging.getLogger(__name__).warning("Legacy timestamp fallback is being used!")
            return abs(te.end_timestamp - latest_timestamp) <= _CURRENT_CALL_TIMESTAMP_TOLERANCE
            
        return False'''

content = content.replace(old_belongs, new_belongs)


# 2. Update stage_context signature and implementation

old_stage_context = '''    def stage_context(
        self,
        vlm_response: VLMResponse,
        user_prompt: str,
        latest_timestamp: float,
        rule_matches,
        temporal_events: Optional[List] = None,
    ):'''

new_stage_context = '''    def stage_context(
        self,
        vlm_response: VLMResponse,
        user_prompt: str,
        latest_timestamp: float,
        rule_matches,
        temporal_events: Optional[List] = None,
        analysis_mode: str = "vlm_direct",
        context: Optional[AnalysisContext] = None,
    ):'''
content = content.replace(old_stage_context, new_stage_context)


old_stage_call = '''        current_model_call_ids = set()
        current_chunk_ids = set()
        current_analysis_id = None
        current_evidence_ids = set()
        is_vlm_direct = False
        
        # Bu setler KESINLIKLE gecmisten gelmez.
        # Yalnizca o anki "EventEngine.detect(...)" ciktisi olan detected_events icinden suzulur.
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

new_stage_call = '''        current_model_call_ids = set()
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
            raise ValueError(f"Bir cagrinın DetectedEvent listesinde iki farkli analysis ID bulundu: {current_analysis_ids}")
            
        final_analysis_id = current_analysis_ids.pop() if current_analysis_ids else (context.analysis_id if context else None)
        
        allow_legacy = (analysis_mode != "vlm_direct")
                
        current_call_events = _select_current_call_events(
            temporal_events=temporal_events,
            latest_timestamp=latest_timestamp,
            current_model_call_ids=current_model_call_ids,
            current_chunk_ids=current_chunk_ids,
            current_analysis_id=final_analysis_id,
            current_evidence_ids=current_evidence_ids,
            allow_legacy_timestamp_fallback=allow_legacy
        )'''
content = content.replace(old_stage_call, new_stage_call)

old_run_stage_context = '''        prompt_block, context = self.stage_context(
            vlm_response, user_prompt, latest_timestamp, rule_matches, temporal_events
        )'''
new_run_stage_context = '''        prompt_block, context = self.stage_context(
            vlm_response, user_prompt, latest_timestamp, rule_matches, temporal_events,
            analysis_mode=analysis_mode, context=context
        )'''
content = content.replace(old_run_stage_context, new_run_stage_context)


with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated main.py with Gate 2 constraints.")
