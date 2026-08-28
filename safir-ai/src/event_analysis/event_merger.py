from typing import List, Dict, Set, Any
import hashlib

from src.event_analysis.schemas import TemporalEvent
from src.utils.config_loader import EventMergerConfig

class EventMerger:
    def __init__(self, config: EventMergerConfig):
        self.config = config

    def merge(self, events: List[TemporalEvent]) -> List[TemporalEvent]:
        if not self.config.enabled or not events:
            return events

        # Group by analysis_id and video_id
        groups: Dict[tuple, List[TemporalEvent]] = {}
        legacy: List[TemporalEvent] = []

        for e in events:
            # Check if event has valid provenance
            has_prov = bool(e.source_analysis_ids and e.source_video_ids and e.source_chunk_ids)
            if not has_prov:
                legacy.append(e)
                continue

            # We assume a TemporalEvent generated from a chunk has exactly one analysis/video id
            # since they come from the same physical occurrence.
            a_id = e.source_analysis_ids[0]
            v_id = e.source_video_ids[0]
            key = (a_id, v_id)
            groups.setdefault(key, []).append(e)

        merged_events: List[TemporalEvent] = list(legacy)

        for (a_id, v_id), group_events in groups.items():
            # Sort chronologically
            group_events.sort(key=lambda x: x.start_timestamp)
            
            merged = []
            for ev in group_events:
                merged = self._merge_event_into_list(ev, merged)
            merged_events.extend(merged)

        # Telemetry info could be recorded here
        return merged_events

    def _merge_event_into_list(self, new_ev: TemporalEvent, current_merged: List[TemporalEvent]) -> List[TemporalEvent]:
        for i, existing_ev in enumerate(current_merged):
            if self._should_merge(new_ev, existing_ev):
                current_merged[i] = self._merge_two_events(existing_ev, new_ev)
                return current_merged
        current_merged.append(new_ev)
        return current_merged

    def _should_merge(self, e1: TemporalEvent, e2: TemporalEvent) -> bool:
        # 1. Check distinct chunk ids
        c1 = set(e1.source_chunk_ids)
        c2 = set(e2.source_chunk_ids)
        # If they share any chunk ID, they are from the same chunk. Do not merge.
        if c1.intersection(c2):
            return False

        # 2. Check temporal IoU / proximity
        start = max(e1.start_timestamp, e2.start_timestamp)
        end = min(e1.end_timestamp, e2.end_timestamp)
        overlap = end - start
        
        # If they don't overlap but are close?
        gap = e2.start_timestamp - e1.end_timestamp if e2.start_timestamp > e1.end_timestamp else e1.start_timestamp - e2.end_timestamp
        
        if overlap > 0:
            union = (e1.end_timestamp - e1.start_timestamp) + (e2.end_timestamp - e2.start_timestamp) - overlap
            iou = overlap / union if union > 0 else 0
            if iou < self.config.min_temporal_iou:
                return False
        else:
            if gap > self.config.max_boundary_gap_sec:
                return False

        # 3. Check event_name and event_type similarity
        if self.config.require_type_compatibility:
            if e1.event_type != e2.event_type:
                return False

        # Simple text similarity via Jaccard for event_name
        def jaccard(s1: str, s2: str) -> float:
            w1 = set(s1.lower().split())
            w2 = set(s2.lower().split())
            union = w1.union(w2)
            if not union: return 0
            return len(w1.intersection(w2)) / len(union)

        if jaccard(e1.event_name, e2.event_name) < self.config.min_label_similarity:
            return False

        return True

    def _merge_two_events(self, e1: TemporalEvent, e2: TemporalEvent) -> TemporalEvent:
        # ID gen
        all_obs_ids = sorted(list(set(e1.source_observation_ids + e2.source_observation_ids)))
        hash_input = "-".join(all_obs_ids)
        new_id = f"merged_{hashlib.sha256(hash_input.encode()).hexdigest()[:8]}"

        start = min(e1.start_timestamp, e2.start_timestamp)
        end = max(e1.end_timestamp, e2.end_timestamp)
        
        # Combine distinct
        def merge_lists(l1, l2):
            result = list(l1)
            for item in l2:
                if item not in result:
                    result.append(item)
            return result

        uncertainties = merge_lists(getattr(e1, "uncertainties", []), getattr(e2, "uncertainties", []))
        if e1.description != e2.description:
            uncertainties.append(f"Contradicting descriptions: '{e1.description}' vs '{e2.description}'")
            
        merged = TemporalEvent(
            event_id=new_id,
            event_name=e1.event_name, # Can just take first
            event_type=e1.event_type,
            description=e1.description, # Ideally combine or keep first
            start_timestamp=start,
            end_timestamp=end,
            duration=end - start,
            confidence=max(e1.confidence, e2.confidence),
            occurrence_count=max(e1.occurrence_count, e2.occurrence_count), # Policy: max
            matched_keywords=merge_lists(e1.matched_keywords, e2.matched_keywords),
            source_model=e1.source_model,
            related_events=merge_lists(e1.related_events, e2.related_events),
            evidence_ids=merge_lists(e1.evidence_ids, e2.evidence_ids),
            source_analysis_ids=merge_lists(e1.source_analysis_ids, e2.source_analysis_ids),
            source_video_ids=merge_lists(e1.source_video_ids, e2.source_video_ids),
            source_chunk_ids=merge_lists(e1.source_chunk_ids, e2.source_chunk_ids),
            source_model_call_ids=merge_lists(e1.source_model_call_ids, e2.source_model_call_ids),
            source_observation_ids=merge_lists(e1.source_observation_ids, e2.source_observation_ids),
            risk_hint=max(e1.risk_hint, e2.risk_hint) if e1.risk_hint and e2.risk_hint else (e1.risk_hint or e2.risk_hint),
            uncertainties=uncertainties
        )
        return merged
