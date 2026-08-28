import re

with open("src/event_analysis/temporal_reasoner.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add _union_provenance helper
union_provenance_func = """
    @staticmethod
    def _union_provenance(group: List[DetectedEvent], field_name: str) -> List[str]:
        seen = []
        for event in group:
            val = getattr(event, field_name, None)
            if val is not None and val not in seen:
                seen.append(val)
        return seen

    @staticmethod
    def _union_evidence_ids(group: List[DetectedEvent]) -> List[str]:"""

content = content.replace('''    @staticmethod
    def _union_evidence_ids(group: List[DetectedEvent]) -> List[str]:''', union_provenance_func)

# Add provenance fields to _build_temporal_event instantiation
build_insert_point = "            evidence_ids=self._union_evidence_ids(group),"
build_new_fields = """            evidence_ids=self._union_evidence_ids(group),
            source_analysis_ids=self._union_provenance(group, "source_analysis_id"),
            source_video_ids=self._union_provenance(group, "source_video_id"),
            source_chunk_ids=self._union_provenance(group, "source_chunk_id"),
            source_model_call_ids=self._union_provenance(group, "source_model_call_id"),
            source_observation_ids=self._union_provenance(group, "source_observation_id"),"""

content = content.replace(build_insert_point, build_new_fields)

with open("src/event_analysis/temporal_reasoner.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated temporal_reasoner.py with provenance logic.")
