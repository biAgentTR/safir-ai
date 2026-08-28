import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add import
import_stmt = "from src.event_analysis.temporal_reasoner import DEFAULT_RELATION_WINDOW_SEC, TemporalReasoner\nfrom src.event_analysis.event_merger import EventMerger"
content = content.replace("from src.event_analysis.temporal_reasoner import DEFAULT_RELATION_WINDOW_SEC, TemporalReasoner", import_stmt)

# Initialize in __init__
init_stmt = """        self._temporal_reasoner = TemporalReasoner(relation_window_sec=DEFAULT_RELATION_WINDOW_SEC)
        self._event_merger = EventMerger(config.system.merger)"""
content = content.replace("        self._temporal_reasoner = TemporalReasoner(relation_window_sec=DEFAULT_RELATION_WINDOW_SEC)", init_stmt)

# Call in stage_events
call_stmt = """        temporal_events = self._temporal_reasoner.reason(list(self._event_history_buffer))
        temporal_events = self._event_merger.merge(temporal_events)"""
content = content.replace("        temporal_events = self._temporal_reasoner.reason(list(self._event_history_buffer))", call_stmt)

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)
