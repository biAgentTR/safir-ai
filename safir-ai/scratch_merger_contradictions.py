import re

with open("src/event_analysis/event_merger.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add uncertainties to TemporalEvent creation
old_merged = """        merged = TemporalEvent(
            event_id=new_id,
            event_name=e1.event_name, # Can just take first
            event_type=e1.event_type,
            description=e1.description, # Ideally combine or keep first
            start_timestamp=start,"""

new_merged = """        uncertainties = merge_lists(getattr(e1, "uncertainties", []), getattr(e2, "uncertainties", []))
        if e1.description != e2.description:
            uncertainties.append(f"Contradicting descriptions: '{e1.description}' vs '{e2.description}'")
            
        merged = TemporalEvent(
            event_id=new_id,
            event_name=e1.event_name, # Can just take first
            event_type=e1.event_type,
            description=e1.description, # Ideally combine or keep first
            start_timestamp=start,"""
            
content = content.replace(old_merged, new_merged)

# Also add uncertainties to TemporalEvent instantiation arguments
old_kwargs = """            source_model_call_ids=merge_lists(e1.source_model_call_ids, e2.source_model_call_ids),
            source_observation_ids=merge_lists(e1.source_observation_ids, e2.source_observation_ids),
            risk_hint=max(e1.risk_hint, e2.risk_hint) if e1.risk_hint and e2.risk_hint else (e1.risk_hint or e2.risk_hint)
        )"""

new_kwargs = """            source_model_call_ids=merge_lists(e1.source_model_call_ids, e2.source_model_call_ids),
            source_observation_ids=merge_lists(e1.source_observation_ids, e2.source_observation_ids),
            risk_hint=max(e1.risk_hint, e2.risk_hint) if e1.risk_hint and e2.risk_hint else (e1.risk_hint or e2.risk_hint),
            uncertainties=uncertainties
        )"""
content = content.replace(old_kwargs, new_kwargs)

with open("src/event_analysis/event_merger.py", "w", encoding="utf-8") as f:
    f.write(content)
