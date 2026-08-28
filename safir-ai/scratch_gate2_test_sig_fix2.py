import re

with open("tests/test_pipeline_integration.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replace all occurrences of `result = _select_current_call_events(events, <number>, [])`
# and similar calls.

def fix_call(match):
    # args might be: events, 15.0, []
    # or events, 100.0, detected
    args = match.group(1)
    
    # We will just parse the arguments loosely.
    # The signature is: events, latest_timestamp, current_model_call_ids, current_chunk_ids, current_analysis_id, current_evidence_ids, is_vlm_direct
    # We'll just provide dummy empty sets for the tests since they are testing legacy behavior.
    return f"""    result = _select_current_call_events(
        {args.split(',')[0]},
        {args.split(',')[1].strip()},
        current_model_call_ids=set(),
        current_chunk_ids=set(),
        current_analysis_id=None,
        current_evidence_ids={{eid for d in ({args.split(',')[2].strip()} or []) for eid in d.evidence_ids}} if "{args.split(',')[2].strip()}" != "[]" else set(),
        is_vlm_direct=False
    )"""

content = re.sub(r'    result = _select_current_call_events\(([^)]+)\)', fix_call, content)

with open("tests/test_pipeline_integration.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed remaining test signatures.")
