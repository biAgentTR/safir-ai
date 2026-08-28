import re

with open("tests/test_pipeline_integration.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix test_select_current_call_events_vlm_direct_multiple_events_loss_regression
old_test1 = '''    result = _select_current_call_events(events, 60.0, detected)'''
new_test1 = '''    result = _select_current_call_events(
        events, 
        latest_timestamp=60.0, 
        current_model_call_ids={"call1"}, 
        current_chunk_ids={"chunk1"}, 
        current_analysis_id="analysis1", 
        current_evidence_ids=set(),
        is_vlm_direct=True
    )'''

content = content.replace(old_test1, new_test1)

# Fix test_select_current_call_events_preserves_input_order
old_test2 = '''    result = _select_current_call_events(events, 25.0, detected)'''
new_test2 = '''    result = _select_current_call_events(
        events, 
        latest_timestamp=25.0, 
        current_model_call_ids={"call1"}, 
        current_chunk_ids={"chunk1"}, 
        current_analysis_id="analysis1", 
        current_evidence_ids={"id1", "id2"},
        is_vlm_direct=True
    )'''

content = content.replace(old_test2, new_test2)

with open("tests/test_pipeline_integration.py", "w", encoding="utf-8") as f:
    f.write(content)
