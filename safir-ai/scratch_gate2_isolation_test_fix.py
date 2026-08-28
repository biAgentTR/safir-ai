import re

with open("tests/test_selector_isolation.py", "r", encoding="utf-8") as f:
    content = f.read()

# Update test_cross_analysis_isolation
old_test_cross = """    # Not: _select_current_call_events mantiginda biz set(chunk_ids) kesistiriyoruz, fakat ayni
    # chunk0 baska bir videonun veya baska bir analizin chunk_id'si olabilir. 
    # Fakat _select_current_call_events'te source_chunk_ids kullaniliyorsa ve her chunk_id GUID 
    # uretmiyorsa bu risklidir. Biz UUID tasidigini varsayiyoruz, ama eger sadece "chunk0" tasirsa,
    # analysis_id uzerinden ekstra filtre gerekir!
    # Fakat _select_current_call_events kodumuz su an source_chunk_ids ve source_model_call_ids'i 
    # baz aliyor. analysis_id kontrolu yapmamistik. Bunu duzeltip koda ekleyelim!
    pass"""

new_test_cross = """    # Gercek mimaride chunk_id "{analysis_id}:chunk:{index:06d}" seklindedir
    result = _select_current_call_events(
        temporal_events=events,
        latest_timestamp=15.0,
        current_model_call_ids={"analysis_current:chunk:000000:vlm"},
        current_chunk_ids={"analysis_current:chunk:000000"},
        current_analysis_id="analysis_current",
        current_evidence_ids=set(),
        is_vlm_direct=True
    )
    
    assert len(result) == 1
    assert result[0].event_name == "Current Analysis Event"
"""

content = content.replace(old_test_cross, new_test_cross)

# Fix the event initialization to use proper names
content = content.replace(
    'source_model_call_ids=["chunk0:vlm"], source_chunk_ids=["chunk0"]\n        ),',
    'source_model_call_ids=["analysis_old:chunk:000000:vlm"], source_chunk_ids=["analysis_old:chunk:000000"]\n        ),'
)
content = content.replace(
    'source_model_call_ids=["chunk0:vlm"], source_chunk_ids=["chunk0"]\n        )',
    'source_model_call_ids=["analysis_current:chunk:000000:vlm"], source_chunk_ids=["analysis_current:chunk:000000"]\n        )'
)

with open("tests/test_selector_isolation.py", "w", encoding="utf-8") as f:
    f.write(content)
