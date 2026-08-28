import re

with open("tests/test_vlm_contracts.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix mock decision
old_decision = "decision=MagicMock(),"
new_decision = """decision=MagicMock(summary="", actions=[], risk_score=0.0, risk_status="assessed"),"""

content = content.replace(old_decision, new_decision)

with open("tests/test_vlm_contracts.py", "w", encoding="utf-8") as f:
    f.write(content)
