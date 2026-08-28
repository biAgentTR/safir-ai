import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# Update build_report to modify chunk_analysis_result when mixed IDs found
old_block = """        if len(current_analysis_ids) > 1:
            logging.getLogger(__name__).error(f"Security/Invariant violation: Multiple analysis IDs detected in a single pipeline call: {current_analysis_ids}")
            # Controlled failure: Drop mixed events to prevent contamination.
            detected_events = []
            current_analysis_ids.clear()"""

new_block = """        if len(current_analysis_ids) > 1:
            logging.getLogger(__name__).error(f"Security/Invariant violation: Multiple analysis IDs detected in a single pipeline call: {current_analysis_ids}")
            # Controlled failure: Drop mixed events to prevent contamination.
            detected_events = []
            current_analysis_ids.clear()
            if hasattr(vlm_response, "chunk_analysis_result") and vlm_response.chunk_analysis_result:
                vlm_response.chunk_analysis_result.analysis_status = "partial" # Using literal 'partial' since Enum might not be imported directly
                vlm_response.chunk_analysis_result.parse_status = "provenance_integrity_failed"
"""

content = content.replace(old_block, new_block)

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)
