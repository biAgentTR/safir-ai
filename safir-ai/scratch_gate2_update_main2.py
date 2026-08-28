import re

with open("src/main.py", "r", encoding="utf-8") as f:
    content = f.read()

# I didn't successfully replace the old body with the new body! 
# Let's fix the inner function `_belongs_to_current_call` directly.

new_inner = """    def _belongs_to_current_call(te: TemporalEvent) -> bool:
        # Guvenlik Kontrolu: Eger olay baska bir analysis ID'ye aitse KESINLIKLE reddet!
        if te.source_analysis_ids:
            if not current_analysis_id:
                return False
            if current_analysis_id not in te.source_analysis_ids:
                return False

        # 1. Model Call ID kesisimi (Kesin)
        if current_model_call_ids and te.source_model_call_ids:
            if set(te.source_model_call_ids) & current_model_call_ids:
                return True
                
        # 2. Chunk ID kesisimi
        if current_chunk_ids and te.source_chunk_ids:
            if set(te.source_chunk_ids) & current_chunk_ids:
                return True
                
        # 3. Evidence ID kesisimi (Eski kare modu veya fallback)
        if current_evidence_ids and te.evidence_ids:
            if set(te.evidence_ids) & current_evidence_ids:
                return True
                
        # 4. Geriye uyumluluk (Yalnizca opt-in)
        if allow_legacy_timestamp_fallback and not te.source_model_call_ids and not te.source_chunk_ids and not te.evidence_ids:
            import logging
            logging.getLogger(__name__).warning("Legacy timestamp fallback is being used!")
            return abs(te.end_timestamp - latest_timestamp) <= _CURRENT_CALL_TIMESTAMP_TOLERANCE
            
        return False"""

# Find the def _belongs_to_current_call and replace its entire block
content = re.sub(r'    def _belongs_to_current_call\(te: TemporalEvent\) -> bool:.*?return False', new_inner, content, flags=re.DOTALL)

with open("src/main.py", "w", encoding="utf-8") as f:
    f.write(content)
