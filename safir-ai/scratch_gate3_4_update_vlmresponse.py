import re

with open("src/vlm/base_vlm.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add chunk_analysis_result to VLMResponse
old_vlm_response = """    evidence_ids: List[str] = field(default_factory=list)
    \"\"\"Bu yanitin kapsadigi (gonderilen) `EvidenceFrame.evidence_id` degerleri
    (bkz. `analyze_evidence_batched`); tek-cagri (eski/agrege) yanitlarda bos olabilir.\"\"\"
"""

new_vlm_response = """    evidence_ids: List[str] = field(default_factory=list)
    \"\"\"Bu yanitin kapsadigi (gonderilen) `EvidenceFrame.evidence_id` degerleri
    (bkz. `analyze_evidence_batched`); tek-cagri (eski/agrege) yanitlarda bos olabilir.\"\"\"
    chunk_analysis_result: Optional[Any] = None
    \"\"\"Modelin cikti basarimini gosteren tipli sozlesme (bkz. ChunkAnalysisResult).\"\"\"
"""

content = content.replace(old_vlm_response, new_vlm_response)

with open("src/vlm/base_vlm.py", "w", encoding="utf-8") as f:
    f.write(content)
