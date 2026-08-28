import re

with open("src/event_analysis/schemas.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add provenance fields to DetectedEvent
det_insert_point = "    risk_hint: Optional[int] = Field("
det_new_fields = """    # --- C1B Provenance Alanlari ---
    source_analysis_id: Optional[str] = Field(default=None, description="Bu olayin uretildigi ana analiz/job kimligi.")
    source_video_id: Optional[str] = Field(default=None, description="Bu olayin ait oldugu video kimligi.")
    source_chunk_id: Optional[str] = Field(default=None, description="Bu olayin uretildigi chunk kimligi.")
    source_model_call_id: Optional[str] = Field(default=None, description="Bu olayi ureten model cagrisi kimligi.")
    source_observation_id: Optional[str] = Field(default=None, description="Uygulama tarafindan atanan benzersiz ham gozlem kimligi.")
    relative_start_sec: Optional[float] = Field(default=None, description="Chunk icindeki goreceli baslangic zamani (saniye).")
    relative_end_sec: Optional[float] = Field(default=None, description="Chunk icindeki goreceli bitis zamani (saniye).")

    risk_hint: Optional[int] = Field("""

content = content.replace(det_insert_point, det_new_fields)

# Add provenance fields to TemporalEvent
tem_insert_point = "    risk_hint: Optional[int] = Field("
tem_new_fields = """    # --- C1B Provenance Alanlari ---
    source_analysis_ids: List[str] = Field(default_factory=list, description="Bu zamanlanmis olayi olusturan tespitlerin analiz kimlikleri.")
    source_video_ids: List[str] = Field(default_factory=list, description="Bu zamanlanmis olayi olusturan tespitlerin video kimlikleri.")
    source_chunk_ids: List[str] = Field(default_factory=list, description="Bu zamanlanmis olayi olusturan tespitlerin chunk kimlikleri.")
    source_model_call_ids: List[str] = Field(default_factory=list, description="Bu zamanlanmis olayi olusturan tespitlerin model cagri kimlikleri.")
    source_observation_ids: List[str] = Field(default_factory=list, description="Bu zamanlanmis olayi olusturan tespitlerin ham gozlem kimlikleri.")

    risk_hint: Optional[int] = Field("""

content = content.replace(tem_insert_point, tem_new_fields)

with open("src/event_analysis/schemas.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Updated schemas.py with provenance fields.")
