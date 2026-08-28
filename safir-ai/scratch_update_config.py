import re

with open("src/utils/config_loader.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add chunk_overlap_sec to VLLMEndpointConfig
old_endpoint_chunk = """    chunk_duration_sec: Optional[float] = None
    \"\"\"2026-08-25 (EVREN "video cozunurluk zarfi" duzeltmesi):"""

new_endpoint_chunk = """    chunk_duration_sec: Optional[float] = None
    chunk_overlap_sec: float = 0.0
    \"\"\"2026-08-25 (EVREN "video cozunurluk zarfi" duzeltmesi):"""

content = content.replace(old_endpoint_chunk, new_endpoint_chunk)

# Update active_endpoint
old_active = """    def active_endpoint(self) -> VLLMEndpointConfig:
        \"\"\"Config icinde secilen aktif VLM'in baglanti bilgisini dondurur.\"\"\"
        if self.active_model not in self.models:
            raise KeyError(f"Tanimsiz VLM secimi: '{self.active_model}'")
        return self.models[self.active_model]"""

new_active = """    def active_endpoint(self) -> VLLMEndpointConfig:
        \"\"\"Config icinde secilen aktif VLM'in baglanti bilgisini dondurur.\"\"\"
        if self.active_model not in self.models:
            raise KeyError(f"Tanimsiz VLM secimi: '{self.active_model}'")
        endpoint = self.models[self.active_model]
        if self.chunking:
            endpoint.chunk_duration_sec = self.chunking.window_sec
            endpoint.chunk_overlap_sec = self.chunking.overlap_sec
        return endpoint"""

content = content.replace(old_active, new_active)

with open("src/utils/config_loader.py", "w", encoding="utf-8") as f:
    f.write(content)
