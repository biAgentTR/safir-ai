"""Merkezi istem (prompt) katmani: tum VLM ve Ajan istemleri tek yerde toplanir.

Istemleri kod icine gomulu string sabitleri olarak dagitmak yerine burada
merkezilestirmek; versiyonlamayi, A/B denemesini ve test edilebilirligi
kolaylastirir (sartname degerlendirme kriteri: "prompt engineering'in etkin
kullanimi"). VLM istemleri gozlem odakli, Ajan istemleri muhakeme/karar
odaklidir.
"""

from src.prompts.agent_prompts import (
    AGENT_SYSTEM_PROMPT,
    build_agent_user_prompt,
)
from src.prompts.event_classifier_prompts import (
    EVENT_CLASSIFIER_SYSTEM_PROMPT,
    build_event_classifier_user_prompt,
)
from src.prompts.vlm_prompts import VLM_OBSERVER_SYSTEM_PROMPT

__all__ = [
    "VLM_OBSERVER_SYSTEM_PROMPT",
    "AGENT_SYSTEM_PROMPT",
    "build_agent_user_prompt",
    "EVENT_CLASSIFIER_SYSTEM_PROMPT",
    "build_event_classifier_user_prompt",
]
