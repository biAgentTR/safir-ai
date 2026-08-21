"""Ajan Muhakeme Testleri İçin Sabit Mock Senaryolar Veri Seti."""

from __future__ import annotations

from typing import Any, Dict

MOCK_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "high_risk_critical": {
        "name": "Yüksek Riskli Senaryo (Forklift Kazası + Hareketsiz Kişi)",
        "observation": "Sahada forklift yaya geçidinde bir personele çarptı; personel yerde hareketsiz kalıyor ve yaralanma emaresi var.",
        "user_prompt": "Sahnede İSG ve güvenlik ihlali var mı değerlendir.",
        "expected_min_score": 80,
        "expected_level": "kritik",
        "expected_escalation": True,
    },
    "low_risk_routine": {
        "name": "Düşüş Riskli Senaryo (Rutin Çalışma)",
        "observation": "Sahada personel baret ve yansıtıcı yelek ile rutin yürüyüş yapmaktadır. Herhangi bir kaza veya ihlal bulunmamaktadır.",
        "user_prompt": "Saha durumunu değerlendir.",
        "expected_max_score": 25,
        "expected_level": "dusuk",
        "expected_escalation": False,
    },
    "negation_safe": {
        "name": "Negasyonlu Senaryo (Olumsuzlanmış İfadeler)",
        "observation": "Sahada forklift hareketi gözlemleniyor ancak herhangi bir ihlal veya kaza görülmedi. Personelde baret eksikliği tespit edilmedi.",
        "user_prompt": "Saha durumunu kontrol et.",
        "expected_score": 0,
        "expected_level": "dusuk",
        "expected_escalation": False,
    },
    "medium_risk_uncertain": {
        "name": "Orta/Belirsiz Riskli Senaryo (Sıvı Döküntüsü)",
        "observation": "Sahada zemin üzerinde yağ döküntüsü ve kontrolsüz koli birikintisi gözlemlendi. Kayma, düşme ve takılma riski var.",
        "user_prompt": "Saha risk analizi yap.",
        "expected_min_score": 50,
        "expected_escalation": True,
        "expected_category": "safety",
    },
    "security_unauthorized_access": {
        "name": "Güvenlik Senaryosu 1 (İzinsiz Alan Girişi / Tel Örgü Sızması)",
        "observation": "Tesis çevre koruma hattında tel örgü sızması ve izinsiz alan girişi tespit edildi; yetkisiz bir şahıs nizamiye arkasına geçiyor.",
        "user_prompt": "Çevre ve tesis güvenlik durumunu incele.",
        "expected_min_score": 80,
        "expected_level": "kritik",
        "expected_category": "security",
        "expected_escalation": True,
    },
    "security_suspicious_object": {
        "name": "Güvenlik Senaryosu 2 (Terk Edilmiş Şüpheli Çanta/Paket)",
        "observation": "Tesis giriş koridorunda 15 dakikadır sahipsiz bırakılmış şüpheli çanta ve koli tespit edildi.",
        "user_prompt": "Tesis iç güvenlik denetimi yap.",
        "expected_min_score": 60,
        "expected_category": "security",
        "expected_escalation": True,
    },
    "security_negation_safe": {
        "name": "Güvenlik Negasyon Senaryosu (Olumsuzlanmış Sızma)",
        "observation": "Nizamiye ve çevre duvarları tarandı; herhangi bir yetkisiz giriş veya tel örgü sızması tespit edilmedi.",
        "user_prompt": "Güvenlik duvarını denetle.",
        "expected_score": 0,
        "expected_category": "security",
        "expected_escalation": False,
    },
    "ambiguous_dual_risk": {
        "name": "Belirsiz / İkili Risk Senaryosu (Yetkisiz Alanda Giriş + Yerde Hareketsiz Kişi)",
        "observation": "Nizamiye arkasındaki yetkisiz/yasaklı alana kart okutmadan giren bir şahıs dengesini kaybedip düşüyor ve yerde hareketsiz kalıyor.",
        "user_prompt": "Saha risk durumunu incele.",
        "expected_min_score": 80,
        "expected_category": "ambiguous",
        "expected_escalation": True,
    },
}
