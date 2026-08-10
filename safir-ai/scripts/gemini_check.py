"""Gemini (OpenAI-uyumlu uc) teshis scripti.

GEMINI_API_KEY ile: (1) hesabin erisebildigi modelleri listeler, (2) kucuk bir
metin cagrisi, (3) kucuk bir GORSEL (multimodal) cagrisi dener. Boylece
config'teki `model_name` degerinin gecerli olup olmadigi ve VLM/LLM yolunun
calisip calismadigi kesin gorulur.

Kullanim:
    export GEMINI_API_KEY=...          # PowerShell: $env:GEMINI_API_KEY="..."
    python scripts/gemini_check.py
    python scripts/gemini_check.py --model gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import base64
import os
import sys

import httpx

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
# 1x1 seffaf PNG (gorsel yolu testi icin)
_TINY_PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c6360000002000154a24f5f0000000049454e44ae426082"
    )
).decode()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini OpenAI-uyumlu uc teshisi.")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Denenecek model kimligi.")
    args = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("HATA: GEMINI_API_KEY tanimli degil.")
        return 1
    headers = {"Authorization": f"Bearer {key}"}

    print("=" * 64)
    print("1) Erisilebilir modeller (GET /models)")
    print("=" * 64)
    try:
        r = httpx.get(f"{_BASE_URL}/models", headers=headers, timeout=20)
        print("status:", r.status_code)
        if r.status_code == 200:
            ids = [m.get("id", "?") for m in r.json().get("data", [])]
            for mid in ids:
                mark = " <-- config'te kullanilabilir" if "flash" in mid else ""
                print("  -", mid, mark)
        else:
            print(r.text[:400])
    except httpx.HTTPError as exc:
        print("istek hatasi:", exc)

    print("\n" + "=" * 64)
    print(f"2) Metin cagrisi (model={args.model})")
    print("=" * 64)
    text_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Tek kelimeyle yanit ver: merhaba"}],
        "max_tokens": 16,
    }
    _try_chat(headers, text_payload)

    print("\n" + "=" * 64)
    print(f"3) Gorsel (multimodal) cagrisi (model={args.model})")
    print("=" * 64)
    image_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Bu goruntude ne var?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_TINY_PNG}"}},
                ],
            }
        ],
        "max_tokens": 16,
    }
    _try_chat(headers, image_payload)

    print("\nIpucu: 2 ve 3 '200 OK' donuyorsa config'teki model_name dogrudur.")
    print("404 -> model bu hesapta yok (1'deki listeden gecerli birini secin).")
    return 0


def _try_chat(headers: dict, payload: dict) -> None:
    try:
        r = httpx.post(f"{_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30)
        print("status:", r.status_code)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            print("yanit:", (content or "").strip()[:120])
        else:
            print(r.text[:400])
    except httpx.HTTPError as exc:
        print("istek hatasi:", exc)


if __name__ == "__main__":
    sys.exit(main())
