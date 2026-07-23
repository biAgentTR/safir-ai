"""Pytest kok konfigurasyonu: `src` paketinin CWD'den bagimsiz ice aktarilabilmesini saglar.

`tests/` altindaki dosyalar `from src.xxx import ...` kullanir; bu dosya,
pytest hangi dizinden calistirilirsa calistirilsin proje kokunu (bu dosyanin
bulundugu `safir-ai/` klasoru) `sys.path`'e ekler.
"""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.config_loader import load_config  # noqa: E402


@pytest.fixture(scope="session")
def safir_config():
    """`configs/config.yaml`'dan yuklenmis, tum test modulleri tarafindan paylasilan konfigurasyon."""
    return load_config(ROOT_DIR / "configs" / "config.yaml")
