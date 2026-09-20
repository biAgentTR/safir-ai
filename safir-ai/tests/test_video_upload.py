"""`POST /uploads/video` (surukle-birak video yukleme) davranis testleri.

NEDEN BU UC NOKTA VAR: analiz istekleri `video_source` alaninda dosyanin
SUNUCUDA ZATEN bulunmasini (`data/<ad>`) veya masaustu kabugunun verdigi
MUTLAK bir yolu bekliyordu. Tarayicida calisan bir operatorun video secmesinin
hicbir yolu YOKTU. Bu testler, yuklemenin calistigini ve iki guvenlik
ozelliginin korundugunu kilitler:

  1. Hedef dosya adi TAMAMEN sunucuda uretilir - istemcinin gonderdigi ad
     yola HIC girmez (yol gecisi ve ad cakismasi onlenir).
  2. Donen `video_source`, mevcut `normalize_video_source` sozlesmesiyle
     uyumludur, yani ek bir cozumleme kuralina GEREK YOKTUR.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.main import app, normalize_video_source

DATA_DIR = Path("data")


@pytest.fixture
def cleanup_uploads():
    """Test sirasinda olusan `upload_*` dosyalarini sonunda siler."""
    before = set(DATA_DIR.glob("upload_*")) if DATA_DIR.exists() else set()
    yield
    if DATA_DIR.exists():
        for path in set(DATA_DIR.glob("upload_*")) - before:
            path.unlink(missing_ok=True)


def _upload(client: TestClient, filename: str, content: bytes):
    return client.post("/uploads/video", files={"file": (filename, content, "video/mp4")})


def test_upload_returns_reference_that_resolves_into_data_dir(cleanup_uploads) -> None:
    """Yukleme basarili olur ve donen referans `data/` altina cozulur."""
    client = TestClient(app)
    resp = _upload(client, "saha.mp4", b"\x00\x01\x02\x03" * 256)

    assert resp.status_code == 200
    body = resp.json()

    assert body["original_filename"] == "saha.mp4"
    assert body["size_bytes"] == 1024
    # Donen deger SUNUCU YOLU degil, yalnizca dosya adidir.
    assert os.sep not in body["video_source"] and "/" not in body["video_source"]
    # ...ve mevcut cozumleme kurali onu dogru yere goturur.
    resolved = Path(normalize_video_source(body["video_source"]))
    assert resolved.exists()
    assert resolved.parent.name == "data"


def test_client_filename_never_reaches_the_stored_path(cleanup_uploads) -> None:
    """Yol gecisi denemesi dosya adina YANSIMAZ (ad sunucuda uretilir)."""
    client = TestClient(app)
    resp = _upload(client, "../../etc/passwd.mp4", b"x" * 64)

    assert resp.status_code == 200
    stored = resp.json()["video_source"]

    assert stored.startswith("upload_") and stored.endswith(".mp4")
    assert ".." not in stored and "passwd" not in stored
    # Ozgun ad yalnizca GOSTERIM icin tasinir.
    assert resp.json()["original_filename"] == "../../etc/passwd.mp4"


def test_same_filename_twice_does_not_overwrite(cleanup_uploads) -> None:
    """Ayni adli iki farkli video birbirini EZMEZ (gecmiste yasanmis hata)."""
    client = TestClient(app)
    first = _upload(client, "kamera.mp4", b"a" * 128).json()["video_source"]
    second = _upload(client, "kamera.mp4", b"b" * 256).json()["video_source"]

    assert first != second
    assert Path(normalize_video_source(first)).read_bytes() == b"a" * 128
    assert Path(normalize_video_source(second)).read_bytes() == b"b" * 256


@pytest.mark.parametrize("filename", ["belge.pdf", "arsiv.zip", "adsiz"])
def test_unsupported_extension_is_rejected(filename: str, cleanup_uploads) -> None:
    """Video olmayan turler 422 ile reddedilir."""
    client = TestClient(app)
    resp = _upload(client, filename, b"x" * 32)

    assert resp.status_code == 422
    assert "Desteklenmeyen video turu" in resp.json()["detail"]


def test_empty_file_is_rejected_and_not_left_on_disk(cleanup_uploads) -> None:
    """Bos dosya 422 verir ve diskte yarim bir dosya BIRAKMAZ."""
    client = TestClient(app)
    before = set(DATA_DIR.glob("upload_*")) if DATA_DIR.exists() else set()

    resp = _upload(client, "bos.mp4", b"")

    assert resp.status_code == 422
    after = set(DATA_DIR.glob("upload_*")) if DATA_DIR.exists() else set()
    assert after == before, "bos yukleme diskte dosya birakmamali"


def test_oversized_upload_is_rejected_without_keeping_a_partial_file(monkeypatch, cleanup_uploads) -> None:
    """Boyut siniri asilirsa 413 doner ve YARIM dosya silinir.

    Sinir testte kucultulur; amac gercekten 512 MB yazmak degil, sinir asimi
    YOLUNUN dogru davrandigini (erken kesme + temizlik) dogrulamaktir.
    """
    import src.main as main

    monkeypatch.setattr(main, "_MAX_VIDEO_UPLOAD_BYTES", 512)
    client = TestClient(app)
    before = set(DATA_DIR.glob("upload_*")) if DATA_DIR.exists() else set()

    resp = _upload(client, "buyuk.mp4", b"x" * 4096)

    assert resp.status_code == 413
    after = set(DATA_DIR.glob("upload_*")) if DATA_DIR.exists() else set()
    assert after == before, "reddedilen yukleme diskte yarim dosya birakmamali"
