"""Kok neden Hata #1 duzeltmesi icin birim testleri: `src.main.normalize_video_source`.

Onceki davranis TUM girdileri (mutlak yollar dahil) dosya adina indirgeyip
`data/<dosya_adi>` altina yonlendiriyordu; bu, farkli-ama-ayni-adli dosyalarin
sunucuda CAKISMASINA (yanlislikla eski/alakasiz bir dosyanin okunmasina) yol
aciyordu (bkz. VLM Pipeline Kok Neden Analiz Raporu, Hata #1). Bu testler:
  - Mutlak (POSIX/Windows/UNC) yollarin OLDUGU GIBI korundugunu,
  - Bagil yol / yalnizca dosya adi girdilerde MEVCUT (degismemis) `data/`
    yonlendirme davranisinin korundugunu,
  - RTSP/HTTP(S) canli yayin adreslerinin etkilenmedigini
dogrular. Pipeline/VLM/sampler'a hicbir bagimliligi yoktur (saf fonksiyon testi).
"""

from __future__ import annotations

import pytest

from src.main import normalize_video_source


# --------------------------- mutlak yollar: KORUNMALI ---------------------------


def test_posix_absolute_path_is_preserved():
    assert normalize_video_source("/home/user/videos/fire_test.mp4") == "/home/user/videos/fire_test.mp4"


def test_windows_absolute_path_backslash_is_preserved():
    src = r"C:\Users\GOKSU\Desktop\video1.mp4"
    assert normalize_video_source(src) == src


def test_windows_absolute_path_forward_slash_is_preserved():
    src = "C:/Users/GOKSU/Desktop/video2.mp4"
    assert normalize_video_source(src) == src


def test_unc_path_is_preserved():
    src = r"\\fileserver\share\videos\test.mp4"
    assert normalize_video_source(src) == src


def test_different_absolute_paths_with_same_filename_do_not_collide():
    """Kok neden Hata #1'in dogrudan regresyon testi: ayni DOSYA ADINA sahip
    iki FARKLI mutlak yol, artik ayni `data/<ad>` degerine cakismamali."""
    a = normalize_video_source("/data/videos/video1/test.mp4")
    b = normalize_video_source("/data/videos/video2/test.mp4")
    assert a != b
    assert a == "/data/videos/video1/test.mp4"
    assert b == "/data/videos/video2/test.mp4"


# --------------------------- bagil yol / dosya adi: MEVCUT DAVRANIS ---------------------------


def test_bare_filename_still_routes_to_data_dir():
    assert normalize_video_source("test.mp4") == "data/test.mp4"


def test_relative_path_with_subdir_still_reduces_to_basename_in_data_dir():
    """Mevcut (degismemis) davranis: bagil bir alt-yol verilse bile yalnizca
    dosya adi alinip data/ altina yerlestirilir (Docker bind-mount uyumlulugu)."""
    assert normalize_video_source("videos/test.mp4") == "data/test.mp4"


def test_relative_windows_style_path_still_reduces_to_basename_in_data_dir():
    assert normalize_video_source("videos\\test.mp4") == "data/test.mp4"


def test_whitespace_is_trimmed_for_relative_input():
    assert normalize_video_source("  test.mp4  ") == "data/test.mp4"


# --------------------------- canli yayin: DEGISMEMELI ---------------------------


@pytest.mark.parametrize(
    "uri",
    [
        "rtsp://192.168.1.10:554/stream",
        "http://example.com/live.m3u8",
        "https://example.com/live.m3u8",
        "RTSP://camera.local/stream",  # buyuk/kucuk harf duyarsiz
    ],
)
def test_live_stream_uris_are_untouched(uri):
    assert normalize_video_source(uri) == uri
