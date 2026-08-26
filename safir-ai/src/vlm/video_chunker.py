"""03 - VLM Katmani: EVREN "video cozunurluk zarfi" sorunu icin video parcalama (chunking).

Sorun (EVREN dokumantasyonu): sistem, gonderilen videonun TAMAMINA TEK bir
toplam piksel butcesi uygular - 180 saniyelik 720p bir klip ile 60 saniyelik
ayni cozunurlukteki bir klip AYNI oranda kucultulmez; uzun klip cok daha
agresif kucultulur ve detaylar (baret, kucuk alev/duman baslangici vb.)
kaybolabilir. EVREN dokumantasyonu "klibin kisa parcalara bolunmesini"
onerir.

2026-08-26 (ffmpeg + CUDA/NVENC): parcalama ARTIK OpenCV'nin kare-kare
Python dongusu (`cv2.VideoCapture.read` + `cv2.VideoWriter.write`) yerine,
`ffmpeg` alt-surecini KULLANIR - dogru zaman damgali (`-ss`/`-t`) kesin
kesim + mumkunse GPU-hizlandirmali (`h264_nvenc`) kodlama ile COK daha
hizlidir. Uc kademeli, ACIKCA loglanan bir geri-dusme (fallback) zinciri
vardir - HICBIR kademe sessizce "daha kotu ama farkedilmez" bir sonuc
uretmez:

  1. CUDA/NVENC (`h264_nvenc`) - GPU varsa ve surucu/ffmpeg derlemesi
     GERCEKTEN calisirsa (bkz. `_probe_cuda_encoder` - `ffmpeg -encoders`
     listesinde `h264_nvenc` GORUNMESI yeterli DEGILDIR; bu makinede
     GORULDUGU gibi encoder LISTELENEBILIR ama surucu surumu uyumsuzsa
     GERCEK kodlama YINE basarisiz olur - bu yuzden KUCUK, GERCEK bir
     kodlama denemesiyle dogrulanir, sonuc surec omru boyunca ONBELLEKLENIR).
  2. CPU (`libx264`, `veryfast` on-ayari) - CUDA yoksa/basarisiz olursa;
     hala ffmpeg'in tek-seferlik-tam-yeniden-kodlama'sina gore COK daha
     hizli parcalama saglar.
  3. OpenCV (eski implementasyon, `_split_with_opencv`) - `ffmpeg`/`ffprobe`
     ikili dosyalari PATH'te YOKSA veya ffmpeg beklenmedik sekilde
     basarisiz olursa; bu ortam ffmpeg KURULU OLMAYAN bir makineye
     (ör. bazi masaustu kurulumlari) taşinsa bile ozellik TAMAMEN
     BOZULMAZ.

Cozum (degismedi): `split_video_into_chunks`, bir video dosyasini (yerel,
.mp4) sabit sureli, KRONOLOJIK ve KAYIPSIZ parcalara boler.

Video kisa ise (toplam sure <= `chunk_duration_sec`), TEK bir "chunk"
(orijinal dosyanin kendisi, KOPYALANMADAN) dondurulur - kisa videolarda
davranis/performans HIC degismez.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import cv2

logger = logging.getLogger(__name__)

_FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# ffmpeg alt-surec cagrilarinin (sure-orantili) azami bekleme suresi tabani;
# gercek zaman asimi `max(_FFMPEG_TIMEOUT_FLOOR_SEC, chunk_duration_sec * 3)`
# olarak hesaplanir (uzun parcalarda yeterli pay, kisa parcalarda takilan bir
# surecin sonsuza kadar beklenmemesi icin).
_FFMPEG_TIMEOUT_FLOOR_SEC = 60.0

# `_probe_cuda_encoder`in sonucu SUREC OMRU boyunca ONBELLEKLENIR - her
# `split_video_into_chunks` cagrisinda GERCEK bir GPU kodlama denemesi
# TEKRARLANMAZ (maliyetli, sonuc bir makinede calisma suresi boyunca degismez).
_cuda_encoder_cache: Optional[bool] = None


@dataclass
class VideoChunk:
    """Bolunmus bir video parcasi: dosya yolu + orijinal videodaki zaman araligi."""

    path: str
    start_offset_sec: float
    end_offset_sec: float
    index: int
    is_original: bool = False
    """`True` ise `path`, GECICI bir parca DEGIL - orijinal video dosyasinin
    kendisidir (video zaten `chunk_duration_sec`den kisa oldugu icin bolunmedi).
    Cagiran taraf bu dosyayi ASLA silmemeli (bkz. `cleanup_chunks`)."""
    encoder: str = "none"
    """Bu parcanin nasil uretildigi: "cuda" (h264_nvenc), "cpu" (libx264),
    "opencv" (eski kare-kare geri-dusme) veya "none" (`is_original=True` -
    video hic kodlanmadi/bolunmedi). Operator paneline (bkz. `SafirPipeline.
    run`in "sampler" asamasina eklenen `chunking` ozeti) gozlemlenebilirlik
    icin tasinir."""


def _ffmpeg_available() -> bool:
    """`ffmpeg` VE `ffprobe` ikili dosyalarinin PATH'te bulunup bulunmadigini kontrol eder."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_cuda_encoder() -> bool:
    """`h264_nvenc`in bu makinede GERCEKTEN calisip calismadigini KUCUK bir kodlama denemesiyle dogrular.

    `ffmpeg -encoders` ciktisinda `h264_nvenc`in GORUNMESI, GPU'nun
    kodlama icin KULLANILABILIR oldugu anlamina GELMEZ - NVIDIA surucu
    surumu ffmpeg derlemesinin bekledigi NVENC API surumunden eskiyse
    (bu makinede GOZLEMLENDIGI gibi: "Driver does not support the required
    nvenc API version") encoder LISTELENIR ama her GERCEK kodlama denemesi
    basarisiz olur. Bu fonksiyon, `lavfi` ile uretilen 0.1 saniyelik, 64x64
    boyutunda SENTETIK bir kareyi GERCEKTEN kodlamaya calisarak (disk
    G/C'siz, `-f null`) bu belirsizligi ORTADAN KALDIRIR.

    Returns:
        Kodlama denemesi basariliysa `True`. Sonuc surec omru boyunca
        `_cuda_encoder_cache`de ONBELLEKLENIR.
    """
    global _cuda_encoder_cache
    if _cuda_encoder_cache is not None:
        return _cuda_encoder_cache

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-c:v", "h264_nvenc", "-frames:v", "1", "-f", "null", "-",
            ],
            capture_output=True,
            timeout=15.0,
        )
        _cuda_encoder_cache = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("CUDA/NVENC yetenegi dogrulanamadi (%s); CPU kodlayiciya (libx264) dusuluyor.", exc)
        _cuda_encoder_cache = False

    if not _cuda_encoder_cache:
        logger.info(
            "ffmpeg h264_nvenc (CUDA) bu makinede calismiyor (ör. surucu/ffmpeg surum uyumsuzlugu); "
            "video parcalama CPU kodlayici (libx264) ile devam edecek."
        )
    return _cuda_encoder_cache


def _probe_duration_sec(video_path: str) -> Optional[float]:
    """`ffprobe` ile bir video dosyasinin toplam suresini (saniye) okur; basarisizsa `None` doner."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        if result.returncode != 0:
            return None
        return float(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _ffmpeg_extract_chunk(
    video_path: str, start_sec: float, duration_sec: float, out_path: str, use_cuda: bool
) -> bool:
    """`video_path`in `[start_sec, start_sec+duration_sec)` araligini `out_path`e (ayri bir dosyaya) kodlar.

    `-ss`/`-i` sirasi (girdiden ONCE `-ss`) ffmpeg'in HIZLI (anahtar kareye
    atlayip ardindan tam-hassas ILERI decode eden) arama modunu kullanir -
    bu, TUM parcalar icin videoyu bastan tekrar tekrar OKUMAKTAN cok daha
    hizlidir ve yine de saniye-hassasiyetinde dogru kesim SINIRLARI uretir
    (eski OpenCV implementasyonunun KARE-hassas kesimiyle PRATIKTE ayni
    dogrulukta - EVREN'in zaten saniye duzeyinde zaman damgasi bekledigi
    goz onune alindiginda).

    Args:
        use_cuda: `True` ise `h264_nvenc` (GPU) denenir; `False` ise
            `libx264` (CPU, `veryfast`) kullanilir.

    Returns:
        Kodlama basarili VE cikti dosyasi bos-olmayan sekilde uretildiyse `True`.
    """
    cmd = (
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        + (["-hwaccel", "cuda"] if use_cuda else [])
        + ["-ss", f"{start_sec:.3f}", "-i", video_path, "-t", f"{duration_sec:.3f}"]
        + (["-c:v", "h264_nvenc", "-preset", "p4"] if use_cuda else ["-c:v", "libx264", "-preset", "veryfast"])
        + ["-an", out_path]
    )
    timeout = max(_FFMPEG_TIMEOUT_FLOOR_SEC, duration_sec * 3)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg parca kodlama cagrisi basarisiz (cuda=%s): %s", use_cuda, exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "ffmpeg parca kodlama basarisiz (cuda=%s, kod=%d): %s",
            use_cuda,
            result.returncode,
            result.stderr.decode("utf-8", errors="replace")[-500:],
        )
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def _split_with_ffmpeg(
    video_path: str, chunk_duration_sec: float, total_duration_sec: float, out_dir: str
) -> Optional[List[VideoChunk]]:
    """`ffmpeg` ile (mumkunse CUDA/NVENC) sabit sureli parcalar uretir; herhangi bir parca basarisiz olursa `None` doner (cagiran taraf OpenCV'ye duser)."""
    use_cuda = _probe_cuda_encoder()
    num_chunks = math.ceil(total_duration_sec / chunk_duration_sec)
    chunks: List[VideoChunk] = []

    for i in range(num_chunks):
        start = i * chunk_duration_sec
        duration = min(chunk_duration_sec, total_duration_sec - start)
        chunk_path = os.path.join(out_dir, f"chunk_{i:03d}.mp4")

        ok = _ffmpeg_extract_chunk(video_path, start, duration, chunk_path, use_cuda=use_cuda)
        if not ok and use_cuda:
            logger.warning("ffmpeg NVENC parcasi basarisiz (chunk=%d); bu parca CPU (libx264) ile yeniden deneniyor.", i)
            ok = _ffmpeg_extract_chunk(video_path, start, duration, chunk_path, use_cuda=False)
        if not ok:
            logger.warning(
                "ffmpeg parca %d/%d icin de CPU kodlama basarisiz oldu; TUM video OpenCV-tabanli parcalamaya donduruluyor.",
                i + 1,
                num_chunks,
            )
            return None

        chunks.append(
            VideoChunk(
                path=chunk_path,
                start_offset_sec=start,
                end_offset_sec=start + duration,
                index=i,
                encoder="cuda" if use_cuda else "cpu",
            )
        )

    logger.info(
        "Video %s ffmpeg (%s) ile parcalara bolundu: %.1fs -> %d parca (%.1fs/parca)",
        video_path,
        "CUDA/NVENC" if use_cuda else "CPU/libx264",
        total_duration_sec,
        len(chunks),
        chunk_duration_sec,
    )
    return chunks


def _split_with_opencv(video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None) -> List[VideoChunk]:
    """Eski, kare-kare OpenCV implementasyonu - YALNIZCA `ffmpeg` kullanilamadiginda/basarisiz oldugunda geri-dusme (fallback) olarak kullanilir."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video dosyasi acilamadi: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0.0

        if total_frames <= 0 or total_duration <= chunk_duration_sec:
            return [
                VideoChunk(
                    path=video_path, start_offset_sec=0.0, end_offset_sec=total_duration, index=0, is_original=True
                )
            ]

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_chunk = max(1, int(round(chunk_duration_sec * fps)))
        out_dir = out_dir or tempfile.mkdtemp(prefix="safir_vlm_chunks_")
        os.makedirs(out_dir, exist_ok=True)

        chunks: List[VideoChunk] = []
        writer: Optional[cv2.VideoWriter] = None
        chunk_path = ""
        chunk_start_frame = 0
        chunk_idx = 0
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frames_per_chunk == 0:
                if writer is not None:
                    writer.release()
                    chunks.append(
                        VideoChunk(
                            path=chunk_path,
                            start_offset_sec=chunk_start_frame / fps,
                            end_offset_sec=frame_idx / fps,
                            index=chunk_idx,
                            encoder="opencv",
                        )
                    )
                    chunk_idx += 1
                chunk_start_frame = frame_idx
                chunk_path = os.path.join(out_dir, f"chunk_{chunk_idx:03d}.mp4")
                writer = cv2.VideoWriter(chunk_path, _FOURCC, fps, (width, height))
            assert writer is not None
            writer.write(frame)
            frame_idx += 1

        if writer is not None:
            writer.release()
            chunks.append(
                VideoChunk(
                    path=chunk_path,
                    start_offset_sec=chunk_start_frame / fps,
                    end_offset_sec=frame_idx / fps,
                    index=chunk_idx,
                    encoder="opencv",
                )
            )

        logger.info(
            "Video %s OpenCV (CPU, kare-kare) ile parcalara bolundu: %.1fs -> %d parca (%.1fs/parca)",
            video_path,
            total_duration,
            len(chunks),
            chunk_duration_sec,
        )
        return chunks
    finally:
        cap.release()


def split_video_into_chunks(
    video_path: str, chunk_duration_sec: float, out_dir: Optional[str] = None
) -> List[VideoChunk]:
    """Bir video dosyasini sabit sureli, kronolojik/kayipsiz parcalara boler.

    Once `ffmpeg` (mumkunse CUDA/NVENC ile) dener; `ffmpeg`/`ffprobe`
    PATH'te yoksa veya beklenmedik sekilde basarisiz olursa OpenCV
    implementasyonuna (`_split_with_opencv`) ACIKCA loglayarak DUSER (bkz.
    modul dokustringi).

    Args:
        video_path: Yerel bir `.mp4` dosyasinin yolu.
        chunk_duration_sec: Her parcanin hedef suresi (saniye); 0/negatifse
            bolme YAPILMAZ (video oldugu gibi TEK parca olarak doner).
        out_dir: Uretilen parca dosyalarinin yazilacagi dizin; verilmezse
            gecici bir dizin olusturulur (cagiran taraf `cleanup_chunks` ile
            temizlemelidir).

    Returns:
        Zaman sirali `VideoChunk` listesi. Video, `chunk_duration_sec`den
        KISA veya esitse (veya `chunk_duration_sec<=0`), TEK elemanli bir
        liste doner ve o eleman ORIJINAL dosyayi (`is_original=True`,
        kopyalanmadan) isaret eder.

    Raises:
        ValueError: Video dosyasi hicbir yontemle (ffprobe VE OpenCV) acilamazsa/okunamazsa.
    """
    if chunk_duration_sec <= 0:
        return [VideoChunk(path=video_path, start_offset_sec=0.0, end_offset_sec=0.0, index=0, is_original=True)]

    if not _ffmpeg_available():
        logger.info("ffmpeg/ffprobe PATH'te bulunamadi; video parcalama OpenCV (CPU, kare-kare) ile yapilacak.")
        return _split_with_opencv(video_path, chunk_duration_sec, out_dir)

    total_duration = _probe_duration_sec(video_path)
    if total_duration is None:
        logger.warning("ffprobe video suresini okuyamadi (%s); OpenCV-tabanli parcalamaya donuluyor.", video_path)
        return _split_with_opencv(video_path, chunk_duration_sec, out_dir)

    if total_duration <= chunk_duration_sec:
        return [
            VideoChunk(
                path=video_path, start_offset_sec=0.0, end_offset_sec=total_duration, index=0, is_original=True
            )
        ]

    out_dir = out_dir or tempfile.mkdtemp(prefix="safir_vlm_chunks_")
    os.makedirs(out_dir, exist_ok=True)

    chunks = _split_with_ffmpeg(video_path, chunk_duration_sec, total_duration, out_dir)
    if chunks is None:
        return _split_with_opencv(video_path, chunk_duration_sec, out_dir)
    return chunks


def cleanup_chunks(chunks: List[VideoChunk]) -> None:
    """`split_video_into_chunks`in urettigi GECICI parca dosyalarini siler.

    `is_original=True` olan (orijinal video dosyasini isaret eden) girdiler
    ASLA silinmez - yalnizca bu modulun urettigi gecici `.mp4` dosyalari
    temizlenir. Bir dosya zaten silinmisse/erisilemezse SESSIZCE gecilir
    (temizlik best-effort'tur, pipeline'i KESMEZ).
    """
    for chunk in chunks:
        if chunk.is_original:
            continue
        try:
            os.remove(chunk.path)
        except OSError:
            logger.debug("Gecici video parcasi silinemedi (zaten yok olabilir): %s", chunk.path)
