"""Sahneleri xfade zinciriyle tek parça 60 sn'lik MP4'e birleştirir.

`build.sh` her sahneyi ayrı ayrı render eder; burada hepsi 0,4 sn'lik çapraz
geçişlerle birleştirilir. xfade her geçişte toplam süreyi `GECIS` kadar
kısalttığı için nihai süre hesaplanarak raporlanır (hedef: <= 60 sn).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

BUILD = Path(".video-demo/build")
GECIS = 0.4  # saniye
CIKTI = BUILD / "safir_demo_60s.mp4"


def sure(path: Path) -> float:
    """Bir video dosyasinin saniye cinsinden suresini dondurur (ffprobe)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def main() -> None:
    sahneler = sorted(BUILD.glob("s??.mp4"))
    if not sahneler:
        raise SystemExit("sahne bulunamadi - once build.sh calistirilmali")

    sureler = [sure(s) for s in sahneler]
    girdiler: list[str] = []
    for s in sahneler:
        girdiler += ["-i", str(s)]

    # xfade zinciri: her adimda offset = o ana kadarki toplam - gecis payi
    filtreler: list[str] = []
    onceki = "[0:v]"
    birikim = sureler[0]
    for i in range(1, len(sahneler)):
        offset = birikim - GECIS
        etiket = f"[v{i}]"
        filtreler.append(
            f"{onceki}[{i}:v]xfade=transition=fade:duration={GECIS}:offset={offset:.3f}{etiket}"
        )
        onceki = etiket
        birikim += sureler[i] - GECIS

    cmd = ["ffmpeg", "-y", "-loglevel", "error", *girdiler,
           "-filter_complex", ";".join(filtreler),
           "-map", onceki, "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", str(CIKTI)]
    subprocess.run(cmd, check=True)

    print(f"sahne sayisi : {len(sahneler)}")
    print(f"ham toplam   : {sum(sureler):.1f} sn")
    print(f"nihai sure   : {sure(CIKTI):.1f} sn")


if __name__ == "__main__":
    main()
