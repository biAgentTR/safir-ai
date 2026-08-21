"""Farklı Karakterde Test Videoları Üretici Betik (generate_test_videos.py).

Görsel hareket profillerine göre 4 farklı kategoride sentetik test videoları
üretir:
1. low_motion_static.mp4    : Düşük hareketli / durağan depo ortamı
2. medium_motion_isg.mp4    : Orta hareketli (aralıklı geçen personel/forklift)
3. high_motion_busy.mp4     : Yüksek hareketli (yoğun saha trafiği)
4. sudden_event_fire.mp4    : Ani olay patlaması (durağan durumdan sonra ani duman/yangın)
"""

from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np


def generate_test_videos(output_dir: str = "data/test_videos"):
    """OpenCV VideoWriter kullanarak 4 farklı test videosu oluşturur."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    width, height = 640, 360
    fps = 30

    configs = [
        ("low_motion_static.mp4", 10, "low"),
        ("medium_motion_isg.mp4", 15, "medium"),
        ("high_motion_busy.mp4", 15, "high"),
        ("sudden_event_fire.mp4", 12, "sudden"),
    ]

    for fname, duration_sec, motion_type in configs:
        file_path = out_path / fname
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))

        total_frames = duration_sec * fps
        bg = np.full((height, width, 3), 40, dtype=np.uint8)
        # Izgara çizgileri ekle (sabit arka plan)
        for y in range(0, height, 40):
            cv2.line(bg, (0, y), (width, y), (60, 60, 60), 1)
        for x in range(0, width, 40):
            cv2.line(bg, (x, 0), (x, height), (60, 60, 60), 1)

        for i in range(total_frames):
            frame = bg.copy()
            # Kamera sensör gürültüsü
            noise = np.random.randint(0, 4, (height, width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)

            if motion_type == "low":
                # Çok hafif, hemen hemen yok denecek kadar küçük sallantı
                pass
            elif motion_type == "medium":
                # Frame 150-300 arasında (5-10. saniyeler) kare içinde yürüyen blok
                if 150 <= i <= 300:
                    x_pos = int(50 + (i - 150) * 3.0)
                    cv2.rectangle(frame, (x_pos, 180), (x_pos + 40, 240), (0, 165, 255), -1)
            elif motion_type == "high":
                # Sürekli 2-3 nesne hareket eder
                x1 = int((i * 4) % width)
                x2 = int(width - ((i * 5) % width))
                cv2.circle(frame, (x1, 100), 20, (255, 100, 100), -1)
                cv2.rectangle(frame, (x2, 220), (x2 + 50, 280), (100, 255, 100), -1)
            elif motion_type == "sudden":
                # 6. saniyeye (frame 180) kadar durağan, sonra ani genişleyen duman/yangın bulutu
                if i >= 180:
                    radius = int(5 + (i - 180) * 1.5)
                    cv2.circle(frame, (320, 180), min(radius, 120), (200, 200, 200), -1)

            writer.write(frame)

        writer.release()
        print(f"[OK] Test videosu üretildi: {file_path} ({duration_sec}s, {total_frames} kare)")


if __name__ == "__main__":
    generate_test_videos()
