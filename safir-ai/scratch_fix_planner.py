import re

with open("src/vlm/video_chunker.py", "r", encoding="utf-8") as f:
    content = f.read()

old_planner = '''def plan_segments(duration_sec: float, window_sec: float, overlap_sec: float = 0.0) -> List[SegmentPlan]:
    """Videoyu sabit zaman pencerelerine bolmek icin kesim plani uretir.

    Sifir veya pozitif overlap degerlerini destekler. Gercek FFT kesimi (Materialization)
    bu fonksiyondan donen plana gore yapilir.

    Args:
        duration_sec: Videonun gercek suresi (saniye).
        window_sec: Hedef parca uzunlugu (saniye).
        overlap_sec: Pencereler arasi bindirme payi (saniye).

    Returns:
        Uretilen segment planlari listesi.
    """
    if duration_sec <= 0.0 or math.isnan(duration_sec) or math.isinf(duration_sec):
        raise ValueError(f"Gecersiz duration_sec: {duration_sec}")
    if window_sec <= 0.0 or math.isnan(window_sec) or math.isinf(window_sec):
        raise ValueError(f"Gecersiz window_sec: {window_sec}")
    if overlap_sec < 0.0 or overlap_sec >= window_sec or math.isnan(overlap_sec) or math.isinf(overlap_sec):
        raise ValueError(f"Gecersiz overlap_sec: {overlap_sec}")

    step_sec = window_sec - overlap_sec
    plans = []
    start = 0.0
    index = 0

    while start < duration_sec:
        end = min(start + window_sec, duration_sec)
        # Guvenlik agi: Floating point hatalari yuzunden 0 sureli chunk cikmasin
        if end - start <= 1e-4:
            break
            
        plans.append(SegmentPlan(
            index=index,
            start_sec=start,
            end_sec=end,
            overlap_left_sec=overlap_sec if index > 0 else 0.0,
            overlap_right_sec=overlap_sec if end < duration_sec else 0.0
        ))
        
        if end >= duration_sec:
            break
            
        start += step_sec
        index += 1

    return plans'''

new_planner = '''def plan_segments(duration_sec: float, window_sec: float, overlap_sec: float = 0.0) -> List[SegmentPlan]:
    """Videoyu sabit zaman pencerelerine bolmek icin kesim plani uretir.

    Sifir veya pozitif overlap degerlerini destekler. Gercek FFT kesimi (Materialization)
    bu fonksiyondan donen plana gore yapilir.

    Args:
        duration_sec: Videonun gercek suresi (saniye).
        window_sec: Hedef parca uzunlugu (saniye).
        overlap_sec: Pencereler arasi bindirme payi (saniye).

    Returns:
        Uretilen segment planlari listesi.
    """
    if duration_sec <= 0.0 or math.isnan(duration_sec) or math.isinf(duration_sec):
        raise ValueError(f"Gecersiz duration_sec: {duration_sec}")
    if window_sec <= 0.0 or math.isnan(window_sec) or math.isinf(window_sec):
        raise ValueError(f"Gecersiz window_sec: {window_sec}")
    if overlap_sec < 0.0 or overlap_sec >= window_sec or math.isnan(overlap_sec) or math.isinf(overlap_sec):
        raise ValueError(f"Gecersiz overlap_sec: {overlap_sec}")

    step_sec = window_sec - overlap_sec
    plans = []
    index = 0

    while True:
        # Kümülatif floating-point sapmalarini onlemek icin index uzerinden hesapla
        start = index * step_sec
        if start >= duration_sec:
            break
            
        end = min(start + window_sec, duration_sec)
        
        # Matematiksel olarak hic ilerleme yoksa sonsuz donguyu kir
        if end <= start:
            break
            
        # Gercek overlap hesaplamasi (komsu segmentlerle fiziksel kesisim)
        if index > 0:
            prev_start = (index - 1) * step_sec
            prev_end = min(prev_start + window_sec, duration_sec)
            overlap_left_sec = max(0.0, min(end, prev_end) - max(start, prev_start))
        else:
            overlap_left_sec = 0.0
            
        next_start = (index + 1) * step_sec
        if next_start < duration_sec:
            next_end = min(next_start + window_sec, duration_sec)
            overlap_right_sec = max(0.0, min(end, next_end) - max(start, next_start))
        else:
            overlap_right_sec = 0.0
            
        plans.append(SegmentPlan(
            index=index,
            start_sec=start,
            end_sec=end,
            overlap_left_sec=overlap_left_sec,
            overlap_right_sec=overlap_right_sec
        ))
        
        index += 1

    return plans'''

if old_planner in content:
    content = content.replace(old_planner, new_planner)
    with open("src/vlm/video_chunker.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully replaced plan_segments.")
else:
    print("Could not find old plan_segments in video_chunker.py")
