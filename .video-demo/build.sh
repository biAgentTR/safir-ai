#!/bin/bash
# SAFİR 60 sn demo videosu — sahne sahne render + birleştirme.
# Bkz. STORYBOARD.md. Tüm görseller gerçek üründen, klipler gerçek saha kayıtlarından.
set -e

UI=".video-demo/assets/ui"
CL=".video-demo/assets/clips"
B=".video-demo/build"
FONT="C\\:/Windows/Fonts/segoeuib.ttf"
mkdir -p "$B"

# Altyazı çizimi: alt bantta koyu şerit + beyaz metin.
# Metin DOSYADAN okunur (textfile): boylece ':' ve ',' gibi karakterler filtre
# sozdizimini bozmaz ve Turkce karakterler dogrudan kullanilabilir.
SUBN=0
sub() {  # $1 = metin -> filtre parcasi (stdout)
  SUBN=$((SUBN+1))
  local f="$B/sub_${SUBN}.txt"
  printf '%s' "$1" > "$f"
  echo "drawbox=x=0:y=906:w=1920:h=116:color=0x0B0F14@0.82:t=fill,drawtext=fontfile='${FONT}':textfile='${f}':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=944"
}

# Ken Burns: PNG'den 16:9 bölge kırp, yükselt, süre boyunca yumuşak zoom.
# $1=girdi $2=cx $3=cy $4=cw $5=ch $6=süre(sn) $7=altyazı $8=çıktı $9=zoom yönü(in|out)
scene_img() {
  local dur=$6 zdir=${9:-in}
  local zexpr
  if [ "$zdir" = "in" ]; then
    zexpr="1.0+0.06*on/(${dur}*30)"
  else
    zexpr="1.06-0.06*on/(${dur}*30)"
  fi
  ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$dur" -i "$1" \
    -vf "crop=$4:$5:$2:$3,scale=3840:2160:flags=lanczos,zoompan=z='${zexpr}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30,$(sub "$7"),format=yuv420p" \
    -c:v libx264 -preset medium -crf 18 -r 30 "$8"
}

# Yakin plan: UI'in KUCUK ama KRITIK bir bolgesini kirpip buyuterek 16:9 tuvale
# ortalar (panel dikey/ince oldugunda 16:9 crop alakasiz alanlari kadraja aliyordu).
# $1=girdi $2=cx $3=cy $4=cw $5=ch $6=hedef_genislik $7=sure $8=altyazi $9=cikti
scene_pad() {
  local dur=$7
  ffmpeg -y -loglevel error -loop 1 -framerate 30 -t "$dur" -i "$1" \
    -filter_complex "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,gblur=sigma=26,eq=brightness=-0.12[bg];\
                     [0:v]crop=$4:$5:$2:$3,scale=$6:-2:flags=lanczos,pad=iw+8:ih+8:4:4:color=0x1E2A33[fg];\
                     [bg][fg]overlay=(W-w)/2:(H-h)/2,zoompan=z='1.0+0.04*on/(${dur}*30)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30,$(sub "$8"),format=yuv420p" \
    -c:v libx264 -preset medium -crf 18 -r 30 "$9"
}

# Video klip sahnesi: ölçekle/kırp, süre kes, altyazı bas.
# $1=girdi $2=başlangıç $3=süre $4=altyazı $5=çıktı
scene_clip() {
  ffmpeg -y -loglevel error -ss "$2" -t "$3" -i "$1" \
    -vf "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,$(sub "$4"),format=yuv420p" \
    -an -c:v libx264 -preset medium -crf 18 -r 30 "$5"
}

echo "[1/11] HOOK"
scene_img "$UI/09_home.png" 14 2 1892 1064 5.5 \
  "Saha kamerası görüntüsünden risk kararına — saniyeler içinde." "$B/s01.mp4" in

echo "[2/11] saha kaydi"
scene_clip "$CL/saha_yangin.mp4" 0 4.5 "Gerçek saha kaydı. O anda kimse izlemiyor olabilir." "$B/s02.mp4"

echo "[3/11] video paneli"
scene_img "$UI/02_dashboard_clean.png" 345 425 1140 641 4.5 \
  "Video doğrudan görsel-dil modeline (EVREN) gider." "$B/s03.mp4" in

echo "[4/11] ikinci saha"
scene_clip "$CL/saha_forklift.mp4" 0 5.5 "Riskli saniyeler zaman çizelgesinde işaretlenir." "$B/s04.mp4"

echo "[5/11] olay tablosu"
scene_pad "$UI/03_events.png" 345 212 1225 205 1720 7 \
  "Nesne değil olay: tür, zaman damgası, güven." "$B/s05.mp4"

echo "[6/11] VLM cikti"
scene_pad "$UI/06_risk_reasoning.png" 1180 843 400 224 1420 7 \
  "Olayın başlangıcı, gelişimi ve sonucu okunur." "$B/s06.mp4"

echo "[7/11] risk kirilimi"
scene_pad "$UI/06_risk_reasoning.png" 345 262 1225 112 1760 7 \
  "Risk skoru iki bağımsız kaynaktan: kural motoru + ajan." "$B/s07.mp4"

echo "[8/11] gerekce"
scene_pad "$UI/06_risk_reasoning.png" 345 345 1225 100 1780 6 \
  "Her skorun gerekçesi ve mevzuat dayanağı görünür." "$B/s08.mp4"

echo "[9/11] operator aksiyonu"
scene_img "$UI/04_ask_answer.png" 1150 60 770 433 5.5 \
  "Operatöre uygulanabilir aksiyon önerisi." "$B/s09.mp4" in

echo "[10/11] KPI"
scene_pad "$UI/08_kpi.png" 12 556 345 482 590 4.5 \
  "Ölçümleme paneli: her metrik gerçek veriden." "$B/s10.mp4"

echo "[11/11] payoff"
scene_img "$UI/09_home.png" 14 2 1892 1064 6.5 \
  "SAFİR — Görüntüyü izlemeyin. Ne olduğunu anlayın." "$B/s11.mp4" out

echo "[birlestir] xfade zinciri"
python .video-demo/concat.py

echo "BITTI: .video-demo/build/safir_demo_60s.mp4"
