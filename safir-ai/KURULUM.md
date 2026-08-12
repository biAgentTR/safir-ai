# SAFİR — Sıfırdan Kurulum Rehberi (Native VM + RTX 5090, Docker'sız)

Bu rehber, **temiz bir Ubuntu 22.04/24.04 sanal makinesinde** (RTX 5090 / Blackwell,
32 GB VRAM geçirmeli) SAFİR'i **tamamen native (Docker'sız), yerel/offline**
çalıştırmayı anlatır.

**Deployment modeli:** VM → Python ortamı → `pip install -r requirements.txt` →
`vllm serve` (2 kez) → `uvicorn` → çalışır. **Docker, container veya Compose
GEREKMEZ** — `vllm` paketi kendi uyumlu torch/CUDA wheel'lerini pip ile getirir;
sisteme yalnızca **NVIDIA sürücüsü** yeterlidir.

Mimari:

```
 (aynı GPU, tek makine, tek Python ortamı)
 ┌───────────────────────────────┐        ┌────────────────────┐
 │ vllm serve  :8001  (Qwen2.5-VL-7B, FP8) │◄──HTTP──┤                    │
 │ vllm serve  :8003  (Qwen2.5-3B, BF16)   │◄──HTTP──┤  SAFİR API :8000   │◄── Frontend
 └───────────────────────────────┘        │  (FastAPI, uvicorn) │   (Nuxt :3000 / Streamlit :8501)
                                           └────────────────────┘
```
- `vllm serve`, sistemdeki NVIDIA sürücüsünü doğrudan kullanan bir Python sürecidir (Docker yok).
- SAFİR API salt bir **HTTP istemcisidir** (`httpx`/`openai`/`langchain-openai` ile); model ağırlığını kendisi yüklemez.
- **API anahtarı yok**, harici servis yok.

---

## 0. Ön koşullar
- Ubuntu 22.04 veya 24.04, `sudo` yetkisi, internet erişimi.
- RTX 5090 makineye geçirilmiş (`lspci | grep -i nvidia` görünmeli).
- Disk: modeller + cache için **≥ 60 GB boş** (7B-VL ~16 GB, 3B ~6 GB, bge-m3 ~2 GB).

## 1. NVIDIA sürücüsü (Blackwell ≥ 570)
```bash
sudo apt update
sudo ubuntu-drivers install        # en güncel uygun sürücüyü kurar (5090 için ≥570)
sudo reboot
# yeniden açılınca:
nvidia-smi                         # 5090 + sürücü + "CUDA Version: 12.x" görünmeli
```
> `ubuntu-drivers` uygun sürüm bulmazsa: `sudo apt install nvidia-driver-570-open` (veya daha güncel).
> **Not:** Bu, yalnızca sürücüdür — ayrı bir CUDA toolkit kurulumu (nvcc vb.) GEREKMEZ;
> `pip install vllm` kendi uyumlu CUDA/torch wheel'lerini getirir.

## 2. Sistem paketleri
```bash
sudo apt install -y git python3.12 python3.12-venv python3.12-dev python3-pip \
                    ffmpeg libgl1 libglib2.0-0
```
> `ffmpeg` + `libgl1` OpenCV'nin video okuması için gerekir. (Python 3.12
> kullanın — vLLM'in güncel wheel'leri bu sürüme göre test edilmiştir; farklı
> bir sürüm sisteminizde zaten kuruluysa `python3 --version` ile kontrol edip
> ona göre uyarlayın.)

## 3. Depoyu klonla
```bash
cd ~
git clone https://github.com/yarengogsu/p3-project.git
cd p3-project
git checkout claude/gemini-api-refactor-r1roi4
```

## 4. Kurulum — 3 fazlı, DOĞRULAMA ZORUNLU

**Neden fazlı?** vLLM'in kendi transitive bağımlılıkları (torch/outlines/openai/
fastapi) çok geniştir; bunları projeyle aynı anda, tek adımda kurup bir sorun
çıktığında "proje mi, vLLM mi bozuk" ayrımını yapmak zordur. Bu yüzden ÖNCE
vLLM'i **projeden tamamen izole**, tek başına doğrularız; SONRA proje
bağımlılıklarını ekleriz. Bir faz atlanırsa/eski dosya kullanılırsa sorunlar
sessizce üst üste biner — **her fazın çıktısını gerçekten kontrol edin.**

### Faz 0 — Repo durumunu doğrula (atlamayın)
```bash
cd ~/p3-project
git log -1 --oneline          # en güncel commit'te olmalısınız
grep -E "^vllm|^outlines|^openai" safir-ai/requirements.txt
```
İkinci komut şunu göstermeli (satır başları farklıysa `git pull origin
claude/gemini-api-refactor-r1roi4` çalıştırıp tekrar kontrol edin):
```
vllm>0.7.2
outlines>=0.1.0
openai>=1.35
```

### Faz 1 — vLLM'i İZOLE doğrula (proje kodundan tamamen bağımsız)
```bash
mkdir -p ~/vllm-sanity-check && cd ~/vllm-sanity-check
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install vllm
python -c "import vllm, outlines; print('vllm', vllm.__version__); print('outlines', outlines.__version__)"
vllm --version
deactivate
cd ~ && rm -rf ~/vllm-sanity-check
```
Bu adım **sürüm sabitlemeden** (`pip install vllm`, ekstra kısıtlama yok) çalışır;
VM'nin torch/CUDA/Blackwell ortamıyla doğal olarak uyumlu en güncel vLLM'i
kurar. `vllm --version` temiz bir sürüm numarası basmalı, traceback OLMAMALI.
**Bu adım başarısız olursa proje kurulumuna GEÇMEYİN** — sorun VM'nin CUDA/
sürücü/Python ortamındadır, `requirements.txt`'te değil (bkz. Sorun giderme).

### Faz 2 — Proje ortamı (Faz 1 başarılıysa)
```bash
cd ~/p3-project/safir-ai
rm -rf .venv                  # varsa ONCEKI/kismi kurulumu temizle
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Doğrula (Faz 1'dekiyle aynı/yakın bir vLLM sürümü görmelisiniz — asla eski
`0.5.x` gibi bir şey değil):
```bash
pip show vllm outlines pyairports 2>&1 | grep -E "^Name|^Version|not found"
```
`pyairports` **hiç görünmemeli** ("Package(s) not found"). `vllm`/`outlines`
Faz 1'dekiyle uyumlu (modern) sürümlerde olmalı.

> Kurulum uzun sürer (`vllm` büyük bir paket — torch/CUDA wheel'leri indirir).
> `--no-deps` ile kısmi/manuel kurulum YAPMAYIN (bkz. `requirements.txt`
> başındaki not) — eski bir dosya kopyasıyla birleşirse vLLM'i sessizce eski
> bir sürüme düşürüp kırık bağımlılık zincirine geri döner.

## 5. vLLM model sunucularını başlat (aynı ortam, arka planda)
`.venv` aktifken, **tek terminalde**, arka planda (`&` + `nohup`) başlatılabilir —
ayrı pencere/terminal şart değildir:

```bash
# VLM (Qwen2.5-VL-7B, FP8) — arka planda, log dosyaya
nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --trust-remote-code \
  --quantization fp8 --dtype bfloat16 --gpu-memory-utilization 0.50 \
  --max-model-len 8192 --limit-mm-per-prompt image=12 --max-num-seqs 2 \
  > ~/vlm.log 2>&1 &

sleep 20   # VLM önce yerleşsin (GPU belleğini ölçerken çakışmasın)

# LLM (Qwen2.5-3B, BF16) — arka planda
nohup vllm serve Qwen/Qwen2.5-3B-Instruct --port 8003 --trust-remote-code \
  --dtype bfloat16 --gpu-memory-utilization 0.30 --max-model-len 4096 --max-num-seqs 4 \
  > ~/llm.log 2>&1 &
```

> `--quantization fp8` bayrağı bazı vLLM sürümlerinde adlandırma/parametre
> değiştirebilir; `vllm serve --help | grep -i quant` ile bu VM'nizde kurulu
> sürümde geçerli seçenekleri doğrulayın. Sorun çıkarsa bu bayrağı tamamen
> kaldırıp BF16 ile deneyin (7B ağırlık ~16.6 GB, yine 32 GB'a sığar — bkz.
> aşağıdaki VRAM tablosu).

**İlk çalıştırma modelleri indirir** (~22 GB, birkaç dakika–saat). İzle:
```bash
tail -f ~/vlm.log     # "Uvicorn running on http://0.0.0.0:8001" görünce hazır
```

Kontrol:
```bash
curl http://127.0.0.1:8001/v1/models      # VLM listelenmeli
curl http://127.0.0.1:8003/v1/models      # LLM listelenmeli
nvidia-smi                                # iki süreç, toplam ~26 GB VRAM
```

## 6. SAFİR API'yi başlat
```bash
cd ~/p3-project/safir-ai
source .venv/bin/activate
nohup python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > ~/api.log 2>&1 &
curl http://127.0.0.1:8000/health         # {"status":"ok","system":"SAFIR"}
```
> `configs/config.yaml`'da `vlm.active_model: qwen`, `llm.active_model: qwen3`
> ve `vllm_host: 127.0.0.1` zaten bu native kuruluma göre ayarlıdır — ekstra
> config değişikliği gerekmez.

## 7. Arayüz — iki seçenek

### Seçenek A (en hızlı): Streamlit paneli
```bash
cd ~/p3-project/safir-ai
source .venv/bin/activate
pip install -r requirements-dashboard.txt
streamlit run src/ui/dashboard.py --server.address 0.0.0.0 --server.port 8501
```
Erişim: dizüstünden SSH tüneli
```bash
ssh -L 8501:localhost:8501 kullanici@VM_IP
```
sonra tarayıcıda `http://localhost:8501`.

### Seçenek B (modern desktop UI, tarayıcıda): Nuxt
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc && nvm install 22
cd ~/p3-project/desktop
npm install
npm run dev        # http://localhost:3000 (VM üzerinde)
```
Erişim: dizüstünden SSH tüneli (hem UI hem API portu)
```bash
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 kullanici@VM_IP
```
sonra tarayıcıda `http://localhost:3000`.
> Native Tauri penceresi (`npm run tauri:dev`) başsız (headless) sunucuda **açılmaz**
> (ekran/WebKit ister); VM'de tarayıcı sürümünü kullanın.

## 8. Uçtan uca doğrulama
1. UI'da **New Analysis** → bir video yolu gir (ör. `~/p3-project/safir-ai/data/test.mp4`).
2. **Analizi Başlat** → Workspace'te 7 aşamanın (Sampling→VLM→Events→Context→Decision→Escalation→Report) canlı aktığını gör.
3. **Ask SAFİR** panelinden soru sor; **History** sekmesinde kalıcı kayıtları gör.

Hızlı test videosu (elde yoksa):
```bash
cd ~/p3-project/safir-ai && source .venv/bin/activate
python -c "import cv2,numpy as np; from pathlib import Path; Path('data').mkdir(exist_ok=True); w=cv2.VideoWriter('data/test.mp4',cv2.VideoWriter_fourcc(*'mp4v'),25.0,(160,120)); [w.write(cv2.rectangle(np.full((120,160,3),30,np.uint8),(20,20),(140,100),(210,210,210),-1) if 20<=i<40 else np.full((120,160,3),30,np.uint8)) for i in range(60)]; w.release(); print('data/test.mp4 hazir')"
```

---

## VRAM özeti (RTX 5090 32 GB)
| Bileşen | Ağırlık | vLLM util | ~Ayrılan |
|---|---|---|---|
| VLM 7B (FP8) | ~8.3 GB | 0.50 | ~16 GB |
| LLM 3B (BF16) | ~6.2 GB | 0.30 | ~9.6 GB |
| **Toplam** | ~14.5 GB | 0.80 | **~26 GB → ~6 GB boş** |

Multimodal aktivasyon + KV cache için rahat pay bırakır.

## Süreçleri durdurmak
```bash
pkill -f "vllm serve"
pkill -f "uvicorn src.main:app"
```

## Sorun giderme
- **`pip install -r requirements.txt` bağımlılık çakışması veriyor** → `vllm`'in
  kendi (daha yeni) `torch`/`transformers`/`pydantic` gereksinimleri diğer
  paketlerle çakışabilir. Önce `pip install "vllm>=0.8.5"` tek başına kur, sonra
  `pip install -r requirements.txt` ile kalanını tamamla (pip zaten kurulu
  sürümleri koruyacaktır); ya da `pip install -r requirements.txt --no-deps`
  ardından eksik kalanları tek tek çözün.
- **vLLM `sm_120` / CUDA hatası** → sürücü ≥570 mi kontrol et (`nvidia-smi`);
  `pip install -U vllm` ile en güncel sürüme çık.
- **OOM** → VLM `--gpu-memory-utilization`'ı 0.45'e, `--max-model-len`'i 4096'ya,
  `--limit-mm-per-prompt image`'ı 6'ya düşür.
- **Model indirme yavaş** → `pip install hf_transfer` + `export HF_HUB_ENABLE_HF_TRANSFER=1`.
- **GPU yokken denemek** → `configs/config.yaml`'da `app.use_mock_vlm: true` +
  `use_mock_llm: true`; vLLM süreçleri gerekmeden tüm pipeline çalışır (cevaplar sahte).

## Docker hakkında
Repo'da eski/opsiyonel `Dockerfile`, `Dockerfile.dashboard` ve `docker-compose.yml`
dosyaları bulunur; bunlar **bu native kurulum için gerekli değildir** ve
kullanılmayacaktır. Yukarıdaki adımlar tamamen bağımsızdır — Docker kurmanıza
gerek yoktur.
