# SAFİR — Sıfırdan Kurulum Rehberi (Linux VM + RTX 5090 32 GB)

Bu rehber, **temiz bir Ubuntu 22.04/24.04 sanal makinesinde** (RTX 5090 / Blackwell,
32 GB VRAM geçirmeli) SAFİR'i tamamen **yerel / offline** çalıştırmayı anlatır.

Mimari:

```
 (aynı GPU, tek makine)
 ┌─────────────────────────────┐        ┌────────────────────┐
 │ vLLM VLM  :8001  (Qwen2.5-VL-7B, FP8)  │◄──HTTP──┤                    │
 │ vLLM LLM  :8003  (Qwen2.5-3B, BF16)    │◄──HTTP──┤  SAFİR API :8000   │◄── Frontend
 └─────────────────────────────┘        │  (FastAPI, CPU)    │   (Nuxt :3000 / Streamlit :8501)
                                         └────────────────────┘
```
- Modeller **vLLM Docker konteynerleri** ile serve edilir (CUDA imaj içinde gelir → host'a CUDA kurmaya gerek yok, sadece NVIDIA sürücüsü).
- SAFİR API salt bir **HTTP istemcisidir**; `vllm` python kütüphanesi gerekmez.
- **API anahtarı yok**, harici servis yok.

---

## 0. Ön koşullar
- Ubuntu 22.04 veya 24.04, `sudo` yetkisi, internet erişimi.
- RTX 5090 makineye geçirilmiş (VM'de `lspci | grep -i nvidia` görünmeli).
- Disk: modeller + cache için **≥ 60 GB boş** (7B-VL ~16 GB, 3B ~6 GB, bge-m3 ~2 GB, imajlar).

---

## 1. NVIDIA sürücüsü (Blackwell ≥ 570)
```bash
sudo apt update
sudo ubuntu-drivers install        # en güncel uygun sürücüyü kurar (5090 için ≥570)
sudo reboot
# yeniden açılınca:
nvidia-smi                         # 5090 + sürücü + "CUDA Version: 12.x" görünmeli
```
> `ubuntu-drivers` uygun sürüm bulmazsa: `sudo apt install nvidia-driver-570-open` (veya daha güncel).

## 2. Docker + NVIDIA Container Toolkit
```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit (GPU'yu konteynerlere açar)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# doğrula (GPU konteynerde görünüyor mu):
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 3. Sistem paketleri (backend için)
```bash
sudo apt install -y git python3.11 python3.11-venv python3.11-dev python3-pip \
                    ffmpeg libgl1 libglib2.0-0 tmux
```
> `ffmpeg` + `libgl1` OpenCV'nin video okuması için gerekir.

## 4. Depoyu klonla
```bash
cd ~
git clone https://github.com/yarengogsu/p3-project.git
cd p3-project
git checkout claude/gemini-api-refactor-r1roi4
```

## 5. Backend Python ortamı (vllm kütüphanesi OLMADAN)
```bash
cd ~/p3-project/safir-ai
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gemini.txt      # = tüm CPU bağımlılıkları, vllm python libi HARİÇ
```
> Neden `requirements-gemini.txt`? Modelleri vLLM Docker serve ettiği için API'nin
> `vllm` python kütüphanesine ihtiyacı yoktur; bu dosya tam da o set (fastapi, opencv,
> langchain-openai, faiss, sentence-transformers…) — `vllm` hariç. `requirements.txt`
> ise `vllm==0.5.4` içerir ve gereksizdir (Blackwell/py3.12'de derlenmesi de zordur).

## 6. vLLM model sunucularını başlat (2 konteyner, aynı GPU)
`tmux` içinde iki pencere aç (uzun süreçler). **İlk çalıştırma modelleri indirir** (~22 GB, birkaç dk–saat).

**Pencere 1 — VLM (Qwen2.5-VL-7B, FP8):**
```bash
docker run --rm --gpus all --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.10.2 \
  --model Qwen/Qwen2.5-VL-7B-Instruct --port 8001 --trust-remote-code \
  --quantization fp8 --dtype bfloat16 \
  --gpu-memory-utilization 0.50 --max-model-len 8192 \
  --limit-mm-per-prompt image=12 --max-num-seqs 2
```

**Pencere 2 — LLM (Qwen2.5-3B, BF16):**
```bash
docker run --rm --gpus all --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.10.2 \
  --model Qwen/Qwen2.5-3B-Instruct --port 8003 --trust-remote-code \
  --dtype bfloat16 --gpu-memory-utilization 0.30 --max-model-len 4096 --max-num-seqs 4
```

Her ikisi de `Application startup complete` / `Uvicorn running on http://0.0.0.0:800X`
yazınca hazırdır. Kontrol:
```bash
curl http://127.0.0.1:8001/v1/models      # VLM listelenmeli
curl http://127.0.0.1:8003/v1/models      # LLM listelenmeli
nvidia-smi                                # iki süreç, toplam ~26 GB VRAM
```
> `--network host` sayesinde konteynerler `127.0.0.1:8001/8003`'te açılır — `config.yaml`'daki
> `vllm_host: 127.0.0.1` ile birebir uyumlu. Tag (`v0.10.2`) `sm_120`/CUDA hatası verirse
> daha güncel bir `vllm/vllm-openai` tag'ine çıkın (Blackwell CUDA 12.8'li build şart).

## 7. SAFİR API'yi başlat (Pencere 3)
```bash
cd ~/p3-project/safir-ai
source .venv/bin/activate
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```
Kontrol (Pencere 4):
```bash
curl http://127.0.0.1:8000/health         # {"status":"ok","system":"SAFIR"}
```
> İlk analizde RAG için `BAAI/bge-m3` (~2 GB) CPU'ya inip yüklenir (bir kez).

## 8. Arayüz — iki seçenek

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
# Node 20+ (nvm ile)
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
> (ekran/WebKit ister); VM'de tarayıcı sürümünü kullanın. Native pencere ancak masaüstü
> ortamı olan bir makinede gerekir.

## 9. Uçtan uca doğrulama
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

## Sorun giderme
- **`nvidia-smi` konteynerde çalışmıyor** → NVIDIA Container Toolkit adımını tekrar et, `sudo systemctl restart docker`.
- **vLLM `sm_120` / CUDA hatası** → imaj tag'ini daha güncel bir Blackwell build'ine çıkar.
- **OOM** → VLM `--gpu-memory-utilization`'ı 0.45'e, `--max-model-len`'i 4096'ya, `--limit-mm-per-prompt image`'ı 6'ya düşür.
- **Model indirme yavaş/gated** → Qwen ve bge-m3 herkese açık, token gerekmez; `HF_HUB_ENABLE_HF_TRANSFER=1` ile hızlanır (`pip install hf_transfer`).
- **GPU yokken denemek** → `configs/config.yaml`'da `app.use_mock_vlm: true` + `use_mock_llm: true`; vLLM konteynerleri gerekmeden tüm pipeline çalışır (cevaplar sahte).

## Kalıcılık / servisleştirme (opsiyonel)
Demo sonrası kalıcı çalışsın istersen vLLM `docker run` yerine `docker compose`
(bkz. `docker-compose.yml`) veya `systemd` servisleri kullanılabilir. `docker compose`
yolunda API ve vLLM ayrı konteynerlerde olduğundan `config.yaml`'daki `vllm_host`
değerlerini servis adlarına (`vllm-vlm` / `vllm-llm`) çevir; tek makinede yukarıdaki
`--network host` + `127.0.0.1` yolu daha basittir.
