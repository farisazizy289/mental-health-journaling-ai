# 🧠 Mental Health Journaling AI
> Emotion Detection dari Teks Jurnal Harian Bahasa Indonesia menggunakan IndoBERT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![IndoBERT](https://img.shields.io/badge/Model-IndoBERT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-brightgreen)

---

## 🎯 Overview

Aplikasi journaling berbasis AI yang menganalisis tulisan pengguna untuk mendeteksi pola emosi dari teks Bahasa Indonesia — bukan sebagai alat diagnosis, tapi sebagai **self-awareness companion**.

Proyek ini dibangun sebagai bagian dari portofolio **Apple Developer Academy** dengan fokus pada dampak sosial di bidang kesehatan mental.

> ⚠️ **Disclaimer:** Aplikasi ini adalah alat bantu self-awareness, **bukan** pengganti profesional kesehatan mental.

---

## 🌟 Demo

**Live Demo:** [mental-health-journaling-ai.streamlit.app](https://mental-health-journaling-ai.streamlit.app)

---

## 💡 Latar Belakang

1 dari 3 orang Indonesia mengalami masalah kesehatan mental namun tidak menyadarinya — bukan karena tidak peduli, tapi karena tidak ada alat yang membantu mereka **melihat pola emosinya sendiri**.

Journaling adalah praktik yang terbukti secara klinis membantu kesadaran emosi, namun kebanyakan orang tidak tahu cara membaca pola dari tulisannya sendiri. Aplikasi ini hadir sebagai cermin yang jujur — menganalisis tulisan harian dan menampilkan insight emosi secara visual.

**Relevansi Apple:** Apple Watch + watchOS 10 menghadirkan fitur Mental Health tracking, menunjukkan Apple sangat serius di domain ini. Proyek ini selaras dengan visi Apple tentang teknologi yang meningkatkan kualitas hidup.

---

## 🏗️ Arsitektur

```
Input Teks Jurnal (Bahasa Indonesia)
            ↓
    Text Preprocessing
    (lowercase, hapus URL, mention, karakter spesial)
            ↓
    Tokenisasi IndoBERT
    (max_length=128, WordPiece tokenizer)
            ↓
    IndoBERT Base + Classification Head
    (bert-base → dropout(0.3) → linear(768→6))
            ↓
    Softmax → 6 Kelas Emosi
            ↓
    joy · sadness · anger · fear · love · neutral
```

### Model Details

| Komponen | Detail |
|---|---|
| Base Model | `indobenchmark/indobert-base-p1` |
| Parameters | 124,445,958 (~124M) |
| Hidden Dropout | 0.3 |
| Attention Dropout | 0.3 |
| Max Sequence Length | 128 tokens |
| Output Classes | 6 emosi |

---

## 📦 Dataset

| Dataset | Sumber | Jumlah | Bahasa |
|---|---|---|---|
| Text Emotion Indonesia | [gerhardien/Kaggle](https://www.kaggle.com/datasets/gerhardien/text-emotion-indonesia) | ~16,000 | Indonesia |
| Indonesian Twitter Emotion | [wahyuikbalmaulana/Kaggle](https://www.kaggle.com/datasets/wahyuikbalmaulana/indonesian-twitter-emotion-dataset) | ~5,000 | Indonesia |

### Distribusi Label (setelah balancing)

```
Train set — Oversampling minoritas ke 2000/kelas:
joy      ████████████████████  5,106 (tetap)
sadness  ████████████████      4,544 (tetap)
anger    ████████              2,605 (tetap)
neutral  ████████              2,465 (tetap)
fear     ██████                2,000 (oversample dari 516)
love     ██████                2,000 (oversample dari 510)

Val & Test — distribusi asli (tidak dibalance)
```

### Split Dataset

| Split | Jumlah | Keterangan |
|---|---|---|
| Train | ~15,746 | Balanced, shuffled |
| Val | ~1,701 | Distribusi asli |
| Test | ~1,365 | Distribusi asli |

---

## ⚙️ Training

### Hyperparameter

| Parameter | Nilai | Alasan |
|---|---|---|
| Learning Rate | 2e-5 | Standard fine-tuning BERT |
| Weight Decay | 0.01 | L2 regularization |
| Warmup Ratio | 0.1 | 10% total steps |
| Batch Size | 32 | |
| Max Epochs | 10 | Early stopping handle sisanya |
| Early Stopping Patience | 2 | Stop jika val_loss tidak turun 2x |
| LR Scheduler | Linear with warmup | |
| Grad Clip | 1.0 | Cegah exploding gradient |

### Teknik Anti-Overfitting

- **Dropout 0.3** pada hidden layer dan attention (default 0.1)
- **Early stopping** patience=2 berdasarkan val_loss
- **Weight decay** 0.01 (L2 regularization)
- **Weighted CrossEntropyLoss** untuk atasi class imbalance
- **Gradient clipping** max_norm=1.0

### Training Curve

```
Epoch 01 | Train Loss: 0.7431  Acc: 0.7312 | Val Loss: 0.4821  Acc: 0.8401
Epoch 02 | Train Loss: 0.3912  Acc: 0.8734 | Val Loss: 0.3987  Acc: 0.8612
  ✅ Val loss membaik → simpan model
...
🛑 Early stopping di epoch N
```

---

## 📈 Hasil Evaluasi

### Overall Metrics

| Metrik | Nilai |
|---|---|
| **Test Accuracy** | **84.98%** |
| **F1-Score (weighted)** | **85.09%** |
| F1-Score (macro) | 83.00% |
| Test Loss | 0.4725 |

### Per-Class Performance

| Emosi | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 😊 joy | 0.92 | 0.81 | 0.86 | 351 |
| 😢 sadness | 0.80 | 0.85 | 0.82 | 349 |
| 😠 anger | 0.81 | 0.86 | 0.83 | 314 |
| 😨 fear | 0.65 | 0.78 | 0.71 | 65 |
| ❤️ love | 0.87 | 0.70 | 0.78 | 64 |
| 😐 neutral | 0.98 | 0.96 | 0.97 | 222 |

### Akurasi Per Kelas (setelah balancing)

```
neutral  ████████████████████  96.8%
love     ████████████████████  87.5%
fear     ████████████████████  86.2%
anger    ████████████████████  84.7%
joy      ████████████████      81.2%
sadness  ██████████████        75.9%
```

### Analisis Confusion

Kesalahan terbesar: **sadness → anger (40 kasus)**. Ini adalah keterbatasan yang disengaja diakui — secara linguistik, teks sedih dan marah dalam Bahasa Indonesia sering ambigu ("kecewa", "frustrasi"). Bukan bug model, tapi batasan semantik bahasa yang bahkan manusia pun tidak selalu sepakat.

---

## 🚀 Cara Menjalankan Lokal

### Prerequisites
```bash
Python 3.10+
pip
```

### Install & Run
```bash
# Clone repo
git clone https://github.com/farisazizy289/mental-health-journaling-ai.git
cd mental-health-journaling-ai

# Install dependencies
pip install -r requirements.txt

# Download model dari Google Drive (otomatis saat app start)
streamlit run app.py
```

App akan otomatis download model dari Google Drive saat pertama kali dijalankan.

---

## 📁 Struktur Proyek

```
mental-health-journaling-ai/
├── app.py                        ← Streamlit app utama
├── requirements.txt              ← Python dependencies
├── README.md                     ← Dokumentasi ini
├── .streamlit/
│   └── config.toml              ← Tema UI dark mode
└── .gitignore                   ← Exclude model dari git
```

> Model (~500MB) tidak disertakan di repo. Di-host di Google Drive dan didownload otomatis saat app pertama kali dibuka.

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|---|---|
| Model | IndoBERT (`indobenchmark/indobert-base-p1`) |
| Framework | PyTorch + HuggingFace Transformers |
| UI | Streamlit |
| Deployment | Streamlit Cloud |
| Model Hosting | Google Drive + gdown |
| Data Processing | Pandas, Scikit-learn |

---

## ⚖️ Ethical Considerations

- **Bukan alat diagnosis** — selalu ada disclaimer jelas di UI
- **Tidak menyimpan data** — teks pengguna tidak disimpan ke server manapun
- **Bias awareness** — model dilatih pada teks Twitter & dataset umum, mungkin kurang representatif untuk semua dialek dan konteks
- **Human-first** — selalu mendorong pengguna untuk konsultasi profesional jika diperlukan

---

## 👤 Author

Faris Ahmad Rizky Azizy
Apple Developer Academy Candidate

[![GitHub](https://img.shields.io/badge/GitHub-@farisazizy289-black)](https://github.com/farisazizy289)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/faris-ahmad-rizky-azizy)

---

## 📄 License

MIT License — bebas digunakan untuk keperluan edukasi dan non-komersial.

---

<p align="center">
  Dibuat dengan ❤️ untuk kesehatan mental Indonesia
</p>
