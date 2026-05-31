import os
import re
import json
import time
import gdown
import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ═══════════════════════════════════════════════════════════════════
#  KONFIGURASI
# ═══════════════════════════════════════════════════════════════════
MODEL_DIR   = "best_emotion_model"
ASSETS_FILE = "emotion_assets.json"
MAX_LEN     = 128

GDRIVE_MODEL_ZIP_ID = "1v_HGPv-_y9vS0n59aDy3AfCRNzxe9g4H"   # ← ZIP model
GDRIVE_ASSETS_ID    = "12lmZurvtbHVz5IXuRvrxEFb5xjDdFi4M"   # ← emotion_assets.json

EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'love', 'neutral']
EMOTION_EMOJI  = {'joy':'😊','sadness':'😢','anger':'😠','fear':'😨','love':'❤️','neutral':'😐'}
EMOTION_COLORS = {'joy':'#f59e0b','sadness':'#3b82f6','anger':'#ef4444',
                  'fear':'#8b5cf6','love':'#ec4899','neutral':'#6b7280'}
EMOTION_DESC   = {
    'joy':     'Kamu sedang merasakan kebahagiaan dan energi positif.',
    'sadness': 'Kamu tampaknya sedang bersedih. Tidak apa-apa untuk merasakan ini.',
    'anger':   'Ada rasa marah atau frustrasi yang perlu disalurkan dengan sehat.',
    'fear':    'Kamu mungkin sedang cemas atau khawatir akan sesuatu.',
    'love':    'Ada perasaan cinta dan kasih sayang yang kamu rasakan.',
    'neutral': 'Kamu sedang dalam kondisi tenang dan netral.',
}

DEVICE = torch.device('cpu')  

# ═══════════════════════════════════════════════════════════════════
#  DOWNLOAD MODEL DARI GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════
def download_model():
    """Download model dari Google Drive jika belum ada."""
    if os.path.exists(MODEL_DIR) and os.path.exists(ASSETS_FILE):
        return  # sudah ada, skip

    with st.spinner('⏳ Memuat model pertama kali (~500MB), mohon tunggu...'):
        # Download ZIP model
        if not os.path.exists(MODEL_DIR):
            zip_path = "best_emotion_model.zip"
            url = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ZIP_ID}"
            gdown.download(url, zip_path, quiet=False)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(".")
            os.remove(zip_path)

        # Download assets JSON
        if not os.path.exists(ASSETS_FILE):
            url = f"https://drive.google.com/uc?id={GDRIVE_ASSETS_ID}"
            gdown.download(url, ASSETS_FILE, quiet=False)

    st.success('✅ Model berhasil dimuat!')


# ═══════════════════════════════════════════════════════════════════
#  LOAD MODEL (cache agar tidak reload setiap interaksi)
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    download_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    model.to(DEVICE)
    return tokenizer, model


# ═══════════════════════════════════════════════════════════════════
#  FUNGSI PREDIKSI
# ═══════════════════════════════════════════════════════════════════
def preprocess(text):
    text = str(text).lower().strip()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict(text, tokenizer, model, threshold=0.45):
    clean    = preprocess(text)
    encoding = tokenizer(
        clean, max_length=MAX_LEN, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'].to(DEVICE),
            attention_mask=encoding['attention_mask'].to(DEVICE)
        )
    probs   = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    top3    = np.argsort(probs)[::-1][:3]

    return {
        'emotion':      EMOTION_LABELS[top_idx],
        'confidence':   float(probs[top_idx]),
        'is_confident': float(probs[top_idx]) >= threshold,
        'emoji':        EMOTION_EMOJI[EMOTION_LABELS[top_idx]],
        'desc':         EMOTION_DESC[EMOTION_LABELS[top_idx]],
        'color':        EMOTION_COLORS[EMOTION_LABELS[top_idx]],
        'top3':         [(EMOTION_LABELS[i], float(probs[i])) for i in top3],
        'all_probs':    {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))},
    }


# ═══════════════════════════════════════════════════════════════════
#  UI STREAMLIT
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Mental Health Journaling AI",
    page_icon="🧠",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center'>🧠 Mental Health Journaling AI</h1>
<p style='text-align:center;color:gray'>
    Tulis jurnal harianmu — AI akan mendeteksi emosi dari tulisanmu.<br>
    <small>⚠️ Bukan pengganti profesional kesehatan mental.</small>
</p>
<hr>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────
tokenizer, model = load_model()

# ── Input teks ────────────────────────────────────────────────────
st.markdown("### 📝 Tulis Jurnalmu")
text_input = st.text_area(
    label="Ceritakan hari atau perasaanmu hari ini:",
    placeholder="Contoh: Hari ini aku merasa sangat lelah dan kecewa karena pekerjaan menumpuk...",
    height=150,
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    analyze_btn = st.button("🔍 Analisis", use_container_width=True, type="primary")

# ── Hasil prediksi ────────────────────────────────────────────────
if analyze_btn:
    if not text_input.strip():
        st.warning("⚠️ Tulis sesuatu dulu ya!")
    elif len(text_input.strip()) < 10:
        st.warning("⚠️ Teks terlalu pendek, ceritakan lebih banyak.")
    else:
        with st.spinner("Menganalisis emosi..."):
            result = predict(text_input, tokenizer, model)

        st.markdown("---")
        st.markdown("### 🎯 Hasil Deteksi Emosi")

        # ── Emosi utama ───────────────────────────────────────────
        col_emoji, col_info = st.columns([1, 3])
        with col_emoji:
            st.markdown(
                f"<div style='font-size:80px;text-align:center'>{result['emoji']}</div>",
                unsafe_allow_html=True
            )
        with col_info:
            st.markdown(
                f"<h2 style='color:{result['color']};margin-bottom:4px'>"
                f"{result['emotion'].upper()}</h2>",
                unsafe_allow_html=True
            )
            conf_pct = result['confidence'] * 100
            st.progress(result['confidence'])
            st.caption(f"Tingkat keyakinan: **{conf_pct:.1f}%**")
            if not result['is_confident']:
                st.caption("⚠️ Keyakinan rendah — emosi mungkin campuran")

        # ── Deskripsi ─────────────────────────────────────────────
        st.info(f"💬 {result['desc']}")

        # ── Top 3 emosi ───────────────────────────────────────────
        st.markdown("#### 📊 Distribusi Emosi")
        for emotion, prob in result['all_probs'].items():
            col_label, col_bar = st.columns([1, 4])
            with col_label:
                st.markdown(
                    f"{EMOTION_EMOJI[emotion]} **{emotion}**",
                    unsafe_allow_html=True
                )
            with col_bar:
                st.progress(prob, text=f"{prob*100:.1f}%")

        # ── Disclaimer ────────────────────────────────────────────
        st.markdown("---")
        st.caption(
            "🔒 Tulisanmu tidak disimpan. "
            "Hasil ini hanya untuk self-awareness, bukan diagnosis medis. "
            "Jika kamu merasa membutuhkan bantuan, hubungi profesional kesehatan mental."
        )

# ── Sidebar: Contoh jurnal ────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Contoh Jurnal")
    examples = {
        "😊 Joy":     "Hari ini sangat menyenangkan! Aku berhasil menyelesaikan project dan dapat pujian dari dosen.",
        "😢 Sadness": "Aku merasa sangat lelah dan tertekan akhir-akhir ini. Semua pekerjaan terasa berat banget.",
        "😠 Anger":   "Tadi ada yang nyerobot antrean di depanku, rasanya kesal sekali nggak bisa ditahan!",
        "😨 Fear":    "Besok ada presentasi penting dan aku takut banget tidak bisa menjawab pertanyaan penguji.",
        "❤️ Love":    "Aku sangat menyayangi keluargaku dan bersyukur atas semua yang mereka berikan.",
        "😐 Neutral": "Hari ini biasa saja, tidak ada yang spesial terjadi.",
    }
    for label, example in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state['example'] = example

    if 'example' in st.session_state:
        st.markdown("**Teks contoh:**")
        st.code(st.session_state['example'], language=None)

    st.markdown("---")
    st.markdown("### ℹ️ Tentang Model")
    st.markdown("""
    - **Model:** IndoBERT base-p1
    - **Task:** 6-class emotion classification
    - **Bahasa:** Indonesia
    - **Akurasi:** ~85%
    """)
