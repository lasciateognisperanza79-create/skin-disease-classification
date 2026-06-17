import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
from datetime import datetime
from fpdf import FPDF
import pandas as pd
from collections import Counter

# ---------- Функция потерь ----------
def smooth_sparse_ce(y_true, y_pred, smoothing=0.1):
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), 3)
    y_true_smooth = y_true_oh * (1.0 - smoothing) + smoothing / 3
    return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'skin_model.keras',
        custom_objects={'smooth_sparse_ce': smooth_sparse_ce}
    )

st.set_page_config(page_title="Дерматологический помощник", layout="wide")

# ---------- ДИЗАЙН ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #6c63ff !important;
        text-shadow: 2px 2px 4px rgba(108, 99, 255, 0.2);
        text-align: center;
        font-size: 3rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #6c63ff, #8b83ff) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
        transition: 0.3s;
        border: none !important;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.5);
        background: linear-gradient(135deg, #5a52d5, #7a72f5) !important;
    }
    .stFileUploader {
        background: white;
        border: 3px dashed #6c63ff !important;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
    }
    .stAlert {
        background: #f0f0ff;
        border-left: 8px solid #6c63ff;
        border-radius: 15px;
        padding: 15px;
    }
    .stMetric {
        background: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border: 1px solid #e0e0ff;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 12px;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
        color: #555;
        border-top: 2px solid #6c63ff;
        font-size: 13px;
        z-index: 100;
    }
    .prob-bar {
        background: white;
        border-radius: 20px;
        padding: 10px 15px;
        margin: 8px 0;
        border: 1px solid #e0e0ff;
        box-shadow: 0 2px 10px rgba(108, 99, 255, 0.05);
    }
    .prob-bar .bar {
        height: 28px;
        border-radius: 15px;
        background: linear-gradient(90deg, #6c63ff, #ff6b9d);
        transition: width 0.4s;
        margin-top: 4px;
        color: white;
        font-weight: bold;
        padding: 0 10px;
        line-height: 28px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background: white !important;
        border-radius: 30px !important;
        padding: 10px 25px !important;
        margin: 0 5px !important;
        font-weight: bold;
        color: #333 !important;
        border: 1px solid #ddd !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(135deg, #6c63ff, #8b83ff) !important;
        color: white !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Константы ----------
CLASS_NAMES = ['allergy', 'chickenpox', 'normal']
DISPLAY_NAMES = {'allergy': 'Аллергия', 'chickenpox': 'Ветрянка', 'normal': 'Норма'}
EMOJI = {'allergy': '🌸', 'chickenpox': '🐣', 'normal': '✅'}
INFO = {
    'allergy': '🌿 **Аллергия** — реакция кожи на раздражитель. Рекомендуется избегать контакта с аллергеном, использовать антигистаминные средства.',
    'chickenpox': '🐣 **Ветрянка** — вирусное заболевание. Обратитесь к врачу для подтверждения диагноза.',
    'normal': '✅ **Норма** — кожа в порядке. Продолжайте уход и следите за изменениями.'
}

# Инициализация сессии
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pred_probs' not in st.session_state:
    st.session_state.pred_probs = None
if 'pred_class' not in st.session_state:
    st.session_state.pred_class = None
if 'pred_conf' not in st.session_state:
    st.session_state.pred_conf = None

model = load_model()

def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    return preds

# ---------- БОКОВАЯ ПАНЕЛЬ ----------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/health-heart.png", width=80)
    st.header("💖 О проекте")
    st.markdown("""
    **Учебная работа** по ML  
    **Цель:** классификация кожных заболеваний.
    
    **Модель:** ResNet50  
    **Точность:** 88.75%
    """)
    st.markdown("---")
    st.subheader("📌 Условные обозначения")
    st.markdown("""
    **Классы:**
    - 🌸 Аллергия
    - 🐣 Ветрянка
    - ✅ Норма
    
    **Уверенность:**
    - ✅ ≥ 80% – высокая
    - ⚠️ 50–79% – средняя
    - ❓ < 50% – низкая (диагноз не выводится)
    
    **Уровень тревожности:**
    - 🟢 Низкий (норма с выс. увер.)
    - 🟡 Средний (аллергия с выс. увер.)
    - 🔴 Высокий (ветрянка)
    """)
    st.markdown("---")
    st.caption("👩‍⚕️ **Автор:** Бусыгина Анастасия · 2026")

# ---------- ОСНОВНОЙ КОНТЕНТ ----------
st.title("Дерматологический помощник")
st.caption("Загрузите фото — модель определит класс с уверенностью.")
st.warning("⚠️ Учебный проект. Не заменяет консультацию врача.")

tab1, tab2, tab3 = st.tabs(["🔬 Анализ", "📜 История", "📄 Отчёты"])

# ---------- ВКЛАДКА 1: АНАЛИЗ ----------
with tab1:
    st.markdown("### 📸 Загрузите изображение")
    uploaded_files = st.file_uploader(
        "Выберите одно или несколько фото (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1)
        progress.empty()

        if len(uploaded_files) > 1:
            st.info(f"📸 Загружено {len(uploaded_files)} изображений. Нажмите «Анализировать все».")
            cols = st.columns(min(4, len(uploaded_files)))
            for idx, file in enumerate(uploaded_files[:4]):
                with cols[idx]:
                    img = Image.open(file)
                    st.image(img, width=100, caption=file.name[:12])
            if len(uploaded_files) > 4:
                st.caption(f"и ещё {len(uploaded_files)-4} файла(ов)")

        if st.button("🔍 Анализировать все", use_container_width=True):
            with st.spinner("🤔 Анализируем..."):
                for idx, file in enumerate(uploaded_files):
                    image = Image.open(file)
                    probs = predict(image)
                    idx_pred = np.argmax(probs)
                    cls = CLASS_NAMES[idx_pred]
                    conf = probs[idx_pred] * 100
                    display = DISPLAY_NAMES[cls]

                    # Сохраняем в историю (даже если низкая уверенность)
                    st.session_state.history.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'class': cls if conf >= 50 else 'unknown',
                        'display': display if conf >= 50 else 'Не уверена',
                        'conf': conf,
                        'probs': probs.copy(),
                        'filename': file.name
                    })
                    st.session_state.pred_probs = probs
                    st.session_state.pred_class = cls
                    st.session_state.pred_conf = conf

                    st.markdown(f"**Результат для `{file.name}`:**")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, width=150)
                    with col2:
                        # ---------- ПОРОГ УВЕРЕННОСТИ 50% ----------
                        if conf < 50:
                            st.warning("🤔 Модель не уверена. Попробуйте загрузить другое фото при хорошем освещении.")
                            st.metric("🎯 Уверенность", f"{conf:.1f}% (низкая)")
                        else:
                            if conf >= 80:
                                st.success(f"{EMOJI[cls]} {display} ✅")
                            elif conf >= 50:
                                st.warning(f"{EMOJI[cls]} {display} ⚠️")
                            st.metric("🎯 Уверенность", f"{conf:.1f}%")

                            # Уровень тревожности
                            if cls == 'normal' and conf > 80:
                                st.success("🟢 Низкий уровень тревожности")
                            elif cls == 'allergy' and conf > 70:
                                st.warning("🟡 Средний — рекомендация врача")
                            elif cls == 'chickenpox':
                                st.error("🔴 Высокий — срочная консультация!")

                            st.info(INFO[cls])

                    # Прогресс-бары показываем всегда (чтобы пользователь видел все вероятности)
                    st.markdown("**📊 Вероятности:**")
                    for i, name in enumerate(CLASS_NAMES):
                        p = probs[i] * 100
                        color = "linear-gradient(90deg, #6c63ff, #ff6b9d)" if i == idx_pred else "#e0e0e0"
                        st.markdown(f"""
                        <div class="prob-bar">
                            <span>{EMOJI[name]} {DISPLAY_NAMES[name]}</span>
                            <span style="float:right; font-weight:bold;">{p:.1f}%</span>
                            <div class="bar" style="width:{p:.1f}%; background:{color};"></div>
                        </div>
                        """, unsafe_allow_html=True)

                    if conf < 50:
                        st.caption("💡 Попробуйте фото при хорошем освещении, крупным планом.")
                    st.divider()

                st.success("✅ Анализ завершён!")

    else:
        st.info("👆 Загрузите фото, чтобы начать.")

# ---------- ВКЛАДКА 2: ИСТОРИЯ ----------
with tab2:
    st.markdown("### 📜 История предсказаний")
    if not st.session_state.history:
        st.info("Пока нет записей. Проанализируйте фото на вкладке «Анализ».")
    else:
        st.subheader("📈 Статистика по классам")
        counts = Counter([h['class'] for h in st.session_state.history if h['class'] != 'unknown'])
        cols = st.columns(3)
        for idx, name in enumerate(CLASS_NAMES):
            with cols[idx]:
                st.metric(DISPLAY_NAMES[name], counts.get(name, 0))
        st.caption(f"Всего предсказаний: {len(st.session_state.history)} (из них неопределённых: {sum(1 for h in st.session_state.history if h['class'] == 'unknown')})")

        st.subheader("📋 Детали")
        df = pd.DataFrame(st.session_state.history)
        df_display = df[['time', 'display', 'conf']].rename(columns={'time': 'Время', 'display': 'Класс', 'conf': 'Уверенность, %'})
        st.dataframe(df_display, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать историю (CSV)",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

# ---------- ВКЛАДКА 3: ОТЧЁТЫ ----------
with tab3:
    st.markdown("### 📄 Скачать отчёт (PDF)")
    if st.session_state.pred_class is None:
        st.info("Сначала выполните анализ фото на вкладке «Анализ».")
    else:
        if st.session_state.pred_conf < 50:
            st.warning("Последнее предсказание было неопределённым. Отчёт не генерируется.")
        else:
            st.markdown(f"**Последний диагноз:** {DISPLAY_NAMES[st.session_state.pred_class]}, уверенность: {st.session_state.pred_conf:.1f}%")
            if st.button("📄 Сгенерировать PDF-отчёт"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt="Отчёт о диагностике", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.cell(200, 10, txt=f"Класс: {DISPLAY_NAMES[st.session_state.pred_class]}", ln=True)
                pdf.cell(200, 10, txt=f"Уверенность: {st.session_state.pred_conf:.1f}%", ln=True)
                pdf.ln(5)
                pdf.cell(200, 10, txt="Вероятности по классам:", ln=True)
                for name in CLASS_NAMES:
                    prob = st.session_state.pred_probs[CLASS_NAMES.index(name)] * 100
                    pdf.cell(200, 10, txt=f"{DISPLAY_NAMES[name]}: {prob:.1f}%", ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", size=10)
                pdf.cell(200, 10, txt="Этот отчёт сгенерирован автоматически и не заменяет консультацию врача.", ln=True)
                pdf.cell(200, 10, txt="© 2026 Бусыгина Анастасия", ln=True)
                pdf_output = pdf.output(dest='S').encode('latin-1')
                st.download_button(
                    label="📥 Скачать PDF",
                    data=pdf_output,
                    file_name=f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

# ---------- ФУТЕР ----------
st.markdown("""
<div class="footer">
    👩‍⚕️ Бусыгина Анастасия · ResNet50 · 2026 · Учебный проект
</div>
""", unsafe_allow_html=True)
