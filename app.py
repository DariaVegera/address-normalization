import streamlit as st
import os
import sys
from pathlib import Path

# Добавляем текущую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.normalizers import (
    RuleBasedNormalizer,
    BiLSTMCRFNormalizer, 
    TransformerNormalizer,
    LLMNormalizer
)

# Настройка страницы
st.set_page_config(
    page_title="Address Normalization",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .example-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">🏠 Нормализация российских адресов</h1>', unsafe_allow_html=True)

# Описание
st.markdown("""
Система приводит "грязные" адреса к стандартному формату ГАР (Государственный адресный реестр).
Исправляет опечатки, раскрывает сокращения и восстанавливает правильный порядок адресных элементов.
""")

# Боковая панель с выбором модели
st.sidebar.header("⚙️ Настройки")

model_type = st.sidebar.selectbox(
    "Выберите метод нормализации:",
    ["Rule-Based", "BiLSTM-CRF", "Transformer (T5)", "LLM (Mistral AI)"],
    help="Различные подходы к нормализации адресов"
)

# Информация о модели
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 О выбранной модели")

model_info = {
    "Rule-Based": {
        "description": "Быстрый метод на основе правил и регулярных выражений",
        "speed": "⚡⚡⚡ Очень быстро",
        "accuracy": "⭐⭐ Средняя точность",
        "requirements": "Нет"
    },
    "BiLSTM-CRF": {
        "description": "Нейросетевая модель с CRF слоем для sequence labeling",
        "speed": "⚡⚡ Быстро",
        "accuracy": "⭐⭐⭐ Хорошая точность",
        "requirements": "GPU (опционально)"
    },
    "Transformer (T5)": {
        "description": "Seq2seq трансформер на базе ruT5",
        "speed": "⚡ Умеренно",
        "accuracy": "⭐⭐⭐⭐ Высокая точность",
        "requirements": "GPU (рекомендуется)"
    },
    "LLM (Mistral AI)": {
        "description": "Использует Mistral AI API для нормализации",
        "speed": "⚡ Зависит от API",
        "accuracy": "⭐⭐⭐⭐⭐ Отличная точность",
        "requirements": "API ключ Mistral AI"
    }
}

info = model_info[model_type]
st.sidebar.markdown(f"**Описание:** {info['description']}")
st.sidebar.markdown(f"**Скорость:** {info['speed']}")
st.sidebar.markdown(f"**Точность:** {info['accuracy']}")
st.sidebar.markdown(f"**Требования:** {info['requirements']}")

# API ключ для LLM
api_key = None
if model_type == "LLM (Mistral AI)":
    st.sidebar.markdown("---")
    api_key = st.sidebar.text_input(
        "Mistral API Key:",
        type="password",
        help="Введите ваш API ключ Mistral AI или установите переменную окружения MISTRAL_API_KEY"
    )
    if not api_key:
        api_key = os.getenv("MISTRAL_API_KEY")
        if api_key:
            st.sidebar.success("✅ API ключ загружен из переменной окружения")
        else:
            st.sidebar.warning("⚠️ API ключ не найден")

# Кэширование загрузки моделей
@st.cache_resource
def load_rule_based():
    return RuleBasedNormalizer.load("models/rule_based.pkl")

@st.cache_resource
def load_bilstm():
    return BiLSTMCRFNormalizer.load("models/bilstm_crf_trained.pt")

@st.cache_resource
def load_transformer():
    return TransformerNormalizer.load("models/transformer_t5")

def load_llm(key):
    return LLMNormalizer(api_key=key)

# Основной интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Входной адрес")
    
    # Примеры для быстрого тестирования
    examples = [
        "респ; татарстан; (татарстан),; альметьевский; р-н,; город,; гаражный; массив; территория,; д;",
        "Кужное Мордовский район, улица, Почтовая село, дом Тамбовская область, 27",
        "ВороYнежЯ2кая оLбласть, ВорSонеж гоBрод, АмбIулатlорная улwица, дом 15",
        "мск обл г королев ул калинина д 1",
        "спб василеостровский р-н большой пр 55"
    ]
    
    selected_example = st.selectbox(
        "Выберите пример или введите свой адрес:",
        [""] + examples,
        format_func=lambda x: "Введите свой адрес..." if x == "" else x[:60] + "..."
    )
    
    input_address = st.text_area(
        "Адрес для нормализации:",
        value=selected_example,
        height=100,
        placeholder="Введите адрес, который необходимо нормализовать..."
    )

with col2:
    st.markdown("### ✨ Нормализованный адрес")
    
    result_placeholder = st.empty()
    
    if input_address:
        with st.spinner(f"Обработка с помощью {model_type}..."):
            try:
                # Загрузка и использование модели
                if model_type == "Rule-Based":
                    normalizer = load_rule_based()
                    result = normalizer.predict(input_address)
                    
                elif model_type == "BiLSTM-CRF":
                    normalizer = load_bilstm()
                    result = normalizer.predict(input_address)
                    
                elif model_type == "Transformer (T5)":
                    normalizer = load_transformer()
                    result = normalizer.predict(input_address)
                    
                elif model_type == "LLM (Mistral AI)":
                    if not api_key:
                        st.error("❌ Необходим API ключ Mistral AI")
                        result = None
                    else:
                        normalizer = load_llm(api_key)
                        result = normalizer.predict(input_address)
                
                if result:
                    result_placeholder.markdown(f"""
                    <div class="result-box">
                        <h4 style="margin-top: 0;">Результат:</h4>
                        <p style="font-size: 1.1rem; margin-bottom: 0;">{result}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Ошибка при обработке: {str(e)}")
                st.exception(e)
    else:
        result_placeholder.info("👆 Введите или выберите адрес для нормализации")

# Дополнительная информация
st.markdown("---")
st.markdown("### 📖 Примеры работы системы")

ex_col1, ex_col2, ex_col3 = st.columns(3)

with ex_col1:
    st.markdown("""
    <div class="example-box">
        <h4>Пример 1</h4>
        <b>Входной:</b><br>
        <code>респ татарстан альметьевский р-н город</code><br><br>
        <b>Нормализованный:</b><br>
        <code>Республика Татарстан, Альметьевский район, Альметьевск город</code>
    </div>
    """, unsafe_allow_html=True)

with ex_col2:
    st.markdown("""
    <div class="example-box">
        <h4>Пример 2</h4>
        <b>Входной:</b><br>
        <code>мск обл г королев ул калинина д 1</code><br><br>
        <b>Нормализованный:</b><br>
        <code>Московская область, Королёв город, Калинина улица, дом 1</code>
    </div>
    """, unsafe_allow_html=True)

with ex_col3:
    st.markdown("""
    <div class="example-box">
        <h4>Пример 3</h4>
        <b>Входной:</b><br>
        <code>спб василеостровский р-н большой пр 55</code><br><br>
        <b>Нормализованный:</b><br>
        <code>Санкт-Петербург город, Большой проспект, дом 55</code>
    </div>
    """, unsafe_allow_html=True)

# Футер
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Разработано для нормализации адресов Российской Федерации | 
    <a href="https://github.com/yourusername/address-normalization">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
