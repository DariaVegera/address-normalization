import re
import os
import pickle
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from mistralai import Mistral
from typing import Dict, List


class RuleBasedNormalizer:
    """Универсальный rule-based нормализатор без хардкода"""
    
    def __init__(self):
        self.name = "RuleBased"
        
        # Словарь замены сокращений
        self.replacements = {
            # Регионы
            r'\bресп\.?\b': 'республика',
            r'\bобл\.?\b': 'область',
            r'\bао\b': 'автономный округ',
            r'\bкрай\b': 'край',
            
            # Населенные пункты
            r'\bг\.?\b': 'город',
            r'\bгор\.?\b': 'город',
            r'\bд\.?\b': 'деревня',
            r'\bдер\.?\b': 'деревня',
            r'\bс\.?\b': 'село',
            r'\bсел\.?\b': 'село',
            r'\bп\.?\b': 'поселок',
            r'\bпос\.?\b': 'поселок',
            
            # Улицы
            r'\bул\.?\b': 'улица',
            r'\bпр\.?\b': 'проспект',
            r'\bпр-т\b': 'проспект',
            r'\bпер\.?\b': 'переулок',
            r'\bб-р\b': 'бульвар',
            r'\bбульв\.?\b': 'бульвар',
            r'\bал\.?\b': 'аллея',
            r'\bш\.?\b': 'шоссе',
            
            # Дома
            r'\bд\.?\b': 'дом',
            r'\bк\.?\b': 'корпус',
            r'\bкорп\.?\b': 'корпус',
            r'\bстр\.?\b': 'строение',
            r'\bкв\.?\b': 'квартира',
        }
        
        # Паттерны для чисел
        self.number_patterns = [
            (r'(\d+)\s*[/\\]\s*(\d+)', r'\1/\2'),
            (r'(\d+)[кkK]\s*(\d+)', r'\1 корпус \2'),
            (r'(\d+)[сcC]\s*(\d+)', r'\1 строение \2'),
            (r'(\d+)[аaA]', r'\1а'),
            (r'(\d+)[бbB6]', r'\1б'),
            (r'дом\s*(\d+)\s*к\s*(\d+)', r'дом \1 корпус \2'),
            (r'(\d+)\s+(\d+)\s*([а-я])', r'\1/\2\3'),
        ]
        
        # Латинские буквы для замены
        self.latin_to_cyrillic = str.maketrans(
            "ABCEHKMOPTXabcehopcxy",
            "АВСЕНКМОРТХабсеносрсуу"
        )
        
        # Типы адресных объектов
        self.address_types = {
            'республика', 'область', 'край', 'автономный округ',
            'район', 'город', 'деревня', 'село', 'поселок',
            'улица', 'проспект', 'переулок', 'бульвар', 'аллея', 'шоссе',
            'дом', 'корпус', 'строение', 'квартира'
        }
    
    def _clean_text(self, text: str) -> str:
        """Базовая очистка текста"""
        if not text:
            return ""
        
        # Заменяем латинские буквы
        text = text.translate(self.latin_to_cyrillic)
        
        # Приводим к нижнему регистру
        text = text.lower()
        
        # Заменяем разделители
        text = text.replace(';', ',')
        text = re.sub(r'[^\w\s\-\/\(\)\.,]', ' ', text)
        
        # Убираем цифры в начале слов
        text = re.sub(r'\b\d+(\w+)\b', r'\1', text)
        
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _apply_replacements(self, text: str) -> str:
        """Заменяем сокращения"""
        for pattern, replacement in self.replacements.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Нормализуем номера"""
        for pattern, replacement in self.number_patterns:
            text = re.sub(pattern, replacement, text)
        return text
    
    def _fix_word_order(self, text: str) -> str:
        """Исправляем порядок слов: название перед типом"""
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            current = words[i]
            
            if current in self.address_types:
                if i > 0 and words[i-1] not in self.address_types:
                    if not re.match(r'^\d+', words[i-1]):
                        if result and result[-1] == words[i-1]:
                            result.pop()
                        result.append(words[i-1])
                        result.append(current)
                    else:
                        result.append(current)
                        result.append(words[i-1])
                else:
                    result.append(current)
            elif i == 0 or (i > 0 and words[i-1] not in self.address_types):
                if i + 1 < len(words) and words[i+1] in self.address_types:
                    pass
                elif re.match(r'^\d+', current):
                    if result and result[-1] == 'дом':
                        result.append(current)
                    else:
                        result.append('дом')
                        result.append(current)
                elif current not in self.address_types:
                    result.append(current)
            
            i += 1
        
        return ' '.join(result)
    
    def _capitalize_smart(self, text: str) -> str:
        """Умная капитализация"""
        parts = [p.strip() for p in text.split(',') if p.strip()]
        result_parts = []
        
        for part in parts:
            words = part.split()
            capitalized = []
            
            for i, word in enumerate(words):
                if word in self.address_types and i > 0:
                    capitalized.append(word)
                else:
                    if '(' in word and ')' in word:
                        before, inside = word.split('(', 1)
                        if ')' in inside:
                            inside, after = inside.split(')', 1)
                            capitalized.append(f"{before.capitalize()}({inside.capitalize()}){after}")
                        else:
                            capitalized.append(word.capitalize())
                    else:
                        capitalized.append(word.capitalize())
            
            result_parts.append(' '.join(capitalized))
        
        return ', '.join(result_parts)
    
    def _restore_structure(self, text: str) -> str:
        """Восстанавливаем структуру адреса"""
        words = text.split()
        components = []
        current_name = []
        
        for word in words:
            if word in self.address_types:
                if current_name:
                    components.append(f"{' '.join(current_name)} {word}")
                    current_name = []
                else:
                    components.append(word)
            elif re.match(r'^\d+', word):
                if components and components[-1] == 'дом':
                    components.append(word)
                else:
                    components.append('дом')
                    components.append(word)
            else:
                current_name.append(word)
        
        if current_name:
            components.append(' '.join(current_name))
        
        return ', '.join(components)
    
    def predict(self, address: str) -> str:
        """Основной метод предсказания"""
        if not address or not isinstance(address, str):
            return ""
        
        try:
            text = self._clean_text(address)
            text = self._apply_replacements(text)
            text = self._normalize_numbers(text)
            text = self._fix_word_order(text)
            text = self._restore_structure(text)
            text = self._capitalize_smart(text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r',\s*,', ',', text)
            
            return text
            
        except Exception as e:
            print(f"Ошибка в RuleBasedNormalizer: {e}")
            return address
    
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


class BiLSTM_CRF(nn.Module):
    """BiLSTM-CRF модель для нормализации адресов"""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.crf = CRF(vocab_size, batch_first=True)
    
    def forward(self, x, tags=None):
        mask = x != self.pad_idx
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss
        else:
            preds = self.crf.decode(emissions, mask=mask)
            return preds


class BiLSTMCRFNormalizer:
    """Нормализатор на основе BiLSTM-CRF"""
    
    def __init__(
        self,
        vocab_size=256,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        max_len=128
    ):
        self.name = "BiLSTM-CRF"
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2char = {0: "<PAD>", 1: "<UNK>"}
        
        self.model = BiLSTM_CRF(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout,
            pad_idx=0
        ).to(self.device)
    
    def predict(self, address: str) -> str:
        """Предсказание для одного адреса"""
        self.model.eval()
        
        encoded = [
            self.char2idx.get(c, self.char2idx["<UNK>"])
            for c in address[:self.max_len]
        ]
        
        pad_len = self.max_len - len(encoded)
        encoded += [self.char2idx["<PAD>"]] * pad_len
        
        x = torch.tensor(encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_ids = self.model(x)[0]
        
        chars = [
            self.idx2char[i]
            for i in pred_ids
            if i != self.char2idx["<PAD>"]
        ]
        
        return "".join(chars)
    
    @classmethod
    def load(cls, path: str):
        """Загрузка модели"""
        data = torch.load(path, map_location="cpu")
        obj = cls()
        obj.char2idx = data["char2idx"]
        obj.idx2char = data["idx2char"]
        obj.model.load_state_dict(data["model_state"])
        obj.model.to(obj.device)
        return obj


class TransformerNormalizer:
    """Нормализатор на основе T5"""
    
    def __init__(self, model_name: str = "cointegrated/rut5-base"):
        self.name = "Transformer-T5"
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        self.max_length = 128
        self.prefix = "normalize address: "
    
    def predict(self, address: str) -> str:
        """Предсказание для одного адреса"""
        self.model.eval()
        
        input_text = self.prefix + address
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=3,
                early_stopping=True,
                repetition_penalty=2.0
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @classmethod
    def load(cls, path: str):
        """Загрузка модели"""
        normalizer = cls.__new__(cls)
        normalizer.name = "Transformer-T5"
        normalizer.model_name = path
        normalizer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        normalizer.tokenizer = AutoTokenizer.from_pretrained(path)
        normalizer.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        normalizer.model.to(normalizer.device)
        
        normalizer.max_length = 128
        normalizer.prefix = "normalize address: "
        
        return normalizer


class LLMNormalizer:
    """LLM нормализатор через Mistral AI API"""
    
    def __init__(self, api_key: str = None):
        self.name = "LLM-Mistral"
        
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY не установлен")
        
        self.client = Mistral(api_key=self.api_key)
        self.cache = {}
        
        self.prompt = """Ты эксперт по российским адресам. Приводи адреса к формату ГАР.
На вход тебе передается одна строка - это кривой адрес.
Правила:
1. Разверни все сокращения (ул → улица, обл → область и т.д.)
2. Исправь опечатки
3. Приведи к корректному порядку адресных элементов по ГАР: Регион, Район, Населенный пункт, Улица, Дом и т.д.
4. Используй правильные падежи для русского языка
5. Удали лишние знаки препинания
6. Формат: запятые между адресными элементами, пробелы после запятых
7. Не додумывай адресные элементы. Если номер дома, название улицы и т.д. не указаны, то просто пропусти этот адресный элемент.

Примеры (Неправильный адрес -> Адрес правильный в формате ГАР):
- '79 дом, город,, нагорная, тыва,, республика, ак-довурак, улица,,' → 'Республика Тыва, Ак-Довурак город, Нагорная улица, дом 79'
- 'Алтац, Тюгурюу Молодежпая улиЫа, дом 22' → 'Республика Алтай, Тюгурюк поселок, Молодежная улица, дом 22'
- 'ПскLовская областuь, tПсков гzород, kНародная улиdца, дом 49' → 'Псковская область, Псков город, Народная улица, дом 49'

Адрес для нормализации: {input_address}

В ответе верни только нормализованный адрес одной строкой, без пояснений, без лишних примеров, без исходного адреса, без спецсимволов,
 без слова ответ. Только одна строка корректного адреса в формате ГАР на русском языке."""
    
    def predict(self, address: str) -> str:
        """Предсказание с кэшированием"""
        if not self.client:
            return f"[API KEY не установлен] {address}"
        
        if address in self.cache:
            return self.cache[address]
        
        try:
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": self.prompt.format(input_address=address)}],
                temperature=0.1,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            result = result.replace('"', '').replace("'", "").strip()
            self.cache[address] = result
            return result
            
        except Exception as e:
            print(f"LLM ошибка: {e}")
            return address
