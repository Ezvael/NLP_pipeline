import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from pymorphy3 import MorphAnalyzer

# Download necessary NLTK data (if not already downloaded)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Regex patterns for initial clustering
patterns = {
    "chatbot": r"""\b(
(чат[\s\-]?бот\w*)|
(chat[\s\-]?bot\w*)|
(chatbot\b)|
(бот(ы|а|ов|у|ом|ами)?)|
((?:онлайн|ии|виртуал|электрон)[\s\-]?ассист\w*)|
(assist\w*)|
((?:онлайн|ии|виртуал|электрон)[\s\-]?помо[шщ]н\w*)
)""",
    "personalization": r"""\b(
(персонализ\w*)|
(рекомендац\w*)|
(персонифик\w*)|
(индивидуализ\w*)
)""",
    "pricing": r"""\b(
((?:цен|рассчит|прайс|стоимость).[1-15](?:мен[аяи]|формирован|установлен|расс?чет)\w*)|
((?:мен[аяи]|формирован|установлен|расс?чет).*?(?:цен|рассчит|прайс|стоимость))|
(ценообразование\b)|
(прейскурант\b)
)""",
    "suggest": r"""\b(
(саджест\b)|
(подсказ\w*)|
(автозап\w*)|
(автопод\w*)
)""",
    "timing": r"""\b(
(дост.*?врем\w*)|
(врем.*?дост\w*)|
(когда.*?(?:дост|прий?деё?т|заказ)\w*)|
(задерж\w*)|
(ожида\w*)|
(жд[уа]\w*)|
(долго\b)|
(отсроч\w*)|
(затягива\w*)|
(промедлен\w*)
)"""
}

# Initialize tokenizer, stopwords, and lemmatizer
tokenizer = TweetTokenizer()
stopwords_ru = set(stopwords.words("russian"))
morph = MorphAnalyzer()

def lemmatize(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    splitted_txt = tokenizer.tokenize(text.lower())
    words = [morph.parse(token)[0].normal_form for token in splitted_txt if token not in stopwords_ru]
    return ' '.join(words)

def classify_multi(df, text_col, patterns, new_col="clusters"):
    compiled = {k: re.compile(v, re.IGNORECASE | re.VERBOSE) for k, v in patterns.items()}

    def classify(text):
        if pd.isna(text) or not isinstance(text, str):
            return []
        found = []
        for label, pattern in compiled.items():
            if pattern.search(text):
                found.append(label)
        return found

    df[new_col] = df[text_col].apply(classify)
    return df

def preprocess_data(df, text_column='Комментарий'):
    # Apply regex-based classification
    df = classify_multi(df, text_column, patterns)
    
    # Lemmatize text
    df['new comment'] = df[text_column].astype(str).apply(lemmatize)
    
    return df
