from functools import lru_cache
from pymorphy3 import MorphAnalyzer
from stop_words import get_stop_words

morph = MorphAnalyzer()
stopwords_ru = set(get_stop_words("ru"))

@lru_cache(maxsize=100000)
def get_lemma(word: str):

    return morph.parse(word)[0].normal_form


class Lemmatizer:

    def lemmatize(self, text):

        tokens = text.split()
        lemmas = []

        for token in tokens:

            if len(token) < 3:
                continue

            if token in stopwords_ru:
                continue

            lemmas.append(get_lemma(token))

        return " ".join(lemmas)