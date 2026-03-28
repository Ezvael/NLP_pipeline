POS_EMOJI = {"😀","😃","😄","😁","😆","😊","😍","🥰","😘","❤️","💖","👍","🔥","💯","🎉"}
NEG_EMOJI = {"😡","😠","😞","😢","😭","👎","💔","😤","🤬"}

POS_SLANG = {"топ","огонь","имба","кайф","шик","бомба","круто"}
NEG_SLANG = {"жесть","капец","пипец","трэш","мрак","отстой","фигня"}

PROFANITY = {"блин","черт","хрен","дерьмо","сука","блять","хер","нахер","мудила","блядь","мразь","урод"}

BRANDS = {
    "мвидео": "Mvideo",
    "mvideo": "Mvideo",
    "эльдорадо": "Eldorado",
    "dns": "DNS",
    "днс": "DNS",
    "ozon": "OZON",
    "озон": "OZON",
    "wildberries": "Wildberries",
    "вайлдберрис": "Wildberries",
    "вб": "Wildberries",
    "aliexpress": "Aliexpress",
    "алиэкспресс": "Aliexpress"
}

SARCASM_MARKERS = {"ага","ну да","конечно","очень смешно","прям","ясно"}

NEG_WORDS = {"не","нет","ни","никогда"}

TOPIC_KEYWORDS = {
    "TOPIC_CHATBOT": {"чат","бот","чатбот"},
    "TOPIC_PRICING": {"цена","дорого","дешево"},
}

TOPIC_KEYWORDS = {k: set(v) for k, v in TOPIC_KEYWORDS.items()}