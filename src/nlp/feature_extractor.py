from src.config import POS_EMOJI, NEG_EMOJI, PROFANITY, POS_SLANG, NEG_SLANG

class FeatureExtractor:

    def extract_tags(self, text, lemmas):

        tags = []
        chars = set(text)

        if chars & POS_EMOJI:
            tags.append("TAG_EMO_POS")

        if chars & NEG_EMOJI:
            tags.append("TAG_EMO_NEG")

        for lemma in lemmas:

            if lemma in PROFANITY:
                tags.append("TAG_PROFANITY")

            if lemma in POS_SLANG:
                tags.append("TAG_SLANG_POS")

            if lemma in NEG_SLANG:
                tags.append("TAG_SLANG_NEG")

        return list(set(tags))