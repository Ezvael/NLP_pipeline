import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "blanchefort/rubert-base-cased-sentiment"

class SentimentAnalyzer:

    def __init__(self, batch_size=16, device=None):

        self.batch_size = batch_size
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, texts):

        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():

                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=1)
            probs = probs.cpu().numpy()

            for neg, neu, pos in probs:

                scores = {
                    "negative": float(neg),
                    "neutral": float(neu),
                    "positive": float(pos),
                }

                label = max(scores, key=scores.get)
            
                results.append({
                    "sentiment": label,
                    "sentiment_confidence": (scores[label])
                })

            del inputs
            del outputs
            del probs

            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return pd.DataFrame(results)