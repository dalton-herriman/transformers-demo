import torch
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze(text):
    if not text:
        return {'error': 'No text provided'}

    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()
    score = float(result['score'])

    return {
        'label': 'neutral' if score < 0.6 else label,
        'score': score
    }

if __name__ == '__main__':
    print(analyze(input("Enter text to analyze: ")))
