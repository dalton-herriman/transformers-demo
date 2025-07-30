import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
import re

# Load pre-trained BERT model and tokenizer for NER
model_name = "dslim/bert-base-NER"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def is_ad_username(entity_text):
    # Simple heuristic: AD usernames are often alphanumeric, no spaces, may include dot or underscore
    return bool(re.fullmatch(r"[a-zA-Z0-9._\\-]{3,64}", entity_text))

def detect_ad_usernames(text):
    ner_results = ner_pipeline(text)
    usernames = []
    for entity in ner_results:
        # You may want to adjust this logic based on your AD username patterns
        if is_ad_username(entity['word']):
            usernames.append(entity['word'])
    return usernames

if __name__ == "__main__":
    sample_text = "The ticket was created by jdoe and assigned to jane.smith in Active Directory."
    usernames = detect_ad_usernames(sample_text)
    print("Detected AD usernames:", usernames)