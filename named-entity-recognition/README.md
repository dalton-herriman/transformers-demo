# Named Entity Recognition for AD Windows Usernames

This project demonstrates how to use BERT-based models from the Hugging Face Transformers library to perform Named Entity Recognition (NER) for detecting Active Directory (AD) Windows usernames in text.

## Features

- Utilizes pre-trained BERT models for NER tasks.
- Customizes entity detection to identify AD Windows usernames (e.g., `DOMAIN\username` or `username@domain.com`).
- Easily extensible for other entity types.

## Requirements

- Python 3.7+
- [transformers](https://huggingface.co/transformers/)
- [torch](https://pytorch.org/)

Install dependencies:
```bash
pip install transformers torch
```

## Usage

1. Prepare your dataset with labeled AD Windows usernames.
2. Train or fine-tune a BERT model for NER using your dataset.
3. Run inference to extract usernames from new text.

Example inference:
```python
from transformers import pipeline

ner = pipeline("ner", model="your-finetuned-bert-model")
text = "The user DOMAIN\\jdoe logged in."
results = ner(text)
print(results)
```

## Customization

- Update the training data to match your organization's username patterns.
- Adjust model hyperparameters for improved accuracy.

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
