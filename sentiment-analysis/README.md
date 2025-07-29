# Sentiment Analysis Demo

This project provides a simple sentiment analysis tool using Hugging Face Transformers and PyTorch.

## Features

- Uses the `distilbert-base-uncased-finetuned-sst-2-english` model.
- Classifies input text as `positive`, `negative`, or `neutral` (if confidence is low).
- Command-line interface for quick testing.

## Requirements

- Python 3.7+
- `torch`
- `transformers`

Install dependencies:

```bash
pip install torch transformers
```

## Usage

Run the script and enter text to analyze:

```bash
python sentiment_analysis.py
```

Example output:

```
Enter text to analyze: I love this product!
{'label': 'positive', 'score': 0.9998}
```

## How it works

- The script loads a pre-trained sentiment analysis pipeline.
- The `analyze` function returns a sentiment label and confidence score.
- If the model's confidence is below 0.6, the label is set to `neutral`.

## License

MIT