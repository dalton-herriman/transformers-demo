from transformers import pipeline

# Set up a summarization pipeline using BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")