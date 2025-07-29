from transformers import pipeline

# Set up a summarization pipeline using BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarizes the input text using the BART model.
    
    Args:
        text (str): The text to summarize.
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.
    
    Returns:
        str: The summarized text.
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Example text to summarize
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "This is a classic example of a pangram, a sentence that contains every letter of the alphabet. "
        "Pangrams are often used in typing practice and font testing."
    )
    
    # Summarize the text
    summary = summarize_text(text)
    print("Original Text:", text)
    print("Summary:", summary)