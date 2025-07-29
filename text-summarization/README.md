# Text Summarization with BART

This project demonstrates text summarization using the BART model from Hugging Face Transformers.

## Features

- Summarizes long texts into concise summaries
- Utilizes pre-trained BART models
- Simple interface for input and output

## Setup

1. Clone the repository.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the summarization script:
    ```bash
    python summarize.py --input input.txt --output summary.txt
    ```

## Usage

- Place your text in `input.txt`.
- The summary will be saved to `summary.txt`.

## Model

- Uses `facebook/bart-large-cnn` from Hugging Face.

## References

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
