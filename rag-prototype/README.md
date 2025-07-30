# RAG Prototype with Transformers and FAISS

This repository contains a prototype script demonstrating Retrieval-Augmented Generation (RAG) using the [Hugging Face Transformers](https://github.com/huggingface/transformers), [Sentence Transformers](https://www.sbert.net/), and [FAISS](https://github.com/facebookresearch/faiss) libraries. The script combines a semantic retriever (using FAISS and Sentence Transformers) and a sequence-to-sequence generator model to answer questions based on a set of documents.

## Features

- Loads a sample set of documents (customizable or load from file).
- Uses `sentence-transformers` to embed documents and queries.
- Uses [FAISS](https://github.com/facebookresearch/faiss) for fast vector similarity search.
- Uses a sequence-to-sequence generator model (`facebook/bart-large`) for answer generation.
- Simple command-line interface for querying.

## Requirements

- Python 3.8+
- `torch`
- `transformers`
- `sentence-transformers`
- `faiss-cpu` (or `faiss-gpu` if using GPU)
- `argparse`
- `numpy`

Install dependencies:

```bash
pip install torch transformers sentence-transformers faiss-cpu numpy
```

## Usage

1. (Optional) Edit the `DOCUMENTS` list in `rag_prototype.py` or load your own documents.
2. Run the script:

```bash
python rag_prototype.py --question "What is retrieval-augmented generation?"
```

3. The script retrieves relevant documents using FAISS and generates an answer.

## Example

```bash
$ python rag_prototype.py --question "What is RAG?"
Loading retriever model and FAISS index...
Loading generator model...
Retrieving relevant documents...
Generating answer...
Answer: Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of documents with generative models to answer questions based on external knowledge.
```

## Notes

- This is a minimal prototype for demonstration purposes.
- For production use, consider more robust retrieval, indexing, and document management.
- FAISS enables efficient similarity search for large document sets.

## License

MIT License
