import argparse
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from transformers.pipelines.pt_utils import RetrieverMixin

DOCUMENTS = [
    "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of documents with generative models to answer questions based on external knowledge.",
    "The Hugging Face Transformers library provides state-of-the-art machine learning models for natural language processing.",
    "Sentence Transformers are models that produce semantically meaningful sentence embeddings for retrieval tasks.",
    "RAG models use a retriever to fetch relevant documents and a generator to produce answers conditioned on those documents.",
    "Python is a popular programming language for machine learning and data science."
]

class FaissRetriever(RetrieverMixin):
    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):
        self.documents = documents
        self.embedder = SentenceTransformer(model_name)
        self.doc_embeddings = self.embedder.encode(documents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings)

    def retrieve(self, query, top_k=2):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in I[0]]

def generate_answer(question, context, generator, tokenizer, max_length=128):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output_ids = generator.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="RAG Prototype with Transformers + FAISS")
    parser.add_argument("--question", type=str, required=True, help="Question to answer")
    args = parser.parse_args()

    print("Loading retriever model and FAISS index...")
    retriever = FaissRetriever(DOCUMENTS)

    print("Loading generator model...")
    generator_name = "facebook/bart-large"
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    generator = AutoModelForSeq2SeqLM.from_pretrained(generator_name)

    print("Retrieving relevant documents...")
    retrieved_docs = retriever.retrieve(args.question, top_k=2)
    context = " ".join(retrieved_docs)

    print("Generating answer...")
    answer = generate_answer(args.question, context, generator, tokenizer)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()