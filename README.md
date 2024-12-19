# 1) Abstractive Question & Answering with Generative AI

This project implements an abstractive Question & Answering (Q&A) system using Generative AI. It retrieves relevant documents based on a natural language question and generates human-readable answers using a generator model.

---

## Overview

### Workflow
1. **Input**: Ask a question in natural language.
2. **Retriever**:
   - Each context passage is converted into a embedding and stored in a vector database (e.g., Pinecone). This embedding captures the semantic and syntactic meaning of the text.
   - Convert the input question into a query vector and compare it with the stored vectors to find the most relevant segments.
3. **Generator**:
   - Convert the relevant vectors back to text.
   - Combine the original question with the retrieved text and pass it to a generator model (e.g., GPT-3 or BART) to produce a human-readable answer.

---

## Data Source
- **Wiki Snippets Dataset**:
  - Contains over 17 million passages from Wikipedia.
  - For simplicity, we use 50,000 passages that include "History" in the `section_title` column.
- **Streaming Mode**:
  - The dataset is loaded iteratively to avoid downloading the entire 9GB file upfront.
  - Extracted fields: `article_title`, `section_title`, and `passage_text`.

---

## Retriever
- **Model**: `flax-sentence-embeddings/all_datasets_v3_mpnet-base` by Microsoft.
- **Embedding**:
  - Encodes sentences into a 768-dimensional vector.
  - Stores embeddings in Pinecone with `dimensions=768` and `metric=cosine`.
- **Processing**:
  - Passages are encoded in batches of 64.
  - Metadata (`article_title`, `section_title`, `passage_text`) is attached to each embedding.
  - Data is indexed and upserted into the Pinecone database.

---

## Generator
- **Model**: `bart_lfqa`, trained on the ELI5 dataset.
- **Input Format**:
  - A single string combining the query and relevant documents, separated by a special `<P>` token:
    ```
    question: What is a sonic boom? context: <P> A sonic boom is a sound associated with shock waves created when an object travels through the air faster than the speed of sound. <P> Sonic booms generate enormous amounts of sound energy, sounding similar to an explosion or a thunderclap to the human ear. <P> Sonic booms due to large supersonic aircraft can be particularly loud and startling, tend to awaken people, and may cause minor damage to some structures.
    ```
- **Processing**:
  - Retrieves the most relevant context vectors from the query vector (`xq`).
  - Extracts metadata (`passage_text`) and concatenates all context passages.
  - Adds the original query to the concatenated context.
- **Output**:
  - The model tokenizes the final input, generates answers in token IDs, and decodes them to human-readable text.

---

## References
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [ELI5 Dataset](https://huggingface.co/datasets/eli5)

---
