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

# 2) Fine-Tuning BERT for Classification

## Project Overview
This project demonstrates the fine-tuning of the pre-trained open-source **BERT** model for a **binary classification** task. 

---

## Steps Involved

### 1. Preprocessing
- **Duplicate and Null Values:** Removed any duplicate rows or rows containing null values.
- **Class Imbalance:** Checked for class imbalance. Addressed it by assigning higher penalties for misclassifying the minority class during training, ensuring the model focuses adequately on minority class instances.

### 2. Data Splitting
- The dataset was split into three sets: 
  - **Training Set (70%)**
  - **Validation Set (15%)**
  - **Test Set (15%)**

### 3. Model and Tokenizer
- **Model:** Used the pre-trained `BERT-base` (uncased) model.
- **Tokenizer:** Loaded the `BERT-base` tokenizer for tokenizing the input text.

### 4. Input Length and Batch Encoding
- Since the **BERT model** accepts inputs of a maximum length of **512 tokens**:
  - **Padding:** Smaller sentences were padded to match the batch length.
  - **Truncation:** Longer sentences were truncated to retain relevant information.
- Chose an optimal max input length of **25 tokens** based on dataset characteristics.
![image](https://github.com/user-attachments/assets/3d9d6bdd-9411-4782-8cbb-5d4288f90437)


### 5. Data Preparation
- Converted input IDs, attention masks, and labels into **PyTorch tensors**.
- Used `DataLoader` and samplers for batching and shuffling data at every epoch.

---

## Fine-Tuning Approaches
We explored the following methods of fine-tuning:
1. **Train all weights:** All layers are trained.
2. **Freeze a few layers:** Only the unfreezed layer weights are trained.
3. **Freeze all layers:** Added new layers on top and trained only the new layers.

For this project, we opted to **train all weights**.

---

## Model-Architecture
![image](https://github.com/user-attachments/assets/9c3eaddf-d13c-45c5-a8f0-8613ae492fb2)

---

## Model Training
### Configurations
- **Optimizer:** `AdamW` with a learning rate of `1e-5`.
- **Loss Function:** Cross-entropy loss, handling class imbalance.
- **Training Epochs:** Set to **10**.

### Additional Notes
- A learning rate scheduler was considered but omitted due to the small dataset size.

![image](https://github.com/user-attachments/assets/0ab82711-edc7-4691-8319-e7b84e00d134)

---

# 3) Neural Machine Translation (German to English)

## Project Overview
This project focuses on text language translation from German to English using a neural network-based approach. The model takes a German sentence as input and outputs its English translation.

## Preprocessing
1. **Data Cleaning**:
   - Removed duplicates and null values.
   - Converted all text to lowercase and stripped punctuation.
2. **Feature Engineering**:
   - **Max Length**: Set to 8 for both English and German sentences.
   - **Tokenization**:
     - Separate tokenizers for German and English.
     - Vocabulary size: **6098** (English) and **10071** (German).
   - **Padding & Truncation**: Performed to ensure uniform input sizes within a batch.
   - Converted tokenized sentences to tensors for model training.
     
   ![image](https://github.com/user-attachments/assets/b16bc50d-8ee1-48a5-b551-14e2e03f3a8b)


---

## Model Architecture
![image](https://github.com/user-attachments/assets/40b5c24e-bc94-41b6-bf9f-0ae8097164be)

### Encoder
- **Embedding Layer**:
  - Converts input integers to dense vectors of fixed size.
  - Dropout applied with a probability of **0.2**.
- **Bidirectional LSTM**:
  - Three stacked layers, with the first two returning sequences.
  - Each layer is followed by Layer Normalization and Dropout (**0.2**).

### Decoder
- **RepeatVector**:
  - Repeats encoder output to match decoder time steps.
- **LSTM Layers**:
  - Two stacked layers with **2× units** size.
  - Includes Layer Normalization and Dropout (**0.2**).
- **Output Layer**:
  - Dense layer with a **softmax activation** to generate probabilities for the target vocabulary.

---

## Training Parameters
- **Optimizer**: Adam with learning rate **0.001**.
- **Loss Function**: Sparse categorical crossentropy.
- **Metric**: Accuracy.
- **Early Stopping**:
- Monitored validation loss with a patience of **5 epochs**.

![image](https://github.com/user-attachments/assets/b3c8ff47-68a8-4e8e-96e0-b2b410d46053)

---
# 4) Quora Question Pairs

## Project Overview
The aim of this project is to identify pairs of questions that have the same intent, even if they are phrased differently due to variations in wording or grammar. This task is inspired by a Kaggle competition hosted by Quora, with the goal of improving user experience by reducing fragmented answers across duplicate questions.

## Preprocessing
Key transformations performed on the data include:
1. Converted text to lowercase.
2. Replaced emojis with their meanings.
3. Expanded contractions (e.g., `'ve` to `have`).
4. Replaced special characters with descriptive names.
5. Shortened large numbers (e.g., `1,000,000` to `1m`).
6. Removed HTML tags.
7. Abbreviated common terms (e.g., `GM` for `Good Morning`).
8. Applied stemming to reduce words to their root forms.

**Important Notes**:
- **Stop Words**: Retained to create new features for classification.
- **Spelling Correction**: Omitted due to computational constraints with the large dataset.

---

## Feature Engineering
### Batch 1: Basic Features
1. Length of both questions (q1, q2).
2. Number of words in both questions.
3. Common words between q1 and q2.
4. Total words in q1 and q2.
5. Ratio of common words to total words.

![image](https://github.com/user-attachments/assets/25f50796-dd58-46ef-bd53-ce6308a5e62e)


### Batch 2: Word Ratios
1. Ratio of common words to minimum and maximum lengths of both questions.
2. Ratio of common stop words to minimum and maximum lengths.
3. Ratio of common tokens to minimum and maximum lengths.
4. First and last word match status between both questions.

### Batch 3: Token-Based Features
1. Absolute difference in the number of tokens.
2. Average number of tokens.
3. Ratio of longest common substring to the minimum length.

### Batch 4: Fuzzy Matching
1. **Fuzzy Ratio**
2. **Fuzzy Partial Ratio**
3. **Token Sort Ratio**
4. **Token Set Ratio**

![image](https://github.com/user-attachments/assets/f65925f3-17fc-4eec-8cd6-d6aa412913c7)

---

## Bag of Words Representation
- A vocabulary of the top **3000 most frequent words** was created for both questions (q1, q2).
- Each sentence was represented as a vector using this vocabulary.
- Combined these **6000 dimensions** with the **23 features** created earlier to form a total of **6023 features**.

**Challenges with Bag of Words**:
- Sparse matrix representation.
- Lack of semantic context.
- High-dimensional data.
- Out-of-vocabulary (OOV) issues.

---

## Model Training & Evaluation
1. **Dataset Split**:
   - 80% for training and 20% for testing.
2. **Classifier**: Random Forest.
3. **Evaluation**:
   - Achieved an accuracy of approximately **0.7863** on the test dataset.

---

## References
- [Kaggle Quora Question Pairs Competition](https://www.kaggle.com/c/quora-question-pairs)

---
# 5) Text-Summarization-Amazon-Fine-Food-Reviews

## Project Overview
This project focuses on text summarization. The input is a long review of Amazon Fine Food products, and the output is a concise summary of the review.

---

## Pre-Processing

### Transformations:
1. Converted all text to lowercase.
2. Replaced emojis with their meanings.
3. Removed pre-encoded emojis that could not be demojized.
4. Expanded contractions (e.g., `'ve` to `have`).
5. Removed HTML tags.
6. Abbreviated common terms (e.g., `GM` for "Good Morning").
7. Removed stop words.

### Exclusions:
- **Stemming**: Avoided because root forms may not produce proper English words.
- **Lemmatization**: Skipped due to high computation time.
- **Spelling Correction**: Omitted because of the dataset's size and associated time costs.

#### Example:
- Input: `Bought several vitality canned dog food products, found good quality.`
- Output: `good quality dog food`

---

## Feature Engineering

![image](https://github.com/user-attachments/assets/72dc998a-9eec-43d7-b6e1-b8a83efb2c0f)

1. **Input Length Optimization**:
   - Text max length: **80 tokens**.
   - Summary max length: **7 tokens**.

2. **Padding and Truncation**:
   - Smaller sentences are padded.
   - Longer sentences are truncated.

3. **Tokenization**:
   - Used **BERT tokenizer (uncased)** to convert text into token IDs.
   - Pre-existing vocabulary ensures better linguistic coverage.

---

## Model Architecture

### Encoder:
1. Embedding layer converts tokens into 256-dimensional embeddings.
2. Three LSTM layers process input, propagating context across timesteps.
3. Final LSTM outputs:
   - Hidden states at each timestep.
   - Final hidden and cell states (`state h`, `state c`).

### Decoder:
1. Inputs:
   - Final encoder states (`state h`, `state c`).
   - Summary tokens embedded similarly.
2. Processes sequence with attention applied to encoder outputs.

### Attention Mechanism:
- **Keys/Values**: Encoder outputs.
- **Queries**: Decoder's current hidden states.
- Produces **context vectors** focusing on relevant parts of the input.

### Dense Layer:
- Combines context vectors with decoder outputs.
- Applies softmax to predict the next token probabilities.

---

## Training

1. **Parameters**:
   - Optimizer: **RMSprop**.
   - Loss function: **Sparse categorical crossentropy**.
   - Metric: **Accuracy**.

2. **Regularization**:
   - Early stopping: Halt training after 5 epochs of no improvement.
  
   ![image](https://github.com/user-attachments/assets/55013280-92dd-470e-9f66-e28166d91146)


---

## Inference

### Encoder Inference:
- **Inputs**: Review text.
- **Outputs**:
  - Encoder hidden states.
  - Final states (`state h`, `state c`).

### Decoder Inference (Per Timestep):
1. **Inputs**:
   - Current token.
   - Previous hidden and cell states (`state h`, `state c`).
   - Encoder outputs for attention.
2. **Outputs**:
   - Updated states.
   - Token probability distribution.

---

## Key Functions

### `decode_sequence(input_seq)`:
- **Purpose**: Generates a summary from the input review.
- **Process**:
  1. Encode input sequence using the encoder.
  2. Initialize with the `start` token.
  3. Iteratively predict tokens until:
     - Reaching the `end` token.
     - Hitting the maximum summary length.
  4. Update states and target sequence at each step.

### `seq2summary(input_seq)`:
- **Purpose**: Converts the sequence into a readable summary.
- **Process**:
  - Remove padding (`0`) and special tokens (`start`, `end`).
  - Convert indices into words using the reverse target vocabulary.

### `seq2text(input_seq)`:
- **Purpose**: Converts the input sequence into readable text.
- **Process**:
  - Remove padding (`0`).
  - Convert indices into words using the reverse source vocabulary.

---

# 6) Twitter Sentiment Analysis

## Project Overview
This project addresses a Kaggle competition hosted by Twitter. The goal is to identify tweets containing racist or sexist content, enabling measures to block such tweets and reduce online bullying and negativity.

---

## Pre-Processing

### Steps:
1. Converted all text to **lowercase**.
2. Replaced emojis with their **meanings**.
3. Removed pre-encoded emojis that could not be **demojized**.
4. Expanded **contractions** (e.g., `'ve` → `have`).
5. Replaced **special characters** with their names, except for `#` (useful for identifying trends).
6. Replaced usernames (`@tags`) with `user` and subsequently **dropped them** to maintain privacy.
7. Removed **HTML tags**.
8. Abbreviated **common terms** (e.g., `GM` → "Good Morning").
9. Removed **stop words**.
10. Applied **stemming** to reduce words to their root forms.

### Exclusions:
- **Spelling correction**: Skipped due to the large dataset size and the computational cost.

---

## Feature Engineering

### Key Steps:
1. **Hashtag Extraction**:
   - Analyzed hashtags (`#`) to evaluate their association with racist or non-racist tweets.
  
   ![image](https://github.com/user-attachments/assets/3d95e271-94f8-4955-a45d-b7868396d593)

2. **Corpus Creation**:
   - Combined training and testing datasets to build a **corpus of words**.
   - Selected the **top 1000 most frequent words** as vector dimensions to represent each tweet.

3. **Comparison**:
   - Evaluated **Bag of Words (BOW)** and **TF-IDF** methods for vector representation.

4. **Feature Combination**:
   - Created 20 additional features based on 10 common labels from each category (racist and non-racist).
   - Final dataset comprised **1020 features** (1000 from BOW/TF-IDF + 20 custom features).

---

## Model Training & Evaluation

### Classifier:
- Used a **Logistic Regression** model for classification.

### Results:
1. **Bag of Words (BOW)**:
   - Achieved an **f1 score** of **0.544**.
2. **TF-IDF**:
   - Achieved an improved **f1 score** of **0.559**.

---
