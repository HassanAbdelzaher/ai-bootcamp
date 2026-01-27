# Step 7e: NLP Applications

> **Learn practical Natural Language Processing techniques for real-world applications**

**Time**: ~90 minutes  
**Prerequisites**: Step 7 (RNNs), Step 7d (Transformers)

---

## 🎯 Learning Objectives

By the end of this step, you'll understand:
- Tokenization methods (character, word, subword)
- Word embeddings and their importance
- Named Entity Recognition (NER)
- Sentiment analysis implementation
- Text classification pipelines
- Sequence-to-sequence models
- Real-world NLP applications

---

## 📚 What is Natural Language Processing?

**Natural Language Processing (NLP)** enables computers to understand, interpret, and generate human language.

### Key NLP Tasks

- **Tokenization**: Breaking text into words/tokens
- **Word Embeddings**: Representing words as vectors
- **Named Entity Recognition**: Finding people, places, organizations
- **Sentiment Analysis**: Determining positive/negative emotions
- **Text Classification**: Categorizing documents
- **Machine Translation**: Translating between languages
- **Question Answering**: Answering questions from text
- **Text Summarization**: Creating summaries

---

## 🔤 Tokenization

### What is Tokenization?

**Tokenization** is the process of breaking text into smaller units called tokens.

### Types of Tokenization

#### 1. Character-Level

**What**: Each character is a token

**Example**: "Hello" → ['H', 'e', 'l', 'l', 'o']

**Pros**:
- Small vocabulary
- Handles any text
- No unknown words

**Cons**:
- Very long sequences
- Loses word meaning
- Harder to learn

**When to use**: Very limited vocabulary, character-based languages

#### 2. Word-Level

**What**: Each word is a token

**Example**: "Hello world!" → ['Hello', 'world']

**Pros**:
- Preserves word meaning
- Shorter sequences
- Intuitive

**Cons**:
- Large vocabulary
- Unknown words problem
- Doesn't handle subwords

**When to use**: Most common, good starting point

#### 3. Subword-Level

**What**: Words broken into smaller pieces

**Example**: "learning" → ['learn', '##ing']

**Pros**:
- Handles unknown words
- Smaller vocabulary
- Captures morphology

**Cons**:
- More complex
- Longer sequences
- Less intuitive

**When to use**: BERT, GPT, multilingual models

### Tokenization Process

1. **Text normalization**: Lowercase, remove extra spaces
2. **Punctuation handling**: Keep, remove, or separate
3. **Splitting**: Break into tokens
4. **Special tokens**: Add `<PAD>`, `<UNK>`, `<START>`, `<END>`

### Implementation

```python
import re

def simple_tokenize(text):
    """Simple word tokenization"""
    text = text.lower()
    text = re.sub(r'[^\w\s\']', ' ', text)  # Remove punctuation
    tokens = text.split()
    return tokens

# Example
text = "Hello world! How are you?"
tokens = simple_tokenize(text)
# ['hello', 'world', 'how', 'are', 'you']
```

---

## 🔢 Word Embeddings

### What are Word Embeddings?

**Word Embeddings** are dense vector representations of words that capture semantic meaning.

### Why Embeddings?

**One-Hot Encoding Problems**:
- Sparse vectors (mostly zeros)
- No semantic relationships
- Large vocabulary = huge vectors
- "cat" and "dog" are equally different from "car"

**Embeddings Benefits**:
- Dense vectors (e.g., 100-300 dimensions)
- Similar words have similar vectors
- Captures relationships (king - man + woman ≈ queen)
- Enables neural networks to process text

### Types of Embeddings

#### 1. Learned Embeddings

**What**: Embeddings learned during model training

**How**: Random initialization, updated via backpropagation

**Pros**:
- Task-specific
- Learns relevant features

**Cons**:
- Requires training data
- May not capture general semantics

#### 2. Pre-trained Embeddings

**What**: Embeddings trained on large corpora

**Popular Methods**:

**Word2Vec**:
- Skip-gram or CBOW
- Learns from word co-occurrence
- Fast and efficient

**GloVe (Global Vectors)**:
- Uses global word-word co-occurrence matrix
- Combines advantages of global and local methods
- Good performance

**FastText**:
- Handles subwords
- Better for rare words
- Good for morphologically rich languages

**Contextual Embeddings (BERT, GPT)**:
- Different embedding for each context
- "bank" (river) vs "bank" (financial) have different embeddings
- State-of-the-art performance

### Embedding Visualization

Similar words cluster together in embedding space:
- **Semantic clusters**: "cat", "dog", "bird" (animals)
- **Syntactic clusters**: "run", "runs", "running" (verbs)
- **Relationships**: king - man + woman ≈ queen

---

## 🏷️ Named Entity Recognition (NER)

### What is NER?

**Named Entity Recognition** identifies and classifies named entities in text.

### Entity Types

- **Person**: "John Smith", "Mary", "Einstein"
- **Organization**: "Apple Inc.", "MIT", "United Nations"
- **Location**: "New York", "Paris", "California"
- **Date**: "January 2024", "Monday", "2023"
- **Money**: "$100", "50 euros", "£25"
- **Product**: "iPhone", "Tesla Model 3"

### NER Approaches

#### 1. Rule-Based

**What**: Hand-crafted rules and patterns

**Example**: Capitalized words might be entities

**Pros**: Simple, interpretable

**Cons**: Limited, doesn't scale

#### 2. Machine Learning

**What**: Train models to recognize entities

**Methods**:
- **BIO Tagging**: B-begin, I-inside, O-outside
- **Sequence Labeling**: LSTM, BERT
- **Token Classification**: Each token gets a label

**Example**:
```
Token:  John  works  at  Apple  Inc.
Label:  B-PER O     O   B-ORG  I-ORG
```

### NER Applications

- **Information Extraction**: Extract structured data
- **Search**: Improve search results
- **Knowledge Graphs**: Build knowledge bases
- **Content Analysis**: Analyze documents

---

## 😊 Sentiment Analysis

### What is Sentiment Analysis?

**Sentiment Analysis** determines the emotional tone or opinion expressed in text.

### Sentiment Levels

1. **Binary**: Positive / Negative
2. **Three-class**: Positive / Negative / Neutral
3. **Fine-grained**: Very positive, positive, neutral, negative, very negative
4. **Aspect-based**: Sentiment about specific aspects

### Approaches

#### 1. Rule-Based

**What**: Use sentiment lexicons and rules

**Example**: Count positive/negative words

**Pros**: Simple, interpretable

**Cons**: Limited, misses context

#### 2. Machine Learning

**What**: Train classifiers on labeled data

**Models**:
- **Naive Bayes**: Simple baseline
- **LSTM/RNN**: Sequence modeling
- **BERT**: State-of-the-art
- **Ensemble**: Combine multiple models

### Implementation Pipeline

1. **Data Collection**: Gather labeled text
2. **Preprocessing**: Clean, tokenize
3. **Feature Extraction**: Embeddings
4. **Model Training**: Classifier
5. **Evaluation**: Accuracy, F1-score

---

## 📋 Text Classification Pipeline

### Complete Pipeline

#### 1. Data Preprocessing

- **Lowercasing**: "Hello" → "hello"
- **Remove punctuation**: "Hello!" → "Hello"
- **Handle special characters**: URLs, emails
- **Normalize**: Unicode normalization

#### 2. Tokenization

- Choose level (word/subword)
- Handle unknown words
- Create vocabulary

#### 3. Vocabulary Building

- Map words to indices
- Handle `<UNK>` (unknown words)
- Add special tokens (`<PAD>`, `<START>`, `<END>`)

#### 4. Sequence Encoding

- Convert tokens to indices
- Padding: Make sequences same length
- Truncation: Cut long sequences

#### 5. Embedding

- Convert indices to dense vectors
- Use pre-trained or learn embeddings
- Handle variable-length sequences

#### 6. Model Training

- **RNN/LSTM**: For sequences
- **CNN**: For text classification
- **Transformer**: For complex tasks
- **Classification head**: Final layer

#### 7. Evaluation

- Accuracy, Precision, Recall, F1
- Confusion matrix
- Per-class metrics

---

## 🔄 Sequence-to-Sequence Models

### What is Seq2Seq?

**Sequence-to-Sequence** models convert one sequence to another.

### Applications

- **Machine Translation**: English → French
- **Text Summarization**: Long text → Summary
- **Question Answering**: Question → Answer
- **Dialogue Systems**: User input → Response

### Architecture

#### Encoder

- Processes input sequence
- Creates context vector
- RNN/LSTM/Transformer

#### Decoder

- Generates output sequence
- Uses context from encoder
- Autoregressive (one token at a time)

### Training

**Teacher Forcing**:
- During training: Use ground truth tokens
- Faster convergence
- More stable

**Inference**:
- Use previous predictions
- Autoregressive generation
- Can accumulate errors

### Attention Mechanism

- **Focus on relevant parts** of input
- **Improves long sequences**
- **Better context understanding**

---

## 🌍 Real-World NLP Applications

### Chatbots & Virtual Assistants

- **Customer service**: Answer questions
- **Siri, Alexa**: Voice assistants
- **Conversational AI**: Natural dialogue

### Email & Text Processing

- **Spam detection**: Filter unwanted emails
- **Auto-categorization**: Organize emails
- **Smart replies**: Suggest responses

### Search & Information Retrieval

- **Google Search**: Find relevant documents
- **Document search**: Enterprise search
- **Question answering**: Answer questions from text

### Content Analysis

- **News categorization**: Classify articles
- **Social media monitoring**: Track sentiment
- **Trend analysis**: Identify topics

### Translation

- **Google Translate**: Multilingual translation
- **Real-time translation**: Live conversations
- **Document translation**: Translate documents

### Text Generation

- **GPT models**: Generate text
- **Content creation**: Write articles
- **Code generation**: Generate code

---

## ✅ Best Practices

### Data Preprocessing

✅ **Clean and normalize text**
✅ **Handle special cases** (URLs, emails)
✅ **Consider language-specific rules**
✅ **Preserve important information**

### Tokenization

✅ **Choose appropriate level** (char/word/subword)
✅ **Handle unknown words**
✅ **Preserve important information**
✅ **Consider vocabulary size**

### Embeddings

✅ **Use pre-trained embeddings** when possible
✅ **Fine-tune for your task**
✅ **Consider contextual embeddings** (BERT)
✅ **Match embedding dimension to task**

### Model Selection

✅ **Simple tasks**: RNN/LSTM
✅ **Complex tasks**: Transformers
✅ **Consider computational budget**
✅ **Start simple, then scale up**

### Evaluation

✅ **Use appropriate metrics**
✅ **Consider class imbalance**
✅ **Test on diverse data**
✅ **Monitor for bias**

---

## 💻 Code Examples

### Tokenization

```python
import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

tokens = tokenize("Hello world!")
# ['hello', 'world']
```

### Word Embeddings

```python
import torch.nn as nn

vocab_size = 10000
embedding_dim = 100

embedding = nn.Embedding(vocab_size, embedding_dim)
# Converts token indices to dense vectors
```

### Sentiment Classification

```python
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 3)  # 3 classes
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])
```

---

## 📊 Visualizations

The step includes:
1. **Tokenization Examples** - Character, word, subword
2. **Embedding Visualization** - 2D projection of word vectors
3. **Sentiment Training** - Learning curve
4. **NER Examples** - Detected entities

---

## ✅ Key Takeaways

1. **Tokenization is foundational** - Choose appropriate level
2. **Embeddings capture semantics** - Similar words cluster together
3. **NLP models learn patterns** - From word co-occurrence to context
4. **Pre-trained models are powerful** - Leverage existing knowledge
5. **Pipeline matters** - Preprocessing → Model → Evaluation

---

## 🚀 Next Steps

After this step, you can:
- Tokenize text appropriately
- Use word embeddings effectively
- Build sentiment classifiers
- Implement NER systems
- Create text classification pipelines
- Understand seq2seq models

**To dive deeper**:
- Try pre-trained models (BERT, GPT)
- Explore advanced tokenization (SentencePiece, BPE)
- Build translation systems
- Implement question answering

---

## 📚 Additional Resources

- [NLTK Documentation](https://www.nltk.org/) - Natural Language Toolkit
- [spaCy](https://spacy.io/) - Industrial-strength NLP
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Pre-trained models
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781) - Original Word2Vec

---

## 🎓 Summary

**Natural Language Processing** enables computers to understand human language:

1. **Tokenization**: Break text into tokens
2. **Embeddings**: Represent words as vectors
3. **NER**: Find entities in text
4. **Sentiment**: Determine emotional tone
5. **Classification**: Categorize documents
6. **Seq2Seq**: Convert sequences

**Key insight**: NLP combines linguistic knowledge with machine learning!

---

**Happy Processing!** 📝🤖
