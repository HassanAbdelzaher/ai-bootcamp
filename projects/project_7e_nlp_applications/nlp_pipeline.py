"""
Project 7e: NLP Applications
Practical NLP: tokenization, embeddings, NER, sentiment analysis
"""

import numpy as np
import torch
import torch.nn as nn
import re
from collections import Counter
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_learning_curve

print("=" * 70)
print("Project 7e: NLP Applications")
print("=" * 70)
print()

# ============================================================================
# Step 1: Tokenization
# ============================================================================
print("=" * 70)
print("Step 1: Tokenization")
print("=" * 70)
print()

def character_tokenize(text):
    """Character-level tokenization"""
    # Convert text to lowercase for consistency
    # .lower(): Converts all characters to lowercase
    # list(...): Converts string to list of characters
    # Example: "Hello" → ['h', 'e', 'l', 'l', 'o']
    return list(text.lower())

def word_tokenize(text):
    """Word-level tokenization"""
    # Convert to lowercase for consistency
    # This ensures "Hello" and "hello" are treated as same word
    text = text.lower()
    
    # Remove punctuation (keep apostrophes for contractions)
    # re.sub(pattern, replacement, string): Replace pattern with replacement
    # r'[^\w\s\']': Regular expression pattern
    #   [^...]: Match anything NOT in brackets
    #   \w: Word characters (letters, digits, underscore)
    #   \s: Whitespace characters
    #   \': Apostrophe (for contractions like "I'm")
    #   So this matches anything that's NOT a word char, space, or apostrophe
    # ' ': Replace with space (removes punctuation)
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Split on whitespace to get words
    # .split(): Splits string into list of words (separated by spaces)
    # Removes multiple spaces and creates clean word list
    # Example: "hello world" → ['hello', 'world']
    return text.split()

def create_vocabulary(texts, min_freq=2):
    """Create vocabulary from texts"""
    # Counter: Counts occurrences of each word
    # word_counts: Dictionary mapping word → count
    word_counts = Counter()
    
    # Count words across all texts
    for text in texts:
        # Tokenize text into words
        words = word_tokenize(text)
        # Update counter with words from this text
        # word_counts.update(words): Increments count for each word
        word_counts.update(words)
    
    # Initialize vocabulary with special tokens
    # <PAD>: Padding token (for sequences of different lengths)
    # <UNK>: Unknown token (for words not in vocabulary)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add words to vocabulary
    idx = 2  # Start indexing from 2 (0 and 1 are special tokens)
    for word, count in word_counts.items():
        # Only add words that appear at least min_freq times
        # min_freq=2: Filter out rare words (reduces vocabulary size)
        # This helps with generalization (rare words → <UNK>)
        if count >= min_freq:
            # Assign unique index to word
            vocab[word] = idx
            idx += 1  # Increment for next word
    
    # Return vocabulary: dictionary mapping word → index
    return vocab

# Example texts
texts = [
    "I love machine learning",
    "Machine learning is fascinating",
    "I study AI and machine learning",
    "Python is great for AI",
    "AI will change the world"
]

print("Sample texts:")
for text in texts[:3]:
    print(f"  '{text}'")
print()

vocab = create_vocabulary(texts)
print(f"Vocabulary size: {len(vocab)}")
print(f"Sample vocabulary: {list(vocab.keys())[:10]}")
print()

# ============================================================================
# Step 2: Word Embeddings
# ============================================================================
print("=" * 70)
print("Step 2: Word Embeddings")
print("=" * 70)
print()

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, x):
        return self.embedding(x)

vocab_size = len(vocab)
embed_dim = 128
embedding_layer = EmbeddingLayer(vocab_size, embed_dim)

# Test embedding
word_idx = vocab.get('machine', vocab['<UNK>'])
embedding = embedding_layer(torch.LongTensor([[word_idx]]))
print(f"Embedding for 'machine': shape {embedding.shape}")
print()

# ============================================================================
# Step 3: Named Entity Recognition (NER)
# ============================================================================
print("=" * 70)
print("Step 3: Named Entity Recognition")
print("=" * 70)
print()

class NERModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_tags=5):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)  # *2 for bidirectional
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.classifier(lstm_out)
        return output

# NER tags: 0=PAD, 1=O, 2=B-PER, 3=I-PER, 4=B-LOC, 5=I-LOC
print("NER Model created")
print("  Tags: PAD, O (outside), B-PER (begin person), I-PER (inside person),")
print("        B-LOC (begin location), I-LOC (inside location)")
print()

# ============================================================================
# Step 4: Sentiment Analysis
# ============================================================================
print("=" * 70)
print("Step 4: Sentiment Analysis")
print("=" * 70)
print()

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=3):
        # super(): Call parent class constructor
        super(SentimentClassifier, self).__init__()
        
        # Embedding layer: Converts word indices to dense vectors
        # vocab_size: Number of words in vocabulary
        # embed_dim: Dimension of embedding vectors (128-dimensional)
        # Each word gets a learnable 128-dimensional vector
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer: Processes sequence of embeddings
        # embed_dim: Input dimension (embedding size)
        # hidden_dim: Hidden state dimension (256)
        # batch_first=True: Input format is (batch, seq_len, features)
        # LSTM processes sequence step-by-step, maintaining hidden state
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Classifier: Maps hidden state to class probabilities
        # hidden_dim: Input dimension (LSTM hidden state size)
        # num_classes: Number of output classes (3: Negative, Neutral, Positive)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch, seq_len) - word indices
        
        # Embed words: Convert indices to dense vectors
        # self.embedding(x): Looks up embedding for each word index
        # Result: (batch, seq_len, embed_dim) - sequence of embeddings
        embedded = self.embedding(x)
        
        # Process with LSTM
        # self.lstm(embedded): Processes sequence through LSTM
        # Returns: (lstm_out, (hidden, cell))
        #   lstm_out: Output at each time step (batch, seq_len, hidden_dim)
        #   hidden: Final hidden state (num_layers, batch, hidden_dim)
        #   cell: Final cell state (not used here)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state for classification
        # hidden[-1]: Last layer's hidden state (batch, hidden_dim)
        # This represents the entire sequence's meaning
        output = self.classifier(hidden[-1])
        
        # Return: (batch, num_classes) - class logits
        return output

# Create sentiment dataset
sentiment_texts = [
    "I love this product!",
    "This is terrible",
    "It's okay, nothing special",
    "Amazing quality!",
    "Very disappointed",
    "Good value for money"
]

sentiment_labels = [2, 0, 1, 2, 0, 1]  # 0=Negative, 1=Neutral, 2=Positive

# Create vocabulary and sequences
sentiment_vocab = create_vocabulary(sentiment_texts)
max_len = 10

def text_to_sequence(text, vocab, max_len):
    """Convert text to sequence of indices"""
    words = word_tokenize(text)
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad or truncate
    if len(indices) < max_len:
        indices += [vocab['<PAD>']] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    
    return indices

X_sentiment = np.array([text_to_sequence(text, sentiment_vocab, max_len) 
                        for text in sentiment_texts])
y_sentiment = np.array(sentiment_labels)

X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(
    X_sentiment, y_sentiment, test_size=0.3, random_state=42
)

X_train_tensor = torch.LongTensor(X_train_sent)
y_train_tensor = torch.LongTensor(y_train_sent)
X_test_tensor = torch.LongTensor(X_test_sent)
y_test_tensor = torch.LongTensor(y_test_sent)

# Train sentiment model
sentiment_model = SentimentClassifier(
    vocab_size=len(sentiment_vocab), 
    num_classes=3
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(sentiment_model.parameters(), lr=0.001)

print("Training Sentiment Classifier...")
epochs = 100
losses = []

for epoch in range(epochs):
    sentiment_model.train()
    optimizer.zero_grad()
    outputs = sentiment_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")

print()

# Evaluation
sentiment_model.eval()
with torch.no_grad():
    outputs = sentiment_model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    accuracy = (predictions == y_test_tensor).float().mean().item()

print(f"Sentiment Classification Accuracy: {accuracy:.2%}")
print()

# ============================================================================
# Step 5: Complete NLP Pipeline
# ============================================================================
print("=" * 70)
print("Step 5: Complete NLP Pipeline")
print("=" * 70)
print()

class NLPipeline:
    def __init__(self, vocab, model):
        self.vocab = vocab
        self.model = model
    
    def preprocess(self, text):
        """Preprocess text"""
        words = word_tokenize(text)
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad to max_len
        max_len = 10
        if len(indices) < max_len:
            indices += [self.vocab['<PAD>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        return torch.LongTensor([indices])
    
    def predict(self, text):
        """Make prediction"""
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.preprocess(text)
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1)
            return prediction.item()

# Test pipeline
pipeline = NLPipeline(sentiment_vocab, sentiment_model)
test_texts = ["I love this!", "This is bad", "It's okay"]

print("Testing NLP Pipeline:")
sentiment_names = ['Negative', 'Neutral', 'Positive']
for text in test_texts:
    result = pipeline.predict(text)
    print(f"  '{text}' → {sentiment_names[result]}")
print()

# Visualize training
plot_learning_curve(losses, title="Sentiment Analysis Training Loss")

print("=" * 70)
print("Project 7e Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Implemented tokenization (character, word)")
print("  ✅ Created word embeddings")
print("  ✅ Built NER model")
print("  ✅ Trained sentiment classifier")
print("  ✅ Created complete NLP pipeline")
print()
