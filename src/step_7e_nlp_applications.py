"""
Step 7e — NLP Applications
Goal: Learn practical NLP techniques: tokenization, embeddings, NER, sentiment analysis
Tools: Python + NumPy + PyTorch + Matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

from plotting import plot_learning_curve

print("=" * 70)
print("Step 7e: NLP Applications")
print("=" * 70)
print()
print("Goal: Learn practical NLP techniques for real-world applications")
print()

# ============================================================================
# 7e.1 What is NLP?
# ============================================================================
print("=== 7e.1 What is Natural Language Processing? ===")
print()
print("NLP enables computers to understand, interpret, and generate human language.")
print()
print("Key NLP Tasks:")
print("  • Tokenization: Breaking text into words/tokens")
print("  • Word Embeddings: Representing words as vectors")
print("  • Named Entity Recognition: Finding people, places, organizations")
print("  • Sentiment Analysis: Determining positive/negative emotions")
print("  • Text Classification: Categorizing documents")
print("  • Machine Translation: Translating between languages")
print()

# ============================================================================
# 7e.2 Tokenization
# ============================================================================
print("=== 7e.2 Tokenization ===")
print()
print("Tokenization: Breaking text into smaller units (tokens)")
print()

# Sample text
text = "Hello world! How are you? I'm learning NLP."

print(f"Original text: '{text}'")
print()

# Character-level tokenization
char_tokens = list(text)
print("Character-level tokens:")
print(f"  {char_tokens[:20]}... (showing first 20)")
print(f"  Total characters: {len(char_tokens)}")
print()

# Word-level tokenization (simple)
word_tokens = text.split()
print("Word-level tokens (simple split):")
print(f"  {word_tokens}")
print(f"  Total words: {len(word_tokens)}")
print()

# Better word-level tokenization (handling punctuation)
import re
def simple_tokenize(text):
    """Simple word tokenization"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation (keep apostrophes for contractions)
    text = re.sub(r'[^\w\s\']', ' ', text)
    # Split on whitespace
    tokens = text.split()
    return tokens

better_tokens = simple_tokenize(text)
print("Word-level tokens (improved):")
print(f"  {better_tokens}")
print(f"  Total words: {len(better_tokens)}")
print()

# Subword tokenization (conceptual)
print("Subword tokenization (conceptual):")
print("  • Breaks words into smaller pieces")
print("  • Handles unknown words better")
print("  • Used in BERT, GPT")
print("  • Example: 'learning' → ['learn', '##ing']")
print()

# Create vocabulary
vocab = {}
for token in better_tokens:
    if token not in vocab:
        vocab[token] = len(vocab)

print(f"Vocabulary size: {len(vocab)}")
print(f"Vocabulary: {list(vocab.keys())}")
print()

# ============================================================================
# 7e.3 Word Embeddings
# ============================================================================
print("=== 7e.3 Word Embeddings ===")
print()
print("Word Embeddings: Represent words as dense vectors")
print("  • Similar words have similar vectors")
print("  • Captures semantic relationships")
print("  • Enables neural networks to process text")
print()

# Simple one-hot encoding
def one_hot_encode(word, vocab):
    """One-hot encode a word"""
    vec = np.zeros(len(vocab))
    if word in vocab:
        vec[vocab[word]] = 1
    return vec

print("One-Hot Encoding (sparse):")
print("  • Each word = vector of zeros with one 1")
print("  • Vocabulary size = vector dimension")
print("  • No semantic relationships")
print()

# Example: Simple word embeddings (learned)
print("Learned Embeddings (dense vectors):")
print("  • Words represented as dense vectors (e.g., 100 dimensions)")
print("  • Similar words are close in vector space")
print("  • Learned during training")
print()

# Create simple embedding layer
vocab_size = len(vocab)
embedding_dim = 8  # Small for demonstration

embedding = nn.Embedding(vocab_size, embedding_dim)

# Convert tokens to indices
token_indices = [vocab[token] for token in better_tokens if token in vocab]
if token_indices:
    token_tensor = torch.LongTensor(token_indices[:5])  # First 5 tokens
    embedded = embedding(token_tensor)
    
    print(f"Token indices: {token_indices[:5]}")
    print(f"Embedded vectors shape: {embedded.shape}")
    print(f"Sample embedding for '{better_tokens[0]}': {embedded[0].detach().numpy()}")
    print()

# Visualize embeddings (2D projection)
if len(better_tokens) >= 3:
    # Get embeddings for first few words
    word_indices = [vocab[w] for w in better_tokens[:min(5, len(better_tokens))] if w in vocab]
    if word_indices:
        word_tensor = torch.LongTensor(word_indices)
        word_embeddings = embedding(word_tensor).detach().numpy()
        
        # Project to 2D using PCA (simplified)
        # Simple 2D projection: use first 2 dimensions
        embeddings_2d = word_embeddings[:, :2]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        words_to_plot = [w for w in better_tokens[:min(5, len(better_tokens))] if w in vocab]
        
        for i, word in enumerate(words_to_plot):
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=200, alpha=0.7, edgecolors='black')
            ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], word, 
                   fontsize=12, fontweight='bold', ha='center', va='bottom')
        
        ax.set_xlabel('Embedding Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Embedding Dimension 2', fontsize=12, fontweight='bold')
        ax.set_title('Word Embeddings (2D Projection)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

print("Popular Embedding Methods:")
print("  • Word2Vec: Learn embeddings from word co-occurrence")
print("  • GloVe: Global vectors from word-word co-occurrence matrix")
print("  • FastText: Handles subwords and rare words")
print("  • Contextual embeddings: BERT, GPT (different for each context)")
print()

# ============================================================================
# 7e.4 Named Entity Recognition (NER)
# ============================================================================
print("=== 7e.4 Named Entity Recognition (NER) ===")
print()
print("NER: Identify and classify named entities in text")
print("  • Person names: 'John Smith', 'Mary'")
print("  • Organizations: 'Apple Inc.', 'MIT'")
print("  • Locations: 'New York', 'Paris'")
print("  • Dates: 'January 2024', 'Monday'")
print("  • Money: '$100', '50 euros'")
print()

# Sample text with entities
ner_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976. Tim Cook is the current CEO."

print(f"Text: '{ner_text}'")
print()

# Simple rule-based NER (for demonstration)
def simple_ner(text):
    """Simple rule-based NER (for demonstration only)"""
    entities = []
    
    # Capitalized words might be entities
    words = text.split()
    for i, word in enumerate(words):
        # Remove punctuation
        clean_word = word.strip('.,!?;:')
        
        # Simple heuristics
        if clean_word[0].isupper() and len(clean_word) > 1:
            # Check if it's a known organization
            if clean_word.lower() in ['apple', 'microsoft', 'google', 'amazon']:
                entities.append((clean_word, 'ORGANIZATION', i))
            # Check if it's a known person
            elif clean_word.lower() in ['steve', 'jobs', 'tim', 'cook']:
                entities.append((clean_word, 'PERSON', i))
            # Check if it's a location (simple)
            elif clean_word.lower() in ['cupertino', 'california', 'new york', 'paris']:
                entities.append((clean_word, 'LOCATION', i))
            # Check if it's a year
            elif clean_word.isdigit() and len(clean_word) == 4:
                entities.append((clean_word, 'DATE', i))
    
    return entities

entities = simple_ner(ner_text)
print("Detected Entities:")
for entity, label, pos in entities:
    print(f"  {entity} → {label}")
print()

print("Real NER uses:")
print("  • Machine learning models (LSTM, BERT)")
print("  • BIO tagging (B-begin, I-inside, O-outside)")
print("  • Sequence labeling")
print()

# ============================================================================
# 7e.5 Sentiment Analysis
# ============================================================================
print("=== 7e.5 Sentiment Analysis ===")
print()
print("Sentiment Analysis: Determine emotional tone of text")
print("  • Positive: 'I love this product!'")
print("  • Negative: 'This is terrible.'")
print("  • Neutral: 'The weather is sunny today.'")
print()

# Create sentiment dataset
sentences = [
    "I love this movie! It's amazing!",
    "This is the worst film I've ever seen.",
    "The weather is nice today.",
    "I'm so happy with my purchase!",
    "This product is terrible and broken.",
    "The book was okay, nothing special.",
]

labels = [1, 0, 2, 1, 0, 2]  # 1=positive, 0=negative, 2=neutral

print("Sentiment Dataset:")
for i, (sent, label) in enumerate(zip(sentences, labels)):
    label_name = ['Negative', 'Positive', 'Neutral'][label]
    print(f"  {i+1}. '{sent}' → {label_name}")
print()

# Simple sentiment classifier
class SimpleSentimentClassifier(nn.Module):
    """Simple sentiment classifier using embeddings"""
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Use last hidden state
        output = self.fc(hidden[-1])
        return output

# Build vocabulary from sentences
sentiment_vocab = {}
for sent in sentences:
    tokens = simple_tokenize(sent)
    for token in tokens:
        if token not in sentiment_vocab:
            sentiment_vocab[token] = len(sentiment_vocab)

# Add padding token
sentiment_vocab['<PAD>'] = len(sentiment_vocab)
vocab_size_sent = len(sentiment_vocab)

print(f"Sentiment vocabulary size: {vocab_size_sent}")
print()

# Convert sentences to sequences
def text_to_sequence(text, vocab, max_len=10):
    """Convert text to sequence of token indices"""
    tokens = simple_tokenize(text)
    seq = [vocab.get(token, vocab['<PAD>']) for token in tokens]
    # Pad or truncate
    if len(seq) < max_len:
        seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# Prepare data
max_len = 10
X_sent = torch.LongTensor([text_to_sequence(sent, sentiment_vocab, max_len) for sent in sentences])
y_sent = torch.LongTensor(labels)

print("Training sentiment classifier...")
model_sent = SimpleSentimentClassifier(vocab_size_sent, embedding_dim=16, hidden_dim=32, num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_sent.parameters(), lr=0.01)

losses_sent = []
epochs_sent = 200

for epoch in range(epochs_sent):
    optimizer.zero_grad()
    outputs = model_sent(X_sent)
    loss = criterion(outputs, y_sent)
    loss.backward()
    optimizer.step()
    losses_sent.append(loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{epochs_sent}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# Evaluate
model_sent.eval()
with torch.no_grad():
    predictions = model_sent(X_sent)
    predicted_labels = torch.argmax(predictions, dim=1)

print("Sentiment Predictions:")
for i, (sent, true_label, pred_label) in enumerate(zip(sentences, labels, predicted_labels)):
    true_name = ['Negative', 'Positive', 'Neutral'][true_label]
    pred_name = ['Negative', 'Positive', 'Neutral'][pred_label.item()]
    match = "✓" if true_label == pred_label.item() else "✗"
    print(f"  '{sent[:40]}...'")
    print(f"    True: {true_name}, Predicted: {pred_name} {match}")
print()

# Visualize training
plot_learning_curve(losses_sent, 
                   title="Sentiment Classifier Training",
                   ylabel="Loss (Cross-Entropy)")

# ============================================================================
# 7e.6 Text Classification Pipeline
# ============================================================================
print("=== 7e.6 Text Classification Pipeline ===")
print()
print("Complete pipeline for text classification:")
print()
print("1. Data Preprocessing:")
print("   • Lowercasing")
print("   • Remove punctuation")
print("   • Handle special characters")
print()
print("2. Tokenization:")
print("   • Split text into tokens")
print("   • Handle subwords if needed")
print()
print("3. Vocabulary Building:")
print("   • Create word-to-index mapping")
print("   • Handle unknown words")
print()
print("4. Sequence Encoding:")
print("   • Convert tokens to indices")
print("   • Padding/truncation")
print()
print("5. Embedding:")
print("   • Convert indices to dense vectors")
print("   • Use pre-trained or learn embeddings")
print()
print("6. Model Training:")
print("   • RNN/LSTM/Transformer")
print("   • Classification head")
print()
print("7. Evaluation:")
print("   • Accuracy, Precision, Recall")
print("   • Confusion matrix")
print()

# ============================================================================
# 7e.7 Sequence-to-Sequence Models
# ============================================================================
print("=== 7e.7 Sequence-to-Sequence Models ===")
print()
print("Seq2Seq: Convert one sequence to another")
print("  • Machine translation: English → French")
print("  • Text summarization: Long text → Summary")
print("  • Question answering: Question → Answer")
print()

print("Architecture:")
print("  Encoder:")
print("    • Processes input sequence")
print("    • Creates context vector")
print("    • RNN/LSTM/Transformer")
print()
print("  Decoder:")
print("    • Generates output sequence")
print("    • Uses context from encoder")
print("    • Autoregressive (generates one token at a time)")
print()

# Simple seq2seq conceptual example
class SimpleSeq2Seq(nn.Module):
    """Simple Seq2Seq model (conceptual)"""
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def encode(self, x):
        """Encode input sequence"""
        embedded = self.embedding(x)
        _, (hidden, cell) = self.encoder(embedded)
        return hidden, cell
    
    def decode(self, hidden, cell, target_length):
        """Decode to output sequence"""
        # Simplified: would need proper decoding loop
        # This is just conceptual
        return None

print("Seq2Seq Training:")
print("  • Teacher forcing: Use ground truth during training")
print("  • Inference: Use previous predictions")
print("  • Attention: Focus on relevant parts of input")
print()

# ============================================================================
# 7e.8 Real-World NLP Applications
# ============================================================================
print("=== 7e.8 Real-World NLP Applications ===")
print()
print("NLP powers many applications:")
print()
print("💬 Chatbots & Virtual Assistants:")
print("   • Customer service")
print("   • Siri, Alexa, Google Assistant")
print("   • Conversational AI")
print()
print("📧 Email & Text Processing:")
print("   • Spam detection")
print("   • Auto-categorization")
print("   • Smart replies")
print()
print("🌐 Search & Information Retrieval:")
print("   • Google Search")
print("   • Document search")
print("   • Question answering")
print()
print("📰 Content Analysis:")
print("   • News categorization")
print("   • Social media monitoring")
print("   • Trend analysis")
print()
print("🌍 Translation:")
print("   • Google Translate")
print("   • Multilingual communication")
print("   • Real-time translation")
print()
print("📝 Text Generation:")
print("   • GPT models")
print("   • Content creation")
print("   • Code generation")
print()

# ============================================================================
# 7e.9 NLP Best Practices
# ============================================================================
print("=== 7e.9 NLP Best Practices ===")
print()
print("✅ Data Preprocessing:")
print("   • Clean and normalize text")
print("   • Handle special cases (URLs, emails)")
print("   • Consider language-specific rules")
print()
print("✅ Tokenization:")
print("   • Choose appropriate level (char/word/subword)")
print("   • Handle unknown words")
print("   • Preserve important information")
print()
print("✅ Embeddings:")
print("   • Use pre-trained embeddings when possible")
print("   • Fine-tune for your task")
print("   • Consider contextual embeddings (BERT)")
print()
print("✅ Model Selection:")
print("   • Simple tasks: RNN/LSTM")
print("   • Complex tasks: Transformers")
print("   • Consider computational budget")
print()
print("✅ Evaluation:")
print("   • Use appropriate metrics")
print("   • Consider class imbalance")
print("   • Test on diverse data")
print()

# ============================================================================
# 7e.10 Summary
# ============================================================================
print("=== 7e.10 Summary ===")
print()
print("✅ You've learned:")
print("  • Tokenization (char, word, subword)")
print("  • Word embeddings (one-hot, learned, pre-trained)")
print("  • Named Entity Recognition")
print("  • Sentiment analysis")
print("  • Text classification pipeline")
print("  • Sequence-to-sequence models")
print()
print("🎯 Key Takeaways:")
print("  1. Tokenization is the foundation")
print("  2. Embeddings capture semantic meaning")
print("  3. NLP models learn from text patterns")
print("  4. Pre-trained models are powerful")
print("  5. Pipeline matters: preprocessing → model → evaluation")
print()

print("=" * 70)
print("Step 7e Complete! You understand NLP applications!")
print("=" * 70)
