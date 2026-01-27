"""
Step 13: Graph Neural Networks (GNNs)
Learn to work with graph-structured data using neural networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_learning_curve

print("=" * 70)
print("Step 13: Graph Neural Networks (GNNs)")
print("=" * 70)
print()

# ============================================================================
# Part 1: Understanding Graphs
# ============================================================================
print("=" * 70)
print("Part 1: Understanding Graphs")
print("=" * 70)
print()

# Create a simple graph
print("Creating a simple graph...")
print("Graph = Nodes (vertices) + Edges (connections)")
print()

# Example: Social network graph
# Nodes: People
# Edges: Friendships
num_nodes = 10
edges = [
    (0, 1), (0, 2), (1, 2), (1, 3),
    (2, 4), (3, 4), (3, 5), (4, 6),
    (5, 6), (6, 7), (7, 8), (8, 9)
]

# Create graph using NetworkX
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from(edges)

print(f"Graph created:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print()

# Visualize graph
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=500, font_size=10, font_weight='bold')
plt.title('Graph Structure', fontweight='bold', fontsize=14)
print("Graph visualization created")
print()

# ============================================================================
# Part 2: Graph Representation
# ============================================================================
print("=" * 70)
print("Part 2: Graph Representation")
print("=" * 70)
print()

# Adjacency Matrix: A[i,j] = 1 if edge exists, 0 otherwise
adjacency_matrix = nx.adjacency_matrix(G).toarray()
print("Adjacency Matrix (A):")
print("  Shape:", adjacency_matrix.shape)
print("  A[i,j] = 1 if nodes i and j are connected")
print(f"  Example: A[0,1] = {adjacency_matrix[0,1]} (nodes 0 and 1 connected)")
print(f"  Example: A[0,3] = {adjacency_matrix[0,3]} (nodes 0 and 3 not connected)")
print()

# Node Features: Each node has features (e.g., age, interests)
# For simplicity, create random features
num_features = 5
node_features = np.random.randn(num_nodes, num_features)
print(f"Node Features (X):")
print(f"  Shape: {node_features.shape} (nodes × features)")
print(f"  Each node has {num_features} features")
print()

# Convert to PyTorch tensors
A = torch.FloatTensor(adjacency_matrix)
X = torch.FloatTensor(node_features)

print("Converted to PyTorch tensors:")
print(f"  A (adjacency): {A.shape}")
print(f"  X (features): {X.shape}")
print()

# ============================================================================
# Part 3: Simple Graph Convolution
# ============================================================================
print("=" * 70)
print("Part 3: Simple Graph Convolution")
print("=" * 70)
print()

def simple_graph_conv(X, A):
    """
    Simple Graph Convolution
    Formula: H = AXW (simplified)
    Where:
      A: Adjacency matrix
      X: Node features
      W: Learnable weights (identity for now)
    """
    # Multiply adjacency matrix with features
    # This aggregates features from neighboring nodes
    # AX: For each node, sum features of its neighbors
    H = torch.matmul(A, X)
    return H

# Apply simple convolution
H = simple_graph_conv(X, A)
print("Simple Graph Convolution:")
print(f"  Input features X: {X.shape}")
print(f"  Output features H: {H.shape}")
print("  Each node's features now include information from neighbors")
print()

# ============================================================================
# Part 4: Graph Convolutional Network (GCN) Layer
# ============================================================================
print("=" * 70)
print("Part 4: Graph Convolutional Network (GCN) Layer")
print("=" * 70)
print()

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer
    Implements: H = σ(D^(-1/2) A D^(-1/2) X W)
    Where:
      D: Degree matrix (diagonal)
      A: Adjacency matrix
      X: Node features
      W: Learnable weight matrix
      σ: Activation function
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # Weight matrix: transforms features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, X, A):
        """
        Forward pass through GCN layer
        X: Node features (num_nodes, in_features)
        A: Adjacency matrix (num_nodes, num_nodes)
        """
        # Step 1: Linear transformation
        # XW: Transform features
        support = torch.matmul(X, self.weight)
        
        # Step 2: Normalize adjacency matrix
        # Add self-loops (A + I)
        A_hat = A + torch.eye(A.size(0))
        
        # Calculate degree matrix (sum of each row)
        degree = torch.sum(A_hat, dim=1)
        # Avoid division by zero
        degree = torch.clamp(degree, min=1.0)
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # Normalize: D^(-1/2) A D^(-1/2)
        A_normalized = torch.matmul(torch.matmul(degree_inv_sqrt, A_hat), degree_inv_sqrt)
        
        # Step 3: Aggregate neighbor features
        # D^(-1/2) A D^(-1/2) XW
        output = torch.matmul(A_normalized, support)
        
        # Step 4: Apply activation
        output = F.relu(output)
        
        return output

# Test GCN layer
print("Creating GCN Layer...")
gcn_layer = GCNLayer(in_features=5, out_features=16)
output = gcn_layer(X, A)
print(f"  Input: {X.shape}")
print(f"  Output: {output.shape}")
print("  Each node now has 16-dimensional features from neighbors")
print()

# ============================================================================
# Part 5: Complete GCN Model
# ============================================================================
print("=" * 70)
print("Part 5: Complete GCN Model")
print("=" * 70)
print()

class GCN(nn.Module):
    """
    Graph Convolutional Network
    Multi-layer GCN for node classification
    """
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        # Layer 1: Input features → Hidden features
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        # Layer 2: Hidden features → Output classes
        self.gcn2 = GCNLayer(hidden_dim, num_classes)
    
    def forward(self, X, A):
        """
        Forward pass
        X: Node features
        A: Adjacency matrix
        """
        # First GCN layer
        h1 = self.gcn1(X, A)
        # Second GCN layer
        h2 = self.gcn2(h1, A)
        # Softmax for classification
        output = F.log_softmax(h2, dim=1)
        return output

print("GCN Model created")
print("  Architecture: Input → GCN Layer 1 → GCN Layer 2 → Output")
print()

# ============================================================================
# Part 6: Node Classification Task
# ============================================================================
print("=" * 70)
print("Part 6: Node Classification Task")
print("=" * 70)
print()

# Create node labels (e.g., classify nodes into categories)
# Example: Classify people into groups (0, 1, or 2)
np.random.seed(42)
node_labels = np.random.randint(0, 3, num_nodes)
print(f"Node Labels: {node_labels}")
print(f"  Class 0: {np.sum(node_labels == 0)} nodes")
print(f"  Class 1: {np.sum(node_labels == 1)} nodes")
print(f"  Class 2: {np.sum(node_labels == 2)} nodes")
print()

# Split into train/test
train_mask = np.random.choice(num_nodes, size=int(0.7 * num_nodes), replace=False)
test_mask = np.setdiff1d(range(num_nodes), train_mask)

y_train = torch.LongTensor(node_labels[train_mask])
y_test = torch.LongTensor(node_labels[test_mask])

print(f"Training nodes: {len(train_mask)}")
print(f"Test nodes: {len(test_mask)}")
print()

# Create model
model = GCN(num_features=5, hidden_dim=16, num_classes=3)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
print("Training GCN for node classification...")
epochs = 100
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(X, A)
    
    # Get predictions for training nodes only
    train_output = output[train_mask]
    
    # Calculate loss
    loss = criterion(train_output, y_train)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")

print()

# Evaluation
model.eval()
with torch.no_grad():
    output = model(X, A)
    predictions = torch.argmax(output[test_mask], dim=1)
    accuracy = accuracy_score(y_test.numpy(), predictions.numpy())

print(f"Test Accuracy: {accuracy:.2%}")
print()

# ============================================================================
# Part 7: Graph Embedding
# ============================================================================
print("=" * 70)
print("Part 7: Graph Embedding")
print("=" * 70)
print()

# Graph embedding: Represent entire graph as a vector
# Useful for graph-level tasks (e.g., classify entire graphs)

class GraphEmbedding(nn.Module):
    """
    Graph-level embedding
    Aggregates node features to create graph representation
    """
    def __init__(self, num_features, hidden_dim, embedding_dim):
        super(GraphEmbedding, self).__init__()
        # GCN layers for node features
        self.gcn1 = GCNLayer(num_features, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        # Graph pooling: Aggregate all nodes
        self.pool = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, X, A):
        """
        Forward pass
        Returns: Graph embedding vector
        """
        # Get node embeddings
        h1 = self.gcn1(X, A)
        h2 = self.gcn2(h1, A)
        
        # Graph-level pooling: Mean pooling
        # Average all node features to get graph representation
        graph_embedding = torch.mean(h2, dim=0)  # Average over nodes
        
        # Transform to embedding dimension
        graph_embedding = self.pool(graph_embedding)
        
        return graph_embedding

print("Creating Graph Embedding Model...")
embedding_model = GraphEmbedding(num_features=5, hidden_dim=16, embedding_dim=8)

# Get graph embedding
with torch.no_grad():
    graph_emb = embedding_model(X, A)

print(f"Graph Embedding:")
print(f"  Input: Graph with {num_nodes} nodes, {num_features} features each")
print(f"  Output: Single vector of size {graph_emb.shape}")
print("  This vector represents the entire graph")
print()

# ============================================================================
# Part 8: Visualization
# ============================================================================
print("=" * 70)
print("Part 8: Visualizing Graph Neural Networks")
print("=" * 70)
print()

# Visualize node embeddings
model.eval()
with torch.no_grad():
    # Get node embeddings from first GCN layer
    h1 = model.gcn1(X, A)
    
    # Use t-SNE or PCA for 2D visualization (simplified: use first 2 dimensions)
    node_emb_2d = h1[:, :2].numpy()

# Plot graph with node colors based on labels
plt.subplot(1, 2, 2)
colors_map = ['red', 'green', 'blue']
node_colors = [colors_map[label] for label in node_labels]
nx.draw(G, pos, with_labels=True, node_color=node_colors,
        node_size=500, font_size=10, font_weight='bold')
plt.title('Graph with Node Labels', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('graph_neural_network.png', dpi=150, bbox_inches='tight')
print("Saved: graph_neural_network.png")
print()

# Plot training loss
plot_learning_curve(losses, title="GCN Training Loss")

# ============================================================================
# Part 9: Applications of GNNs
# ============================================================================
print("=" * 70)
print("Part 9: Applications of Graph Neural Networks")
print("=" * 70)
print()

print("GNNs are used in many real-world applications:")
print()
print("1. SOCIAL NETWORKS")
print("   - Friend recommendation")
print("   - Community detection")
print("   - Influence prediction")
print()

print("2. MOLECULAR ANALYSIS")
print("   - Drug discovery")
print("   - Property prediction")
print("   - Chemical reaction prediction")
print()

print("3. RECOMMENDATION SYSTEMS")
print("   - User-item graphs")
print("   - Collaborative filtering")
print("   - Product recommendations")
print()

print("4. KNOWLEDGE GRAPHS")
print("   - Question answering")
print("   - Entity linking")
print("   - Relation extraction")
print()

print("5. COMPUTER VISION")
print("   - Scene graphs")
print("   - Object relationships")
print("   - Image understanding")
print()

print("6. NATURAL LANGUAGE PROCESSING")
print("   - Dependency parsing")
print("   - Semantic graphs")
print("   - Document understanding")
print()

print("=" * 70)
print("Step 13 Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Understood graph structure and representation")
print("  ✅ Implemented Graph Convolutional Network (GCN)")
print("  ✅ Performed node classification")
print("  ✅ Created graph embeddings")
print("  ✅ Visualized graph neural networks")
print("  ✅ Learned GNN applications")
print()
print("Key Takeaways:")
print("  • Graphs represent relationships between entities")
print("  • GNNs aggregate information from neighbors")
print("  • GCN layers learn node representations")
print("  • Graph embeddings represent entire graphs")
print("  • GNNs are powerful for structured data")
print()
