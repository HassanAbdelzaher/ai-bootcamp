# Step 13: Graph Neural Networks (GNNs)

> **Learn to work with graph-structured data using neural networks**

---

## 📖 Overview

### What You'll Learn

- **Graph Basics**: Understanding nodes, edges, and graph structure
- **Graph Representation**: Adjacency matrices and node features
- **Graph Convolution**: How GNNs aggregate information from neighbors
- **GCN Architecture**: Graph Convolutional Network implementation
- **Node Classification**: Classifying nodes in a graph
- **Graph Embedding**: Representing entire graphs as vectors
- **GNN Applications**: Real-world uses of graph neural networks

### Why This Matters

Many real-world problems involve **structured data** with relationships:
- **Social networks**: Friends, followers, connections
- **Molecules**: Atoms and chemical bonds
- **Knowledge graphs**: Entities and relationships
- **Recommendation systems**: Users and items
- **Citation networks**: Papers and citations

Traditional neural networks (CNNs, RNNs) work on:
- **Grids** (images)
- **Sequences** (text, time series)

But they **can't handle graphs** directly!

**Graph Neural Networks** extend neural networks to work with graph-structured data.

---

## 🎯 Learning Objectives

By the end of this step, you'll be able to:

1. ✅ Understand graph structure and representation
2. ✅ Implement Graph Convolutional Network (GCN) layers
3. ✅ Perform node classification on graphs
4. ✅ Create graph-level embeddings
5. ✅ Apply GNNs to real-world problems

---

## 📚 Concepts

### 1. What is a Graph?

A **graph** consists of:
- **Nodes (Vertices)**: Entities (e.g., people, molecules, documents)
- **Edges**: Relationships between nodes (e.g., friendships, bonds, citations)

#### Example: Social Network
```
Nodes: People (Alice, Bob, Charlie)
Edges: Friendships
  - Alice ↔ Bob (friends)
  - Bob ↔ Charlie (friends)
```

### 2. Graph Representation

#### Adjacency Matrix (A)

**Definition**: Matrix where `A[i,j] = 1` if nodes i and j are connected, else 0

```
     A  B  C
A  [ 0  1  0 ]
B  [ 1  0  1 ]
C  [ 0  1  0 ]
```

- **Symmetric**: Undirected graphs (A[i,j] = A[j,i])
- **Sparse**: Most entries are 0 (few connections)

#### Node Features (X)

Each node has a **feature vector**:
- Social network: Age, interests, location
- Molecules: Atom type, charge
- Documents: Word embeddings

Shape: `(num_nodes, num_features)`

### 3. Graph Convolution

**Key Idea**: Aggregate information from neighboring nodes

#### Simple Graph Convolution

```
H = AX
```

Where:
- `A`: Adjacency matrix
- `X`: Node features
- `H`: Updated features (aggregated from neighbors)

**Intuition**: Each node's new features = sum of its neighbors' features

#### Normalized Graph Convolution (GCN)

```
H = σ(D^(-1/2) A D^(-1/2) XW)
```

Where:
- `D`: Degree matrix (diagonal, sum of connections per node)
- `W`: Learnable weight matrix
- `σ`: Activation function (ReLU)

**Why normalize?**
- Prevents features from growing too large
- Accounts for nodes with many connections
- Makes training more stable

### 4. Graph Convolutional Network (GCN)

**Architecture**:
```
Input Features → GCN Layer 1 → GCN Layer 2 → ... → Output
```

Each GCN layer:
1. **Linear transformation**: `XW` (learn features)
2. **Normalize adjacency**: `D^(-1/2) A D^(-1/2)`
3. **Aggregate neighbors**: Multiply normalized A with features
4. **Activate**: Apply ReLU

**Multi-layer GCN**:
- Layer 1: Aggregates 1-hop neighbors (direct connections)
- Layer 2: Aggregates 2-hop neighbors (neighbors of neighbors)
- Layer K: Aggregates K-hop neighbors

### 5. Node Classification

**Task**: Predict label for each node

**Example**: Classify people in social network
- Input: Graph structure + node features
- Output: Class label for each node (e.g., interests, communities)

**Training**:
- Use labeled nodes for training
- Predict labels for unlabeled nodes
- Semi-supervised learning (few labels, many nodes)

### 6. Graph Embedding

**Task**: Represent entire graph as a single vector

**Methods**:
1. **Mean Pooling**: Average all node features
2. **Max Pooling**: Take maximum across nodes
3. **Sum Pooling**: Sum all node features
4. **Attention Pooling**: Weighted average (learned)

**Use Cases**:
- Classify entire graphs (e.g., molecule properties)
- Compare graphs
- Graph similarity

---

## 💻 Code Walkthrough

### Part 1: Creating a Graph

```python
import networkx as nx

# Create graph
G = nx.Graph()
G.add_nodes_from(range(10))  # 10 nodes
G.add_edges_from([(0,1), (1,2), ...])  # Edges

# Adjacency matrix
A = nx.adjacency_matrix(G).toarray()
```

**Explanation**:
- NetworkX: Python library for graphs
- Adjacency matrix: Represents connections
- Shape: `(num_nodes, num_nodes)`

### Part 2: Node Features

```python
# Each node has features
num_features = 5
X = np.random.randn(num_nodes, num_features)
```

**Explanation**:
- Each node has a feature vector
- Shape: `(num_nodes, num_features)`
- Features can be anything (age, interests, etc.)

### Part 3: GCN Layer

```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        self.weight = nn.Parameter(...)
    
    def forward(self, X, A):
        # Normalize adjacency
        A_normalized = normalize(A)
        
        # Transform features
        support = X @ self.weight
        
        # Aggregate neighbors
        output = A_normalized @ support
        
        return F.relu(output)
```

**Explanation**:
- `weight`: Learnable parameters
- Normalize: Prevent feature explosion
- Aggregate: Combine neighbor features
- ReLU: Non-linearity

### Part 4: Node Classification

```python
# Create model
model = GCN(num_features=5, hidden_dim=16, num_classes=3)

# Train
for epoch in range(epochs):
    output = model(X, A)
    loss = criterion(output[train_mask], y_train)
    loss.backward()
    optimizer.step()
```

**Explanation**:
- Model: Multi-layer GCN
- Training: Only on labeled nodes
- Prediction: All nodes get labels

---

## 📊 Expected Output

When you run `step_13_graph_neural_networks.py`, you'll see:

```
======================================================================
Step 13: Graph Neural Networks (GNNs)
======================================================================

======================================================================
Part 1: Understanding Graphs
======================================================================

Creating a simple graph...
Graph = Nodes (vertices) + Edges (connections)

Graph created:
  Nodes: 10
  Edges: 12

======================================================================
Part 2: Graph Representation
======================================================================

Adjacency Matrix (A):
  Shape: (10, 10)
  A[i,j] = 1 if nodes i and j are connected

Node Features (X):
  Shape: (10, 5) (nodes × features)
  Each node has 5 features

======================================================================
Part 6: Node Classification Task
======================================================================

Training GCN for node classification...
  Epoch 20/100: Loss = 0.8234
  Epoch 40/100: Loss = 0.6543
  ...
  Epoch 100/100: Loss = 0.4321

Test Accuracy: 85.00%
```

**Visualizations Generated**:
- `graph_neural_network.png`: Graph structure with node labels

---

## 🎓 Exercises

### Exercise 1: Add More Layers

Modify the GCN to have 3 layers instead of 2. How does this affect:
- Training time?
- Accuracy?
- Overfitting?

### Exercise 2: Different Pooling Methods

Implement different graph pooling methods:
- Max pooling
- Sum pooling
- Attention pooling

Compare their performance.

### Exercise 3: Real-World Graph

Create a graph from real data:
- Citation network (papers citing papers)
- Social network (friends on social media)
- Co-authorship network

Apply GCN for node classification.

---

## 🔍 Key Takeaways

1. **Graphs Represent Relationships**: Nodes and edges capture structure

2. **GNNs Aggregate Neighbor Information**: Each node learns from its neighbors

3. **Multi-Layer GNNs See Further**: More layers = information from more distant nodes

4. **Normalization is Important**: Prevents features from exploding

5. **Graph Embeddings Summarize Graphs**: Single vector represents entire graph

6. **GNNs are Versatile**: Work for nodes, edges, and graph-level tasks

---

## 📚 Further Reading

- **GCN Paper**: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- **GNN Survey**: [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)
- **PyTorch Geometric**: [Library for GNNs](https://pytorch-geometric.readthedocs.io/)
- **Graph Theory**: [Introduction to Graph Theory](https://en.wikipedia.org/wiki/Graph_theory)

---

## ✅ Checklist

Before moving on, make sure you can:

- [ ] Explain what a graph is (nodes, edges)
- [ ] Create an adjacency matrix
- [ ] Implement a GCN layer
- [ ] Perform node classification
- [ ] Create graph embeddings
- [ ] Understand GNN applications

---

## 🚀 Advanced Topics

Once you're comfortable with basics, explore:

1. **Attention Mechanisms**: Graph Attention Networks (GAT)
2. **Message Passing**: General message passing framework
3. **Graph Pooling**: Advanced pooling methods
4. **Dynamic Graphs**: Graphs that change over time
5. **Heterogeneous Graphs**: Different node/edge types

---

**Next Steps**: You've completed Graph Neural Networks! Consider exploring:
- Model deployment (Step 9)
- Unsupervised learning (Step 10)
- Advanced architectures

---

**Congratulations!** You now understand how to work with graph-structured data! 🎉
