# 🤖 Teen AI Bootcamp

> **Learn AI from scratch — Build neural networks step by step**  
> A hands-on, teen-friendly bootcamp that teaches artificial intelligence fundamentals through practical coding exercises.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

---

## 📚 Overview

This bootcamp takes you from zero to building neural networks using PyTorch. Each step builds on the previous one, teaching you the math, concepts, and code needed to understand how AI really works.

**No prior AI knowledge required!** Just basic Python and a willingness to learn.

---

## 🎯 What You'll Learn

- **Step 0**: Math foundations (vectors, dot products, weights, bias)
- **Step 1**: Linear regression and gradient descent
- **Step 2**: Perceptrons and decision-making
- **Step 3**: Logistic regression with probabilities
- **Step 4**: Multiple neurons and neural network layers
- **Step 5**: Hidden layers and solving XOR problem
- **Step 6**: PyTorch framework and professional AI development
- **Step 7**: RNNs (Recurrent Neural Networks) for sequences and text
  - **Step 7a**: Text Generator (character-level RNN)
  - **Step 7b**: Stock Price Prediction (time series)
  - **Step 7c**: LSTM and GRU (advanced RNNs)
  - **Step 7d**: Transformers (BERT, GPT)
- **Step 8**: CNNs (Convolutional Neural Networks) for images
  - **Step 8a**: Real Datasets (CIFAR-10, ImageNet)
  - **Step 8b**: Image Classifiers
  - **Step 8c**: Transfer Learning
  - **Step 8d**: Object Detection (YOLO, R-CNN)
  - **Step 8e**: Image Generation (GANs, VAEs)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Basic understanding of Python
- Make (for Unix/macOS) - Windows users can run commands directly

### Installation

1. **Clone or download this repository**
   ```bash
   cd ai-codecamp
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   make install
   ```
   This will:
   - Create a Python virtual environment (`venv/`)
   - Install all required dependencies
   - Keep your system Python clean

   Or manually:
   ```bash
   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   make test
   ```

**Note**: 
- The Makefile automatically uses the virtual environment for all commands (Unix/macOS)
- Windows users: Use the manual commands above or run scripts directly after activating venv
- If you run Python scripts directly, activate the venv first:
  ```bash
  # On macOS/Linux:
  source venv/bin/activate
  
  # On Windows:
  venv\Scripts\activate
  ```

### Running the Steps

**Run all steps:**
```bash
make run-all
```

**Run individual steps:**
```bash
make run-step-0    # Math Foundations
make run-step-1    # Linear Regression
make run-step-2    # Perceptron
make run-step-3    # Logistic Regression
make run-step-4    # Multiple Neurons
make run-step-5    # XOR and Hidden Layers
make run-step-6    # PyTorch
make run-step-7    # RNNs (Sequences)
make run-step-7a   # Text Generator
make run-step-7b   # Stock Price Prediction
make run-step-7c   # LSTM and GRU
make run-step-7d   # Transformers (BERT, GPT)
make run-step-8    # CNNs (Images)
make run-step-8a   # Real Datasets (CIFAR-10)
make run-step-8b   # Image Classifiers
make run-step-8c   # Transfer Learning
make run-step-8d   # Object Detection (YOLO, R-CNN)
make run-step-8e   # Image Generation (GANs, VAEs)
```

**Or run directly:**
```bash
cd src
python step_0_math_foundations.py
python step_1_linear_regression.py
# ... etc
```

---

## 📁 Project Structure

```
ai-codecamp/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Makefile                  # Build and run commands
├── docs/                     # Step documentation
│   ├── Step_0_Math_Foundations_for_AI.md
│   ├── Step_1_Linear_Regression.md
│   ├── Step_2_Perceptron.md
│   ├── Step_3_Logistic_Regression.md
│   ├── Step_4_Multiple_Neurons.md
│   ├── Step_5_XOR_and_Hidden_Layers.md
│   └── Step_6_PyTorch.md
└── src/                      # Source code
    ├── plotting.py           # Common plotting utilities
    ├── step_0_math_foundations.py
    ├── step_1_linear_regression.py
    ├── step_2_perceptron.py
    ├── step_3_logistic_regression.py
    ├── step_4_multiple_neurons.py
    ├── step_5_xor_and_hidden_layers.py
    └── step_6_pytorch.py
```

---

## 📖 Learning Path

### Step 0: Math Foundations
**Goal**: Understand the basic math behind AI  
**Concepts**: Vectors, dot products, weights, bias  
**Time**: ~30 minutes

### Step 1: Linear Regression
**Goal**: Learn how AI predicts numbers  
**Concepts**: Gradient descent, error minimization, learning curves  
**Time**: ~45 minutes

### Step 2: Perceptron
**Goal**: Build your first decision-making AI  
**Concepts**: Step function, decision boundaries, binary classification  
**Time**: ~45 minutes

### Step 3: Logistic Regression
**Goal**: Make decisions with confidence (probabilities)  
**Concepts**: Sigmoid function, binary cross-entropy loss  
**Time**: ~45 minutes

### Step 4: Multiple Neurons
**Goal**: Build a neural network layer  
**Concepts**: Matrix operations, multiple neurons, forward pass  
**Time**: ~60 minutes

### Step 5: XOR and Hidden Layers
**Goal**: Understand why deep learning exists  
**Concepts**: Hidden layers, backpropagation, non-linear problems  
**Time**: ~60 minutes

### Step 6: PyTorch
**Goal**: Use professional AI frameworks  
**Concepts**: PyTorch tensors, automatic gradients, neural network modules  
**Time**: ~60 minutes

### Step 7: RNNs (Recurrent Neural Networks)
**Goal**: Process sequences and text  
**Concepts**: RNN architecture, sequence processing, hidden states, text generation  
**Time**: ~60 minutes

**Extended Topics:**
- **Step 7a**: Text Generator - Build character-level text generator
- **Step 7b**: Stock Prices - Predict time series data
- **Step 7c**: LSTM & GRU - Advanced RNN architectures
- **Step 7d**: Transformers - BERT, GPT, and attention mechanism

### Step 8: CNNs (Convolutional Neural Networks)
**Goal**: Process images using CNNs  
**Concepts**: Convolution, pooling, feature detection, image classification  
**Time**: ~60 minutes

**Extended Topics:**
- **Step 8a**: Real Datasets - Train on CIFAR-10 and ImageNet
- **Step 8b**: Image Classifiers - Build improved classification models
- **Step 8c**: Transfer Learning - Use pre-trained models
- **Step 8d**: Object Detection - YOLO and R-CNN algorithms
- **Step 8e**: Image Generation - GANs and VAEs for creating images

---

## 🛠 Available Commands

```bash
make help          # Show all available commands
make venv          # Create virtual environment
make install       # Create venv and install dependencies
make install-dev   # Install with development tools
make clean         # Remove Python cache files and venv
make check         # Check Python syntax
make test          # Test imports
make run-all       # Run all steps
make run-step-N    # Run specific step (N = 0-8)
```

**Note**: All commands automatically use the virtual environment. No need to manually activate it when using Make commands.

---

## 📦 Dependencies

- **numpy** (≥1.21.0) - Numerical computing
- **matplotlib** (≥3.4.0) - Plotting and visualization
- **torch** (≥1.9.0) - PyTorch deep learning framework
- **torchvision** (≥0.10.0) - PyTorch vision utilities
- **torchaudio** (≥0.9.0) - PyTorch audio utilities

---

## 🎓 Features

- ✅ **Learn from scratch** - No black boxes, understand every line
- ✅ **Progressive difficulty** - Each step builds on the previous
- ✅ **Visual learning** - Plots and graphs to see what's happening
- ✅ **Hands-on coding** - Write and run real AI code
- ✅ **Common utilities** - Shared plotting functions for consistency
- ✅ **Professional tools** - End with PyTorch, the industry standard

---

## 📝 How to Use This Bootcamp

1. **Read the documentation** - Each step has a detailed markdown file in `docs/`
2. **Run the code** - Execute each step to see it in action
3. **Experiment** - Modify parameters, try different values
4. **Understand** - Don't just copy code, understand what it does
5. **Build** - Use what you learn to create your own projects

---

## 🎮 Gamified Learning (Optional)

This bootcamp includes a gamified mission system:
- Earn XP for completing steps
- Unlock badges for achievements
- Level up as you progress
- Boss fights (challenging projects)

See `src/Teen_AI_Bootcamp_Missions.md` and `src/Teen_AI_Bootcamp_Gamified_System.md` for details.

---

## 🐛 Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the `src/` directory or have the correct Python path:
```bash
cd src
python step_0_math_foundations.py
```

### NumPy/PyTorch Warning (Step 6)
If you see `UserWarning: Failed to initialize NumPy: _ARRAY_API not found`:
- This is usually harmless and has been suppressed in the code
- If it persists, try reinstalling dependencies:
  ```bash
  make clean
  make install
  ```
- Ensure NumPy is installed before PyTorch (the Makefile handles this automatically)

### PyTorch Installation Issues
If PyTorch fails to install, visit [pytorch.org](https://pytorch.org/) for platform-specific installation instructions.

### Matplotlib Display Issues
If plots don't show:
- On Linux, you may need: `sudo apt-get install python3-tk`
- On macOS, ensure you have Xcode command line tools
- Try using a different backend: `export MPLBACKEND=TkAgg`

---

## 📚 Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Free online book

---

## 🤝 Contributing

Found a bug or want to improve something? Contributions are welcome!

1. Check existing issues
2. Create a new issue or pull request
3. Follow the code style of existing files

---

## 📄 License

This educational bootcamp is provided as-is for learning purposes.

---

## 🎉 Next Steps

After completing this bootcamp, you're ready to:
- Build your own neural networks
- Work with real datasets
- Process sequences and text (RNNs)
- Explore computer vision (CNNs)
- Explore advanced NLP (LSTMs, Transformers)
- Participate in AI competitions
- Continue learning advanced topics

---

## 💡 Tips for Success

1. **Don't skip steps** - Each builds on the previous
2. **Run the code** - Seeing it work helps understanding
3. **Experiment** - Change values, break things, learn
4. **Ask questions** - Understanding > memorization
5. **Build projects** - Apply what you learn

---

**Happy Learning! 🚀**

*Remember: Every expert was once a beginner. You're on the right path!*
