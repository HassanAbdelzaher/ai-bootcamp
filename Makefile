# AI Codecamp Makefile
# Common commands for running and managing the AI bootcamp steps
# Note: This Makefile is optimized for Unix/macOS. Windows users may need to use commands directly.

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
SHELL = /bin/bash

.PHONY: help venv install install-dev clean run-all run-step-0 run-step-1 run-step-2 run-step-3 run-step-4 run-step-5 run-step-6 run-step-7 run-step-7a run-step-7b run-step-7c run-step-7d run-step-8 run-step-8a run-step-8b run-step-8c run-step-8d run-step-8e example-0 example-1 example-2 example-3 example-4 example-5 example-6 example-vg project-1 project-2 test check

# Default target
help:
	@echo "AI Codecamp - Available Commands:"
	@echo ""
	@echo "  make venv             - Create virtual environment"
	@echo "  make install          - Install all dependencies (creates venv if needed)"
	@echo "  make install-dev      - Install dependencies for development"
	@echo "  make clean            - Clean Python cache files and venv"
	@echo "  make run-all          - Run all steps sequentially"
	@echo "  make run-step-0       - Run Step 0: Math Foundations"
	@echo "  make run-step-1       - Run Step 1: Linear Regression"
	@echo "  make run-step-2       - Run Step 2: Perceptron"
	@echo "  make run-step-3       - Run Step 3: Logistic Regression"
	@echo "  make run-step-4       - Run Step 4: Multiple Neurons"
	@echo "  make run-step-5       - Run Step 5: XOR and Hidden Layers"
	@echo "  make run-step-6       - Run Step 6: PyTorch"
	@echo "  make run-step-7       - Run Step 7: RNNs (Sequences)"
	@echo "  make run-step-7a      - Run Step 7a: Text Generator"
	@echo "  make run-step-7b      - Run Step 7b: Stock Price Prediction"
	@echo "  make run-step-7c      - Run Step 7c: LSTM and GRU"
	@echo "  make run-step-7d      - Run Step 7d: Transformers (BERT, GPT)"
	@echo "  make run-step-8       - Run Step 8: CNNs (Images)"
	@echo "  make run-step-8a      - Run Step 8a: Real Datasets (CIFAR-10)"
	@echo "  make run-step-8b      - Run Step 8b: Image Classifiers"
	@echo "  make run-step-8c      - Run Step 8c: Transfer Learning"
	@echo "  make run-step-8d      - Run Step 8d: Object Detection (YOLO, R-CNN)"
	@echo "  make run-step-8e      - Run Step 8e: Image Generation (GANs, VAEs)"
	@echo ""
	@echo "Examples (practical neuron usage):"
	@echo "  make example-0        - Example: Simple neuron (Step 0)"
	@echo "  make example-1        - Example: Linear regression neuron (Step 1)"
	@echo "  make example-2        - Example: Perceptron neuron (Step 2)"
	@echo "  make example-3        - Example: Logistic neuron (Step 3)"
	@echo "  make example-4        - Example: Multiple neurons (Step 4)"
	@echo "  make example-5        - Example: Deep network (Step 5)"
	@echo "  make example-6        - Example: PyTorch neurons (Step 6)"
	@echo "  make example-vg       - Example: Vanishing Gradient visualization"
	@echo ""
	@echo "Projects (practical applications):"
	@echo "  make project-1         - Run Project 1: Simple Predictor"
	@echo "  make project-2         - Run Project 2: Multi-Class Classifier"
	@echo "  make check             - Check Python syntax"
	@echo "  make test             - Test imports"
	@echo ""
	@echo "Note: All commands automatically use the virtual environment."

# Create virtual environment
venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
		echo "Virtual environment created!"; \
	else \
		echo "Virtual environment already exists."; \
	fi

# Install dependencies (creates venv if needed)
install: venv
	@echo "Installing dependencies..."
	@bash -c "source $(VENV)/bin/activate && pip install --upgrade pip"
	@bash -c "source $(VENV)/bin/activate && pip install -r requirements.txt"
	@echo "Dependencies installed!"

# Install development dependencies (if needed)
install-dev: install
	@bash -c "source $(VENV)/bin/activate && pip install flake8 black pylint"
	@echo "Development dependencies installed!"

# Clean Python cache files and venv
clean:
	@find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name ".DS_Store" -delete
	@if [ -d "$(VENV)" ]; then \
		echo "Removing virtual environment..."; \
		rm -rf $(VENV); \
	fi
	@echo "Cleaned Python cache files and virtual environment"

# Run all steps
run-all: run-step-0 run-step-1 run-step-2 run-step-3 run-step-4 run-step-5 run-step-6 run-step-7 run-step-8

# Run individual steps
run-step-0: venv
	@echo "Running Step 0: Math Foundations..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_0_math_foundations.py"

run-step-1: venv
	@echo "Running Step 1: Linear Regression..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_1_linear_regression.py"

run-step-2: venv
	@echo "Running Step 2: Perceptron..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_2_perceptron.py"

run-step-3: venv
	@echo "Running Step 3: Logistic Regression..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_3_logistic_regression.py"

run-step-4: venv
	@echo "Running Step 4: Multiple Neurons..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_4_multiple_neurons.py"

run-step-5: venv
	@echo "Running Step 5: XOR and Hidden Layers..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_5_xor_and_hidden_layers.py"

run-step-6: venv
	@echo "Running Step 6: PyTorch..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_6_pytorch.py"

run-step-7: venv
	@echo "Running Step 7: RNNs (Sequences)..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_7_rnns.py"

run-step-7a: venv
	@echo "Running Step 7a: Text Generator..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_7a_text_generator.py"

run-step-7b: venv
	@echo "Running Step 7b: Stock Price Prediction..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_7b_stock_prices.py"

run-step-7c: venv
	@echo "Running Step 7c: LSTM and GRU..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_7c_lstm_gru.py"

run-step-7d: venv
	@echo "Running Step 7d: Transformers (BERT, GPT)..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_7d_transformers.py"

run-step-8: venv
	@echo "Running Step 8: CNNs (Images)..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_8_cnns.py"

run-step-8a: venv
	@echo "Running Step 8a: Real Datasets (CIFAR-10)..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_8a_real_datasets.py"

run-step-8b: venv
	@echo "Running Step 8b: Image Classifiers..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_8b_image_classifiers.py"

run-step-8c: venv
	@echo "Running Step 8c: Transfer Learning..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_8c_transfer_learning.py"

run-step-8d: venv
	@echo "Running Step 8d: Object Detection (YOLO, R-CNN)..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_8d_object_detection.py"

run-step-8e: venv
	@echo "Running Step 8e: Image Generation (GANs, VAEs)..."
	@bash -c "source $(VENV)/bin/activate && cd src && python step_8e_image_generation.py"

# Example files (practical neuron usage)
example-0: venv
	@echo "Running Example 0: Simple Neuron..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_0_neuron.py"

example-1: venv
	@echo "Running Example 1: Linear Regression Neuron..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_1_neuron.py"

example-2: venv
	@echo "Running Example 2: Perceptron Neuron..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_2_neuron.py"

example-3: venv
	@echo "Running Example 3: Logistic Regression Neuron..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_3_neuron.py"

example-4: venv
	@echo "Running Example 4: Multiple Neurons..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_4_neuron.py"

example-5: venv
	@echo "Running Example 5: Deep Network..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_5_neuron.py"

example-6: venv
	@echo "Running Example 6: PyTorch Neurons..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_6_neuron.py"

example-vg: venv
	@echo "Running Example: Vanishing Gradient Visualization..."
	@bash -c "source $(VENV)/bin/activate && cd src && python example_vanishing_gradient.py"

# Projects
project-1: venv
	@echo "Running Project 1: Simple Predictor..."
	@bash -c "source $(VENV)/bin/activate && cd projects/project_1_simple_predictor && python house_price_predictor.py && python spam_classifier.py"

project-2: venv
	@echo "Running Project 2: Multi-Class Classifier..."
	@bash -c "source $(VENV)/bin/activate && cd projects/project_2_classifier && python digit_classifier.py"

# Check Python syntax
check: venv
	@echo "Checking Python syntax..."
	@bash -c "source $(VENV)/bin/activate && python -m py_compile src/*.py"
	@echo "Syntax check passed!"

# Test imports
test: venv
	@echo "Testing imports..."
	@bash -c "source $(VENV)/bin/activate && cd src && python -c \"import numpy; import matplotlib; print('✓ numpy and matplotlib OK')\""
	@bash -c "source $(VENV)/bin/activate && cd src && python -c \"import torch; print('✓ torch OK')\"" || echo "⚠ torch not installed (needed for Step 6)"
	@echo "Import test complete"
