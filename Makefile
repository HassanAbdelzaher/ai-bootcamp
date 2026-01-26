# AI Codecamp Makefile
# Common commands for running and managing the AI bootcamp steps
# Note: This Makefile is optimized for Unix/macOS. Windows users may need to use commands directly.

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
SHELL = /bin/bash

.PHONY: help venv install install-dev clean run-all run-step-0 run-step-1 run-step-2 run-step-3 run-step-4 run-step-5 run-step-6 run-step-7 test check

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
	@echo "  make check            - Check Python syntax"
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
run-all: run-step-0 run-step-1 run-step-2 run-step-3 run-step-4 run-step-5 run-step-6 run-step-7

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
