# Machine Learning Pipeline: From Data Processing to Model Optimization

## Introduction

This project provides a comprehensive exploration of modern machine learning techniques, covering the entire pipeline from data preparation to model optimization. Key focus areas include:

1. **Data Pre-processing Techniques** - Analysis of methods for cleaning, transforming, and preparing data for modeling
2. **Model Exploration** - Evaluation of various machine learning algorithms and architectures
3. **Hyperparameter Optimization** - Systematic approaches for tuning model parameters
4. **Training Strategies** - Comparative analysis of fine-tuning vs full training approaches

## Requirements

### Core Dependencies

- Python 3.7+
- PyTorch (torch) - For deep learning model implementation
- Transformers - State-of-the-art NLP models and tokenizers
- spaCy + spacy-transformers - Advanced NLP processing
- scikit-learn - Traditional machine learning algorithms
- NumPy - Numerical computing
- datasets - Dataset loading and processing
- seqeval - Sequence labeling evaluation

### Installation

```bash
pip install -r requirements.txt
```

A virtual environment is recommended.

**Project Structure:**

`utilities.py`: Contains core functions for data processing, model evaluation, and general utilities.

**Getting Started:**

1.  **Data Preparation:** Run `Data Preparation.ipynb` to format, preprocess, and generate processed data files. This notebook also handles Google Drive integration (if needed).

2.  **Experiments:** Individual notebooks detail each experimental phase:

    *   Baseline Models: Traditional machine learning.
    *   Neural Architectures: Deep learning implementations.
    *   Hyperparameter Tuning: Optimization experiments.
    *   Comparative Analysis: Fine-tuning vs. full training.

**Contributing:**

Report bugs or request features by opening an issue. Pull requests are welcome.
