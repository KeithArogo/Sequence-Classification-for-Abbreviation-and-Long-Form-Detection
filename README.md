This project demonstrates a complete machine learning pipeline, from data preprocessing to model optimization. It explores:

*   Data Preprocessing: Cleaning, transforming, and preparing data.
*   Model Exploration: Evaluating diverse machine learning algorithms.
*   Hyperparameter Optimization: Tuning model parameters systematically.
*   Training Strategies: Comparing fine-tuning and full training approaches.

**Requirements:**

*   Python 3.7+
*   Dependencies: PyTorch (torch), Transformers, spaCy & spacy-transformers, scikit-learn, NumPy, datasets, seqeval.

**Installation:**

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
