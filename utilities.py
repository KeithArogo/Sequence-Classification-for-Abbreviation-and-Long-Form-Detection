# -*- coding: utf-8 -*-
"""utilities

Utilities for Named Entity Recognition (NER) tasks, including data preprocessing, model evaluation, and random example display.

Sources:
- https://github.com/yrnigam/Named-Entity-Recognition-NER-using-LSTMs/blob/master/Named_Entity_Recognition_(NER)_using_LSTMs.ipynb
- https://huggingface.co/learn/nlp-course/en/chapter7/2
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_metric


class SentenceGetter:
    """Class to organize data in a desired format: Word, POS, Tag."""

    def __init__(self, data):
        """Initialize SentenceGetter.

        Args:
            data: The input data as a pandas DataFrame.
        """
        self.n_sent = 1  # Counter
        self.data = data
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].tolist(), s['POS'].tolist(), s['Tag'].tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]


def preprocess_data(sentences, word2idx, tag2idx, max_len, num_words, num_tags):
    """Preprocess data for training/testing.

    Args:
        sentences: List of sentences, each sentence is a list of tuples (word, POS, tag).
        word2idx: Dictionary mapping words to indices.
        tag2idx: Dictionary mapping tags to indices.
        max_len: Maximum length of sequences.
        num_words: Number of unique words.
        num_tags: Number of unique tags.

    Returns:
        Tuple of preprocessed input and output data (X, y).
    """
    padding_value = word2idx["ENDPAD"]  # Index of "ENDPAD"
    X = [[word2idx.get(w[0], padding_value) for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding='post', value=padding_value)

    y = [[tag2idx.get(w[2], 0) for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding='post', value=tag2idx["PAD"])

    # Convert tags to one-hot encoded format
    y = [to_categorical(i, num_classes=num_tags) for i in y]

    return X, y


def evaluate_model(model, x_test, y_test, class_mapping, plot_title='Confusion Matrix', plot_size=(8, 8)):
    """Evaluate model performance using a confusion matrix and other metrics.

    Args:
        model: The trained model to evaluate.
        x_test: Test input data.
        y_test: Test output data (true labels).
        class_mapping: Dictionary mapping class indices to class names.
        plot_title: Title of the confusion matrix plot.
        plot_size: Size of the confusion matrix plot.

    Returns:
        None.
    """
    # Make predictions on the test data
    y_pred = model.predict(x_test)

    # Convert predictions from one-hot encoding to class labels
    y_pred_labels = np.argmax(y_pred, axis=2)
    y_true_labels = np.argmax(y_test, axis=2)

    # Filter out 'PAD' class (class_mapping value for 0)
    pad_index = list(class_mapping.keys())[list(class_mapping.values()).index('PAD')]
    valid_indices = y_true_labels != pad_index

    # Apply the filter to the true and predicted labels
    filtered_y_true_labels = y_true_labels[valid_indices]
    filtered_y_pred_labels = y_pred_labels[valid_indices]

    # Map numerical indices to string labels
    y_pred_labels_str = np.vectorize(class_mapping.get)(filtered_y_pred_labels)
    y_true_labels_str = np.vectorize(class_mapping.get)(filtered_y_true_labels)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true_labels_str, y_pred_labels_str, labels=[cls for cls in list(class_mapping.values()) if cls != 'PAD'])

    # Plot the confusion matrix
    plt.figure(figsize=plot_size)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[cls for cls in list(class_mapping.values()) if cls != 'PAD'],
                yticklabels=[cls for cls in list(class_mapping.values()) if cls != 'PAD'])
    plt.title(plot_title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate precision, recall, and F1 score for each class
    precision = precision_score(y_true_labels_str, y_pred_labels_str, average=None, labels=[cls for cls in list(class_mapping.values()) if cls != 'PAD'])
    recall = recall_score(y_true_labels_str, y_pred_labels_str, average=None, labels=[cls for cls in list(class_mapping.values()) if cls != 'PAD'])
    f1 = f1_score(y_true_labels_str, y_pred_labels_str, average=None, labels=[cls for cls in list(class_mapping.values()) if cls != 'PAD'])

    # Display precision, recall, and F1 score for each class
    print("Class\tPrecision\tRecall\tF1 Score")
    print("-" * 35)
    for cls, prec, rec, f1_score_cls in zip([cls for cls in list(class_mapping.values()) if cls != 'PAD'], precision, recall, f1):
        print(f"{cls}\t{prec:.4f}\t{rec:.4f}\t{f1_score_cls:.4f}")

    # Calculate weighted and macro F1 scores
    f1_weighted = f1_score(y_true_labels_str, y_pred_labels_str, average='weighted')
    f1_macro = f1_score(y_true_labels_str, y_pred_labels_str, average='macro')

    print(f"\nWeighted F1 score: {f1_weighted:.4f}")
    print(f"Macro F1 score: {f1_macro:.4f}")


def show_random_example(model, x_test, y_test, words, tags, random_state=None):
    """Show a random example from the test data with model predictions.

    Args:
        model: The trained model to use for predictions.
        x_test: Test input data.
        y_test: Test output data (true labels).
        words: List of words in the vocabulary.
        tags: List of tags in the vocabulary.
        random_state: Seed for the random number generator (optional).

    Returns:
        None.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Randomly select an example from the test data
    i = np.random.randint(0, x_test.shape[0])

    # Make predictions using the model for the selected example
    p = model.predict(np.array([x_test[i]]))
    p = np.argmax(p, axis=-1)

    # Get the true tags for the selected example
    y_true = np.argmax(np.array(y_test), axis=-1)[i]

    # Print the header for the output
    print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
    print("-" * 30)

    # Compare and print the true and predicted tags for each word in the selected example
    for w, true, pred in zip(x_test[i], y_true, p[0]):
        print("{:15}{:5}\t{}".format(words[w - 1], tags[true], tags[pred]))


# Load the seqeval metric
metric = load_metric("seqeval")


def model_evaluate(trainer, tokenized_test, label_list):
    """Evaluate a model using the seqeval metric.

    Args:
        trainer: The trained model.
        tokenized_test: Tokenized test data.
        label_list: List of labels.

    Returns:
        Dictionary of evaluation metrics, including the confusion matrix.
    """
    # Get predictions and labels from the trained model
    predictions, labels, _ = trainer.predict(tokenized_test)
    predictions = np.argmax(predictions, axis=2)

    # Process the predictions and labels to remove special tokens
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute multiple metrics using the seqeval metric
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Calculate macro F1-score
    f1_scores = []
    for label, metrics in results.items():
        # Only consider the non-overall labels (skip 'overall_' metrics)
        if label not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
            f1_scores.append(metrics['f1'])
    
    # Calculate macro F1-score by taking the mean of F1 scores
    macro_f1 = np.mean(f1_scores)
    results['macro_f1'] = macro_f1

    true_labels_flat = [label for sublist in true_labels for label in sublist]
    predictions_flat = [label for sublist in true_predictions for label in sublist]

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels_flat, predictions_flat, labels=label_list)

    # Add the confusion matrix to the results dictionary
    results['confusion_matrix'] = cm

    return results


def compute_metrics(p):
    """Compute evaluation metrics including precision, recall, F1, and accuracy.

    Args:
        p: Predictions and labels.

    Returns:
        Dictionary of evaluation metrics including precision, recall, F1, accuracy, and macro F1.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    label_list = ['B-B-O', 'B-B-AC', 'B-B-LF', 'B-I-LF']

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Compute the seqeval metric
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Calculate macro F1
    f1_scores = []
    for label, metrics in results.items():
        if label not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
            # Only consider the non-overall labels
            f1_scores.append(metrics['f1'])

    # Calculate macro F1-score
    macro_f1 = np.mean(f1_scores)

    # Return the metrics as a dictionary, including macro F1
    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy'],
        'macro_f1': macro_f1,
    }


def tokenize_and_align_labels(short_dataset, list_name, tokenizer):
    """Tokenize and align labels for input data.

    Args:
        short_dataset: Dataset containing tokens.
        list_name: List of labels.
        tokenizer: Tokenizer to be used for tokenizing.

    Returns:
        Dictionary containing tokenized inputs and aligned labels.
    """
    tokenized_inputs = tokenizer(short_dataset["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(list_name):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
