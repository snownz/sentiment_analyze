# Yelp Reviews Sentiment Analysis

This project compares LSTM and DistilBERT models for sentiment analysis on Yelp reviews. The goal is to classify reviews as positive, negative, or neutral and evaluate the strengths and weaknesses of each model.

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── data.py         # Data loading and processing
│   ├── layers.py       # Custom neural network layers
│   ├── models.py       # Model definitions
│   ├── trainer.py      # Training and evaluation code
│   └── utils.py        # Utility functions
├── model_configs/      # YAML configuration files
│   ├── lstm_default.yaml
│   ├── lstm_tuning.yaml
│   ├── distilbert_default.yaml
│   └── distilbert_tuning.yaml
├── download_data.py    # Script to download/generate data
├── train_lstm.py       # Script to train LSTM model
├── train_distilbert.py # Script to train DistilBERT model
└── compare_models.py   # Script to compare model performance
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.5+
- Datasets 2.0+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn, tqdm, NLTK
- Optuna (for hyperparameter optimization)
- PyYAML (for configuration files)

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Download Data

Download the Yelp dataset from Hugging Face or generate synthetic data:

```bash
# Download real data from Hugging Face (recommended)
python download_data.py

# Generate synthetic data only
python download_data.py --synthetic --samples 5000
```

### 2. Train LSTM Model Using YAML Configuration

```bash
# Train with default configuration
python train_lstm.py --config model_configs/lstm_default.yaml

# Train with hyperparameter tuning
python train_lstm.py --config model_configs/lstm_tuning.yaml

# Force hyperparameter tuning even if best_params exist
python train_lstm.py --config model_configs/lstm_tuning.yaml --force_tuning
```

### 3. Train DistilBERT Model Using YAML Configuration

```bash
# Train with default configuration
python train_distilbert.py --config model_configs/distilbert_default.yaml

# Train with hyperparameter tuning
python train_distilbert.py --config model_configs/distilbert_tuning.yaml

# Force hyperparameter tuning even if best_params exist
python train_distilbert.py --config model_configs/distilbert_tuning.yaml --force_tuning
```

### 4. Compare Models

After training both models, you can compare their performance:

```bash
python compare_models.py --data_path data/yelp_reviews.json
```

This will generate comparison charts and metrics in the `results/comparison/` directory.

## YAML Configuration Structure

The YAML configuration files contain the following sections:

### Data Configuration
```yaml
data:
  path: data/yelp_reviews.json  # Path to the dataset
  max_length: 128               # Maximum sequence length
  max_vocab_size: 10000         # Maximum vocabulary size (LSTM only)
  batch_size: 32                # Batch size for training
```

### Model Configuration
**LSTM:**
```yaml
model:
  embedding_dim: 128      # Dimensionality of the embedding layer
  hidden_size: 64         # LSTM hidden size
  bidirectional: true     # Whether to use bidirectional LSTM
  dropout: 0.2            # Dropout rate
  use_attention: true     # Whether to use attention mechanism
```

**DistilBERT:**
```yaml
model:
  pretrained_model: distilbert-base-uncased  # Pretrained model to use
  dropout: 0.1                              # Dropout rate
```

### Training Configuration
```yaml
training:
  epochs: 10                 # Number of epochs
  optimizer: adam            # Optimizer to use
  learning_rate: 0.001       # Learning rate
  weight_decay: 0.0          # Weight decay for regularization
  scheduler: plateau         # Learning rate scheduler
  seed: 42                   # Random seed
```

### Hyperparameter Tuning Configuration
```yaml
hyperparameter_tuning:
  enabled: true           # Whether to perform hyperparameter tuning
  n_trials: 30            # Number of Optuna trials
  cv_folds: 5             # Number of cross-validation folds
  best_params: null       # Will be filled with best hyperparameters after tuning
```

After running hyperparameter tuning, the `best_params` section will be populated and saved to the YAML file for future use.

## Model Details

### LSTM Model

- Embedding layer with optional positional embeddings
- Bidirectional LSTM with dropout
- Attention mechanism for capturing important parts of the text
- Fully connected layer with softmax for classification

### DistilBERT Model

- Pretrained DistilBERT model from Hugging Face
- Fine-tuned for sentiment classification
- Smaller and faster than BERT with comparable performance

## Hyperparameter Optimization

Both models use Optuna for hyperparameter optimization with cross-validation, focusing on maximizing F1 score:

### LSTM Hyperparameters Optimized:
- Optimizer: adam, adamw, sgd, rmsprop
- Learning rate (1e-5 to 1e-2)
- Weight decay (1e-6 to 1e-3)
- Hidden size (32, 64, 128, 256)
- Embedding dimension (64, 128, 256)
- Dropout rate (0.1 to 0.5)
- Bidirectionality (True, False)
- Attention mechanism (True, False)

### DistilBERT Hyperparameters Optimized:
- Optimizer: adam, adamw
- Learning rate (1e-5 to 1e-2)
- Weight decay (1e-6 to 1e-3)
- Dropout rate (0.1 to 0.3)

## Results

The models are evaluated using the following metrics:
- Accuracy
- Precision, Recall, and F1-score for each class
- Confusion matrix
- Performance analysis by review length

Visualization of the results can be found in the `results/` directory after training and evaluation.

## Acknowledgments

This project uses the following resources:
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Hugging Face Datasets](https://huggingface.co/datasets/)
- [PyTorch](https://pytorch.org/)
- [Optuna](https://optuna.org/)
- [Yelp Dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)