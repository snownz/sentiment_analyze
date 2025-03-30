import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import nltk
from nltk.corpus import stopwords
import re
import os
import time
import argparse
import pickle
import shap
import lime
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class YelpDataset(Dataset):
    """PyTorch Dataset for Yelp reviews."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return the input_ids, attention_mask, and label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class YelpSentimentDistilBERT:
    def __init__(self, data_path=None, max_length=128, batch_size=16, 
                 epochs=4, learning_rate=2e-5, optimizer='adamw',
                 warmup_steps=0, weight_decay=0.01):
        """
        Initialize the DistilBERT model for Yelp sentiment analysis.
        
        Args:
            data_path: Path to the Yelp dataset
            max_length: Maximum sequence length for DistilBERT
            batch_size: Batch size for training
            epochs: Number of epochs to train for
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer to use (only 'adamw' supported for now)
            warmup_steps: Number of warmup steps for learning rate scheduler
            weight_decay: Weight decay for regularization
        """
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = None
        self.label_encoder = None
        self.history = None
        
        # Create directories for saving models and results
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
    
    def load_data(self, data_path=None):
        """
        Load and preprocess the Yelp dataset.
        If data_path is not provided, it attempts to download the Yelp dataset.
        
        Returns:
            DataFrame with reviews and labels
        """
        if data_path is None:
            data_path = self.data_path
        
        # If data_path is still None, try to download the dataset
        if data_path is None:
            print("Downloading Yelp dataset...")
            # This is a placeholder - in real implementation, you might download from Kaggle or another source
            # For this example, we'll use a small subset of Yelp reviews
            try:
                from sklearn.datasets import fetch_20newsgroups
                # Using 20 newsgroups as a placeholder
                data = fetch_20newsgroups(subset='all', categories=['rec.food.restaurants', 'rec.travel'])
                texts = data.data[:1000]  # Limit to first 1000 reviews
                
                # Create synthetic ratings (1-5 stars)
                ratings = np.random.randint(1, 6, size=len(texts))
                
                df = pd.DataFrame({
                    'text': texts,
                    'stars': ratings
                })
                
                print(f"Created synthetic dataset with {len(df)} reviews")
            except:
                raise ValueError("No data path provided and couldn't download sample data")
        else:
            # Load from provided path
            try:
                df = pd.read_json(data_path, lines=True)
                print(f"Loaded {len(df)} reviews from {data_path}")
            except:
                raise ValueError(f"Could not load data from {data_path}")
        
        # Convert star ratings to sentiment categories
        def star_to_sentiment(star):
            if star <= 2:
                return 'negative'
            elif star == 3:
                return 'neutral'
            else:
                return 'positive'
        
        if 'stars' in df.columns:
            df['sentiment'] = df['stars'].apply(star_to_sentiment)
        else:
            # If stars aren't available, create random sentiments for example purposes
            sentiments = np.random.choice(['negative', 'neutral', 'positive'], size=len(df))
            df['sentiment'] = sentiments
        
        # Ensure we have 'text' column
        if 'text' not in df.columns and 'review_text' in df.columns:
            df['text'] = df['review_text']
        
        return df[['text', 'sentiment']]
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text: Text string to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def prepare_data(self, df=None):
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with reviews and labels
            
        Returns:
            train_dataloader, val_dataloader, test_dataloader
        """
        if df is None:
            df = self.load_data()
        
        # Preprocess text
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Encode sentiment labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(df['sentiment'])
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df['processed_text'], encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
        )
        
        # Create PyTorch datasets
        train_dataset = YelpDataset(X_train.values, y_train, self.tokenizer, self.max_length)
        val_dataset = YelpDataset(X_val.values, y_val, self.tokenizer, self.max_length)
        test_dataset = YelpDataset(X_test.values, y_test, self.tokenizer, self.max_length)
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size
        )
        
        print(f"Training data: {len(train_dataset)} samples")
        print(f"Validation data: {len(val_dataset)} samples")
        print(f"Test data: {len(test_dataset)} samples")
        
        # Save the tokenizer and label encoder
        with open('models/distilbert_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('models/distilbert_label_encoder.pickle', 'wb') as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return train_dataloader, val_dataloader, test_dataloader, df
    
    def build_model(self):
        """
        Build the DistilBERT model for sentiment analysis.
        
        Returns:
            DistilBERT model
        """
        # Load pre-trained model with classification head
        num_labels = 3  # positive, negative, neutral
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        )
        
        # Move model to device (GPU if available)
        model = model.to(device)
        
        self.model = model
        return model
    
    def train(self, train_dataloader, val_dataloader):
        """
        Train the DistilBERT model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Set up optimizer and scheduler
        if self.optimizer_name.lower() == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported for DistilBERT")
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * self.epochs
        
        # Create scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Start training
        print(f"Training DistilBERT model for {self.epochs} epochs...")
        start_time = time.time()
        
        best_val_accuracy = 0
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            # Progress bar for training
            progress_bar = tqdm(train_dataloader, desc="Training")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            # Evaluation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    # Get predictions
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1)
                    
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = accuracy_score(val_true, val_preds)
            
            print(f"Validation loss: {avg_val_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), 'models/distilbert_best_model.pt')
                print(f"Best model saved with validation accuracy: {val_accuracy:.4f}")
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/distilbert_best_model.pt'))
        
        # Save final model
        torch.save(self.model.state_dict(), 'models/distilbert_final_model.pt')
        
        self.history = history
        return history
    
    def evaluate(self, test_dataloader):
        """
        Evaluate the DistilBERT model on test data.
        
        Args:
            test_dataloader: DataLoader for test data
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize predictions and true labels
        test_preds = []
        test_true = []
        
        # Evaluate without gradient calculation
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Evaluation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(test_true, test_preds)
        
        class_names = self.label_encoder.classes_
        report = classification_report(test_true, test_preds, target_names=class_names, output_dict=True)
        
        # Print results
        print(f"\nTest accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(test_true, test_preds, target_names=class_names))
        
        # Create and plot confusion matrix
        cm = confusion_matrix(test_true, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('DistilBERT Confusion Matrix')
        plt.tight_layout()
        plt.savefig('results/distilbert_confusion_matrix.png')
        plt.close()
        
        # Save evaluation results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        return results, test_preds, test_true
    
    def analyze_performance_by_length(self, df, test_preds, test_true):
        """
        Analyze model performance on reviews of different lengths.
        
        Args:
            df: Original DataFrame with text data
            test_preds: Model predictions
            test_true: True labels
        """
        # Get test set
        test_size = len(test_true)
        test_texts = df['processed_text'].iloc[-test_size:].reset_index(drop=True)
        
        # Get text lengths
        text_lengths = [len(text.split()) for text in test_texts]
        
        # Create a DataFrame for analysis
        analysis_df = pd.DataFrame({
            'true_label': test_true,
            'predicted_label': test_preds,
            'correct': np.array(test_true) == np.array(test_preds),
            'text_length': text_lengths
        })
        
        # Define length categories
        length_bins = [0, 50, 100, 200, float('inf')]
        length_labels = ['Very Short (0-50)', 'Short (51-100)', 'Medium (101-200)', 'Long (>200)']
        
        analysis_df['length_category'] = pd.cut(
            analysis_df['text_length'], 
            bins=length_bins, 
            labels=length_labels, 
            right=False
        )
        
        # Calculate accuracy by length category
        accuracy_by_length = analysis_df.groupby('length_category')['correct'].mean()
        
        # Plot results
        plt.figure(figsize=(10, 6))
        sns.barplot(x=accuracy_by_length.index, y=accuracy_by_length.values)
        plt.title('DistilBERT Accuracy by Review Length')
        plt.xlabel('Review Length Category')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig('results/distilbert_accuracy_by_length.png')
        plt.close()
        
        # Print results
        print("\nAccuracy by Review Length:")
        for category, acc in accuracy_by_length.items():
            print(f"{category}: {acc:.4f}")
        
        return accuracy_by_length
    
    def interpret_model(self, df, n_samples=5):
        """
        Interpret model predictions using LIME.
        
        Args:
            df: DataFrame with text data
            n_samples: Number of samples to interpret
        """
        # Get test set
        test_size = int(len(df) * 0.2)
        test_texts = df['text'].iloc[-test_size:].reset_index(drop=True)
        
        # Select random samples to interpret
        indices = np.random.choice(range(len(test_texts)), n_samples, replace=False)
        
        # Initialize LIME explainer
        explainer = LimeTextExplainer(class_names=self.label_encoder.classes_)
        
        # Function to make predictions with DistilBERT
        def predict_fn(texts):
            # Set model to evaluation mode
            self.model.eval()
            
            # Initialize predictions array
            predictions = np.zeros((len(texts), len(self.label_encoder.classes_)))
            
            # Process texts in batches
            for i, text in enumerate(texts):
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Store probabilities
                predictions[i] = probs.cpu().numpy()
            
            return predictions
        
        # Generate explanations
        for idx in indices:
            text = test_texts.iloc[idx]
            exp = explainer.explain_instance(text, predict_fn, num_features=10)
            
            # Get the predicted class
            pred_class = np.argmax(predict_fn([text])[0])
            pred_label = self.label_encoder.classes_[pred_class]
            
            # Save the explanation visualization
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f'DistilBERT Explanation for Predicted Class: {pred_label}')
            plt.tight_layout()
            plt.savefig(f'results/distilbert_lime_explanation_{idx}.png')
            plt.close()
            
            # Print explanation to console
            print(f"\nText: {text[:100]}...")
            print(f"Predicted class: {pred_label}")
            print("Top features:")
            for feature, weight in exp.as_list():
                print(f"  {feature}: {weight:.4f}")
    
    def hyperparameter_tuning(self, df):
        """
        Perform hyperparameter tuning for DistilBERT.
        This is a simplified version - in practice, you might use cross-validation.
        
        Args:
            df: DataFrame with text data
            
        Returns:
            Best hyperparameters
        """
        # Define hyperparameter combinations to try
        hyperparams = [
            {'batch_size': 8, 'learning_rate': 2e-5, 'weight_decay': 0.01},
            {'batch_size': 8, 'learning_rate': 5e-5, 'weight_decay': 0.01},
            {'batch_size': 16, 'learning_rate': 2e-5, 'weight_decay': 0.01},
            {'batch_size': 16, 'learning_rate': 5e-5, 'weight_decay': 0.01},
            {'batch_size': 16, 'learning_rate': 2e-5, 'weight_decay': 0.1},
        ]
        
        best_val_accuracy = 0
        best_params = None
        
        # Preprocess text and encode labels
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df['sentiment'])
        
        # Split into train, validation, and test sets
        X_train, X_val, y_train, y_val = train_test_split(
            df['processed_text'], encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
        )
        
        for i, params in enumerate(hyperparams):
            print(f"\nHyperparameter configuration {i+1}/{len(hyperparams)}: {params}")
            
            # Update parameters
            self.batch_size = params['batch_size']
            self.learning_rate = params['learning_rate']
            self.weight_decay = params['weight_decay']
            
            # Create datasets and dataloaders
            train_dataset = YelpDataset(X_train.values, y_train, self.tokenizer, self.max_length)
            val_dataset = YelpDataset(X_val.values, y_val, self.tokenizer, self.max_length)
            
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=self.batch_size
            )
            
            # Reset model for each configuration
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            del self.model
            
            self.epochs = 1  # Use fewer epochs for tuning
            self.build_model()
            self.train(train_dataloader, val_dataloader)
            
            # Get validation accuracy
            val_accuracy = max(self.history['val_accuracy'])
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Update best parameters if better
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = params
        
        print(f"\nBest hyperparameters: {best_params}")
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        # Reset to best parameters
        self.batch_size = best_params['batch_size']
        self.learning_rate = best_params['learning_rate']
        self.weight_decay = best_params['weight_decay']
        
        return best_params
    
    def plot_training_history(self):
        """
        Plot training and validation metrics.
        """
        if self.history is None:
            print("No training history available")
            return
        
        # Create a figure with 2 subplots
        plt.figure(figsize=(12, 5))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('DistilBERT Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
        plt.title('DistilBERT Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/distilbert_training_history.png')
        plt.close()
    
    def run_end_to_end(self, data_path=None, tune_hyperparams=False):
        """
        Run the entire workflow from data loading to evaluation.
        
        Args:
            data_path: Path to the Yelp dataset
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Results dictionary
        """
        # Set data path if provided
        if data_path:
            self.data_path = data_path
        
        # Load data
        print("Loading data...")
        df = self.load_data()
        
        # Hyperparameter tuning if requested
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            best_params = self.hyperparameter_tuning(df)
            # Update parameters
            self.batch_size = best_params['batch_size']
            self.learning_rate = best_params['learning_rate']
            self.weight_decay = best_params['weight_decay']
            
            # Restore original epochs
            self.epochs = 4
        
        # Prepare data
        print("Preparing data...")
        train_dataloader, val_dataloader, test_dataloader, df = self.prepare_data(df)
        
        # Build and train model
        print("Building model...")
        self.build_model()
        print("Training model...")
        self.train(train_dataloader, val_dataloader)
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate model
        print("Evaluating model...")
        results, test_preds, test_true = self.evaluate(test_dataloader)
        
        # Analyze performance by review length
        print("Analyzing performance by review length...")
        length_analysis = self.analyze_performance_by_length(df, test_preds, test_true)
        
        # Interpret model predictions
        print("Interpreting model predictions...")
        self.interpret_model(df)
        
        print("\nTraining and evaluation complete!")
        return results

def main():
    parser = argparse.ArgumentParser(description='DistilBERT Model for Yelp Review Sentiment Analysis')
    
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Path to the Yelp dataset file')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length for DistilBERT')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=4, 
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=2e-5, 
                        help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay for regularization')
    parser.add_argument('--warmup_steps', type=int, default=0, 
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--tune_hyperparams', action='store_true', 
                        help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Initialize and run the DistilBERT model
    distilbert_model = YelpSentimentDistilBERT(
        data_path=args.data_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps
    )
    
    results = distilbert_model.run_end_to_end(
        data_path=args.data_path,
        tune_hyperparams=args.tune_hyperparams
    )
    
    return results

if __name__ == '__main__':
    main()