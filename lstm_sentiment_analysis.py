import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import nltk
from nltk.corpus import stopwords
import re
import pickle
import os
import time
import argparse
import shap
import lime
from lime.lime_text import LimeTextExplainer

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class YelpSentimentLSTM:
    def __init__(self, data_path=None, max_features=10000, maxlen=200, 
                 embedding_dim=128, lstm_units=64, dropout_rate=0.2, 
                 batch_size=32, epochs=10, optimizer='adam'):
        """
        Initialize the LSTM model for Yelp sentiment analysis.
        
        Args:
            data_path: Path to the Yelp dataset
            max_features: Maximum number of words to consider in the vocabulary
            maxlen: Maximum length of reviews (in words) to consider
            embedding_dim: Dimensionality of the embedding layer
            lstm_units: Number of units in the LSTM layer
            dropout_rate: Dropout rate for regularization
            batch_size: Number of samples per batch
            epochs: Number of epochs to train for
            optimizer: Optimizer to use ('sgd', 'rmsprop', or 'adam')
        """
        self.data_path = data_path
        self.max_features = max_features
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer_name = optimizer
        
        # Initialize model and other components later
        self.model = None
        self.tokenizer = None
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
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optionally remove stopwords
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        
        return text
    
    def prepare_data(self, df=None):
        """
        Prepare data for training by tokenizing and padding sequences.
        
        Args:
            df: DataFrame with reviews and labels
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
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
        
        # Fit tokenizer on training data
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(X_train)
        
        # Save the tokenizer
        with open('models/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Convert text to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_val_seq = self.tokenizer.texts_to_sequences(X_val)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences to ensure uniform length
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.maxlen)
        X_val_pad = pad_sequences(X_val_seq, maxlen=self.maxlen)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.maxlen)
        
        # Convert to one-hot encoding for multi-class classification
        num_classes = len(self.label_encoder.classes_)
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)
        
        print(f"Training data: {X_train_pad.shape[0]} samples")
        print(f"Validation data: {X_val_pad.shape[0]} samples")
        print(f"Test data: {X_test_pad.shape[0]} samples")
        
        return X_train_pad, X_val_pad, X_test_pad, y_train_onehot, y_val_onehot, y_test_onehot, df
    
    def build_model(self):
        """
        Build the LSTM model for sentiment analysis.
        
        Returns:
            Compiled Keras model
        """
        # Define the optimizer
        if self.optimizer_name.lower() == 'sgd':
            optimizer = SGD(learning_rate=0.01, momentum=0.9)
        elif self.optimizer_name.lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=0.001)
        else:  # default to adam
            optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
        
        # Build the model
        model = Sequential()
        
        # Embedding layer
        model.add(Embedding(input_dim=self.max_features, 
                           output_dim=self.embedding_dim, 
                           input_length=self.maxlen))
        
        # Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(units=self.lstm_units, dropout=0.1, recurrent_dropout=0.1)))
        
        # Dropout for regularization
        model.add(Dropout(self.dropout_rate))
        
        # Output layer with softmax activation for multi-class classification
        num_classes = 3  # positive, negative, neutral
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        model.summary()
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, use_cross_validation=False, n_folds=5):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            use_cross_validation: Whether to use k-fold cross-validation
            n_folds: Number of folds for cross-validation
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        if use_cross_validation:
            return self._train_with_cross_validation(X_train, y_train, n_folds)
        else:
            return self._train_standard(X_train, y_train, X_val, y_val)
    
    def _train_standard(self, X_train, y_train, X_val, y_val):
        """
        Standard training without cross-validation.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            
        Returns:
            Training history
        """
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            filepath='models/lstm_model_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        # Train the model
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint]
        )
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Save the final model
        self.model.save('models/lstm_model_final.h5')
        
        self.history = history
        return history
    
    def _train_with_cross_validation(self, X, y, n_folds=5):
        """
        Train with k-fold cross-validation.
        
        Args:
            X: Training data
            y: Training labels
            n_folds: Number of folds
            
        Returns:
            List of training histories
        """
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        histories = []
        fold_accuracies = []
        
        # Initialize fold counter
        fold = 0
        
        for train_idx, val_idx in kfold.split(X):
            fold += 1
            print(f"\nTraining fold {fold}/{n_folds}")
            
            # Get fold data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Reset the model for each fold
            if fold > 1:
                # Clear session and rebuild model for each new fold
                tf.keras.backend.clear_session()
                self.build_model()
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            model_checkpoint = ModelCheckpoint(
                filepath=f'models/lstm_model_fold_{fold}.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
            
            # Train model on this fold
            history = self.model.fit(
                X_train_fold, y_train_fold,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[early_stopping, model_checkpoint]
            )
            
            # Evaluate on validation fold
            scores = self.model.evaluate(X_val_fold, y_val_fold, verbose=0)
            fold_accuracies.append(scores[1])  # Accuracy is at index 1
            print(f"Fold {fold} validation accuracy: {scores[1]:.4f}")
            
            histories.append(history)
        
        # Calculate average performance across folds
        avg_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print(f"\nCross-validation results:")
        print(f"Average accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Save the final model (from last fold)
        self.model.save('models/lstm_model_final.h5')
        
        self.history = histories[-1]  # Save the last fold's history
        return histories
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        class_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Print results
        print(f"\nTest accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Create and plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png')
        plt.close()
        
        # Save evaluation results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def analyze_performance_by_length(self, X_test, y_test, df):
        """
        Analyze model performance on reviews of different lengths.
        
        Args:
            X_test: Test data
            y_test: Test labels
            df: Original DataFrame with text data
        """
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Get original text lengths
        text_lengths = [len(text.split()) for text in df['processed_text'].iloc[-len(y_true):]]
        
        # Create a DataFrame for analysis
        analysis_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'correct': y_true == y_pred,
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
        plt.title('Model Accuracy by Review Length')
        plt.xlabel('Review Length Category')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig('results/accuracy_by_length.png')
        plt.close()
        
        # Print results
        print("\nAccuracy by Review Length:")
        for category, acc in accuracy_by_length.items():
            print(f"{category}: {acc:.4f}")
        
        return accuracy_by_length
    
    def interpret_model(self, X_test_raw, X_test_processed, n_samples=5):
        """
        Interpret model predictions using LIME.
        
        Args:
            X_test_raw: Raw text of test samples
            X_test_processed: Processed and padded sequences
            n_samples: Number of samples to interpret
        """
        # Select random samples to interpret
        indices = np.random.choice(range(len(X_test_raw)), n_samples, replace=False)
        
        # Initialize LIME explainer
        explainer = LimeTextExplainer(class_names=self.label_encoder.classes_)
        
        # Function to make predictions
        def predict_fn(texts):
            # Tokenize and pad the texts
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.maxlen)
            return self.model.predict(padded)
        
        # Generate explanations
        for idx in indices:
            text = X_test_raw.iloc[idx]
            exp = explainer.explain_instance(text, predict_fn, num_features=10)
            
            # Get the predicted class
            pred_class = np.argmax(predict_fn([text])[0])
            pred_label = self.label_encoder.classes_[pred_class]
            
            # Save the explanation visualization
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f'Explanation for Predicted Class: {pred_label}')
            plt.tight_layout()
            plt.savefig(f'results/lime_explanation_{idx}.png')
            plt.close()
            
            # Print explanation to console
            print(f"\nText: {text[:100]}...")
            print(f"Predicted class: {pred_label}")
            print("Top features:")
            for feature, weight in exp.as_list():
                print(f"  {feature}: {weight:.4f}")
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """
        Perform hyperparameter tuning for the LSTM model.
        This is a simplified version - in practice, you might use GridSearchCV or Keras Tuner.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            
        Returns:
            Best hyperparameters
        """
        # Define hyperparameter configurations to try
        hyperparams = [
            # Vary LSTM units
            {'lstm_units': 32, 'dropout_rate': 0.2, 'optimizer': 'adam'},
            {'lstm_units': 64, 'dropout_rate': 0.2, 'optimizer': 'adam'},
            {'lstm_units': 128, 'dropout_rate': 0.2, 'optimizer': 'adam'},
            
            # Vary dropout rate
            {'lstm_units': 64, 'dropout_rate': 0.1, 'optimizer': 'adam'},
            {'lstm_units': 64, 'dropout_rate': 0.3, 'optimizer': 'adam'},
            
            # Vary optimizer
            {'lstm_units': 64, 'dropout_rate': 0.2, 'optimizer': 'sgd'},
            {'lstm_units': 64, 'dropout_rate': 0.2, 'optimizer': 'rmsprop'},
            {'lstm_units': 64, 'dropout_rate': 0.2, 'optimizer': 'adam'},
        ]
        
        best_accuracy = 0
        best_params = None
        
        print("\nHyperparameter Tuning:")
        for i, params in enumerate(hyperparams):
            print(f"\nTrying configuration {i+1}/{len(hyperparams)}: {params}")
            
            # Reset model with new hyperparameters
            tf.keras.backend.clear_session()
            self.lstm_units = params['lstm_units']
            self.dropout_rate = params['dropout_rate']
            self.optimizer_name = params['optimizer']
            
            # Build and train model
            self.build_model()
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Train for fewer epochs during tuning
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=min(5, self.epochs),  # Limit epochs for faster tuning
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate on validation data
            val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Update best parameters if this configuration is better
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_params = params
        
        print(f"\nBest hyperparameters: {best_params}")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        
        # Reset model with best parameters
        tf.keras.backend.clear_session()
        self.lstm_units = best_params['lstm_units']
        self.dropout_rate = best_params['dropout_rate']
        self.optimizer_name = best_params['optimizer']
        
        return best_params
    
    def run_end_to_end(self, data_path=None, tune_hyperparams=False, use_cross_validation=False):
        """
        Run the entire workflow from data loading to evaluation.
        
        Args:
            data_path: Path to the Yelp dataset
            tune_hyperparams: Whether to perform hyperparameter tuning
            use_cross_validation: Whether to use cross-validation
            
        Returns:
            Results dictionary
        """
        # Set data path if provided
        if data_path:
            self.data_path = data_path
        
        # Load and prepare data
        print("Loading and preparing data...")
        X_train, X_val, X_test, y_train, y_val, y_test, df = self.prepare_data()
        
        # Hyperparameter tuning if requested
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            best_params = self.hyperparameter_tuning(X_train, y_train, X_val, y_val)
            # Rebuild model with best params (already done in hyperparameter_tuning)
            self.build_model()
        else:
            # Build model with default parameters
            print("Building model...")
            self.build_model()
        
        # Train the model
        print("Training model...")
        self.train(X_train, y_train, X_val, y_val, use_cross_validation)
        
        # Evaluate the model
        print("Evaluating model...")
        results = self.evaluate(X_test, y_test)
        
        # Analyze performance by review length
        print("Analyzing performance by review length...")
        length_analysis = self.analyze_performance_by_length(X_test, y_test, df)
        
        # Interpret model predictions
        print("Interpreting model predictions...")
        test_texts = df['text'].iloc[-len(y_test):].reset_index(drop=True)
        self.interpret_model(test_texts, X_test)
        
        # Plot training history
        if not use_cross_validation:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='lower right')
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            
            plt.tight_layout()
            plt.savefig('results/training_history.png')
            plt.close()
        
        print("\nTraining and evaluation complete!")
        return results

def main():
    parser = argparse.ArgumentParser(description='LSTM Model for Yelp Review Sentiment Analysis')
    
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Path to the Yelp dataset file')
    parser.add_argument('--max_features', type=int, default=10000, 
                        help='Maximum number of words in the vocabulary')
    parser.add_argument('--maxlen', type=int, default=200, 
                        help='Maximum length of reviews to consider')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Dimensionality of the embedding layer')
    parser.add_argument('--lstm_units', type=int, default=64, 
                        help='Number of units in the LSTM layer')
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                        help='Dropout rate for regularization')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Number of samples per batch')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train for')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'rmsprop', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--tune_hyperparams', action='store_true', 
                        help='Perform hyperparameter tuning')
    parser.add_argument('--cross_validation', action='store_true', 
                        help='Use cross-validation for training')
    
    args = parser.parse_args()
    
    # Initialize and run the LSTM model
    lstm_model = YelpSentimentLSTM(
        data_path=args.data_path,
        max_features=args.max_features,
        maxlen=args.maxlen,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=args.optimizer
    )
    
    results = lstm_model.run_end_to_end(
        data_path=args.data_path,
        tune_hyperparams=args.tune_hyperparams,
        use_cross_validation=args.cross_validation
    )
    
    return results

if __name__ == '__main__':
    main()