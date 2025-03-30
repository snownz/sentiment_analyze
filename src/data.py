import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import nltk
from nltk.corpus import stopwords
import pickle
import os
import json
import logging
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
tqdm.pandas()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class YelpDataProcessor:
    
    def __init__(self, data_path=None, max_length=128, batch_size=32, tokenizer_name='distilbert-base-uncased',
                tokenization_method='bpe'):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the Yelp dataset
            max_length: Maximum sequence length for transformer models
            batch_size: Batch size for training
            tokenizer_name: Name of the tokenizer to use for DistilBERT
            tokenization_method: Method to use for LSTM tokenization ('bpe', 'wordpiece', 'unigram', or 'word')
        """
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenization_method = tokenization_method
        
        # Initialize tokenizer for transformer models
        if tokenizer_name.startswith('distilbert'):
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        else:
            self.bert_tokenizer = None
            
        # Initialize BPE tokenizer for LSTM (will be trained later)
        self.lstm_tokenizer = None
        self.vocab_size = None
        self.word_to_idx = None
        self.idx_to_word = None
    
    def load_data(self, data_path=None):
        """
        Load and preprocess the Yelp dataset.
        If data_path is not provided, it attempts to use the default data path.
        
        Returns:
            DataFrame with reviews and labels
        """
        if data_path is None:
            data_path = self.data_path
        
        # If data_path is still None, try to find the dataset
        if data_path is None:
            possible_paths = [
                'data/yelp_reviews.json',
                'data/yelp_academic_dataset_review.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
        
        if data_path is None or not os.path.exists(data_path):
            logger.warning("Dataset not found. Creating synthetic dataset...")
            try:
                from src.utils import create_synthetic_dataset
            except ImportError:
                # This handles the case when the relative import fails
                from utils import create_synthetic_dataset
                
            synthetic_path = 'data/yelp_reviews.json'
            os.makedirs('data', exist_ok=True)
            create_synthetic_dataset(synthetic_path, n_samples=5000)
            data_path = synthetic_path
        
        # Load from provided path
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_json(data_path, lines=True)
            logger.info(f"Loaded {len(df)} reviews")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
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
            logger.warning("No 'stars' column found. Generating random sentiments.")
            sentiments = np.random.choice(['negative', 'neutral', 'positive'], size=len(df))
            df['sentiment'] = sentiments
        
        # Ensure we have 'text' column
        if 'text' not in df.columns and 'review_text' in df.columns:
            df['text'] = df['review_text']
        
        # Log sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        
        return df[['text', 'sentiment']]
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text: Text string to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
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
    
    def train_bpe_tokenizer(self, texts, vocab_size=10000, save_path='models/bpe_tokenizer.json'):
        """
        Train a BPE tokenizer on the given texts.
        
        Args:
            texts: List of preprocessed texts for training
            vocab_size: Maximum vocabulary size
            save_path: Path to save the trained tokenizer
            
        Returns:
            Trained tokenizer
        """
        logger.info(f"Training BPE tokenizer with vocab size {vocab_size}...")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Initialize a BPE tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        # Add preprocessing components
        tokenizer.normalizer = Lowercase()
        tokenizer.pre_tokenizer = Whitespace()
        
        # Configure post-processing to add special tokens
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )
        
        # Configure the tokenizer trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            min_frequency=2
        )
        
        # Create a list of files to train on
        # Since we have the texts in memory, we'll save them to temporary files
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        # Create a temporary directory for training files
        os.makedirs('temp', exist_ok=True)
        train_file = 'temp/bpe_train.txt'
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Train the tokenizer
        tokenizer.train(files=[train_file], trainer=trainer)
        
        # Save the tokenizer
        tokenizer.save(save_path)
        logger.info(f"BPE tokenizer trained and saved to {save_path}")
        
        # Get vocabulary and mappings
        vocab = tokenizer.get_vocab()
        word_to_idx = vocab
        idx_to_word = {idx: word for word, idx in vocab.items()}
        
        # Store vocabulary information
        self.lstm_tokenizer = tokenizer
        self.vocab_size = len(vocab)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        
        # Save vocab mapping for reference
        with open('models/bpe_vocab.json', 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size
            }, f)
        
        return tokenizer
    
    def load_bpe_tokenizer(self, tokenizer_path='models/bpe_tokenizer.json'):
        """
        Load a trained BPE tokenizer.
        
        Args:
            tokenizer_path: Path to the saved tokenizer
            
        Returns:
            Loaded tokenizer
        """
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
        
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load vocabulary
        vocab = tokenizer.get_vocab()
        self.lstm_tokenizer = tokenizer
        self.vocab_size = len(vocab)
        self.word_to_idx = vocab
        self.idx_to_word = {idx: word for word, idx in vocab.items()}
        
        logger.info(f"Loaded BPE tokenizer from {tokenizer_path} with vocabulary size {self.vocab_size}")
    
    def texts_to_sequences_bpe(self, texts):
        """
        Convert texts to sequences of token IDs using BPE tokenization.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            List of sequences
        """
        if self.lstm_tokenizer is None:
            raise ValueError("BPE tokenizer has not been trained or loaded")
        
        sequences = []
        for text in texts:
            encoded = self.lstm_tokenizer.encode(text)
            sequences.append(encoded.ids)
        
        return sequences
    
    def texts_to_sequences_word(self, texts):
        """
        Convert texts to sequences of word indices using simple word tokenization.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            List of sequences
        """
        if self.word_to_idx is None:
            raise ValueError("Word vocabulary has not been built")
        
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                if word in self.word_to_idx:
                    seq.append(self.word_to_idx[word])
                else:
                    seq.append(self.word_to_idx['<unk>'])
            sequences.append(seq)
        
        return sequences
    
    def build_word_vocab(self, texts, max_vocab_size=10000):
        """
        Build vocabulary from texts for simple word tokenization.
        
        Args:
            texts: List of preprocessed texts
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            word_to_idx: Dictionary mapping words to indices
            idx_to_word: Dictionary mapping indices to words
            vocab_size: Size of vocabulary
        """
        logger.info(f"Building word vocabulary with max size {max_vocab_size}")
        
        # Count word frequencies
        word_counts = {}
        for text in texts:
            for word in text.split():
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
        logger.info(f"Found {len(word_counts)} unique words")
        
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Keep only the top words
        if max_vocab_size < len(sorted_words):
            sorted_words = sorted_words[:max_vocab_size-2]  # Leave room for '<pad>' and '<unk>'
        
        # Create word-to-index and index-to-word mappings
        word_to_idx = {'<pad>': 0, '<unk>': 1}
        for i, (word, _) in enumerate(sorted_words):
            word_to_idx[word] = i + 2
        
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        vocab_size = len(word_to_idx)
        
        logger.info(f"Word vocabulary size: {vocab_size}")
        
        self.vocab_size = vocab_size
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        
        # Save vocab
        os.makedirs('models', exist_ok=True)
        
        with open('models/word_vocab.json', 'w') as f:
            json.dump({
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word,
                'vocab_size': vocab_size
            }, f)
        
        logger.info("Word vocabulary saved to models/word_vocab.json")
        
        return word_to_idx, idx_to_word, vocab_size
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of token IDs using the selected tokenization method.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            List of sequences
        """
        if self.tokenization_method == 'bpe':
            return self.texts_to_sequences_bpe(texts)
        else:  # Default to word tokenization
            return self.texts_to_sequences_word(texts)
    
    def pad_sequences(self, sequences, maxlen=None):
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of sequences
            maxlen: Maximum sequence length
            
        Returns:
            Padded sequences as NumPy array
        """
        if maxlen is None:
            maxlen = self.max_length
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) > maxlen:
                padded_sequences.append(seq[:maxlen])
            else:
                padded_sequences.append(seq + [0] * (maxlen - len(seq)))
        
        return np.array(padded_sequences)
    
    def prepare_data_lstm(self, df=None, max_vocab_size=10000):
        """
        Prepare data for LSTM model using the selected tokenization method.
        
        Args:
            df: DataFrame with reviews and labels
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            train_loader, val_loader, test_loader, df
        """
        logger.info(f"Preparing data for LSTM model using {self.tokenization_method} tokenization")
        
        if df is None:
            df = self.load_data()
        
        # Preprocess text
        logger.info("Preprocessing text")
        df['processed_text'] = df['text'].progress_apply( self.preprocess_text )
        
        # Encode sentiment labels
        logger.info("Encoding labels")
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform( df['sentiment'] )
        
        # Save label encoder
        os.makedirs('models', exist_ok=True)
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Split into train, validation, and test sets
        logger.info("Splitting data into train, validation, and test sets")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df['processed_text'], encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
        )
        
        # Initialize tokenizer based on method
        if self.tokenization_method == 'bpe':
            # Check if we already have a trained BPE tokenizer
            tokenizer_path = 'models/bpe_tokenizer.json'
            if os.path.exists(tokenizer_path):
                logger.info("Loading existing BPE tokenizer")
                self.load_bpe_tokenizer(tokenizer_path)
            else:
                logger.info("Training new BPE tokenizer")
                self.train_bpe_tokenizer(X_train, vocab_size=max_vocab_size)
        else:
            # Use simple word tokenization
            logger.info("Building word vocabulary")
            self.build_word_vocab(X_train, max_vocab_size)
        
        # Convert text to sequences
        logger.info("Converting text to sequences")
        X_train_seq = self.texts_to_sequences(X_train)
        X_val_seq = self.texts_to_sequences(X_val)
        X_test_seq = self.texts_to_sequences(X_test)
        
        # Pad sequences
        logger.info(f"Padding sequences to length {self.max_length}")
        X_train_pad = self.pad_sequences(X_train_seq)
        X_val_pad = self.pad_sequences(X_val_seq)
        X_test_pad = self.pad_sequences(X_test_seq)
        
        # Convert to PyTorch tensors
        logger.info("Converting to PyTorch tensors")
        X_train_tensor = torch.tensor(X_train_pad, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val_pad, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_pad, dtype=torch.long)
        
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create TensorDataset and DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        logger.info(f"Training data: {len(train_dataset)} samples")
        logger.info(f"Validation data: {len(val_dataset)} samples")
        logger.info(f"Test data: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader, df
    
    def prepare_data_bert(self, df=None):
        """
        Prepare data for DistilBERT model.
        
        Args:
            df: DataFrame with reviews and labels
            
        Returns:
            train_dataloader, val_dataloader, test_dataloader, df
        """
        logger.info("Preparing data for DistilBERT model")
        
        if df is None:
            df = self.load_data()
        
        # Preprocess text
        logger.info("Preprocessing text")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Encode sentiment labels
        logger.info("Encoding labels")
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(df['sentiment'])
        
        # Save label encoder
        os.makedirs('models', exist_ok=True)
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Split into train, validation, and test sets
        logger.info("Splitting data into train, validation, and test sets")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df['processed_text'], encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
        )
        
        # Create PyTorch datasets
        logger.info("Creating PyTorch datasets")
        train_dataset = YelpBertDataset(X_train.values, y_train, self.bert_tokenizer, self.max_length)
        val_dataset = YelpBertDataset(X_val.values, y_val, self.bert_tokenizer, self.max_length)
        test_dataset = YelpBertDataset(X_test.values, y_test, self.bert_tokenizer, self.max_length)
        
        # Create DataLoaders
        logger.info(f"Creating DataLoaders with batch size {self.batch_size}")
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
        
        logger.info(f"Training data: {len(train_dataset)} samples")
        logger.info(f"Validation data: {len(val_dataset)} samples")
        logger.info(f"Test data: {len(test_dataset)} samples")
        
        return train_dataloader, val_dataloader, test_dataloader, df

class YelpBertDataset(Dataset):
    """PyTorch Dataset for Yelp reviews using BERT-based models."""
    
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