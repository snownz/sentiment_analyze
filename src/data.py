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
    
    def __init__(self, data_path=None, max_length=128, batch_size=32, tokenizer_name='distilbert-base-uncased'):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the Yelp dataset
            max_length: Maximum sequence length for transformer models
            batch_size: Batch size for training
            tokenizer_name: Name of the tokenizer to use
        """
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize tokenizer for transformer models
        if tokenizer_name.startswith('distilbert'):
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None
            
        self.label_encoder = None
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
    
    def build_vocab(self, texts, max_vocab_size=10000):
        """
        Build vocabulary from texts for LSTM model.
        
        Args:
            texts: List of preprocessed texts
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            word_to_idx: Dictionary mapping words to indices
            idx_to_word: Dictionary mapping indices to words
            vocab_size: Size of vocabulary
        """
        logger.info(f"Building vocabulary with max size {max_vocab_size}")
        
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
        
        logger.info(f"Vocabulary size: {vocab_size}")
        
        self.vocab_size = vocab_size
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        
        # Save vocab
        os.makedirs('models', exist_ok=True)
        
        with open('models/vocab.json', 'w') as f:
            json.dump({
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word,
                'vocab_size': vocab_size
            }, f)
        
        logger.info("Vocabulary saved to models/vocab.json")
        
        return word_to_idx, idx_to_word, vocab_size
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of word indices for LSTM model.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            List of sequences
        """
        if self.word_to_idx is None:
            raise ValueError("Vocabulary has not been built")
        
        sequences = []
        for text in texts:
            seq = []
            for word in text.split():
                if word in self.word_to_idx:
                    seq.append( self.word_to_idx[word] )
                else:
                    seq.append( self.word_to_idx['<unk>'] )
            sequences.append( seq )
        
        return sequences
    
    def pad_sequences(self, sequences, maxlen=None):
        """
        Pad sequences to the same length for LSTM model.
        
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
        Prepare data for LSTM model.
        
        Args:
            df: DataFrame with reviews and labels
            max_vocab_size: Maximum vocabulary size
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, df
        """
        logger.info("Preparing data for LSTM model")
        
        if df is None:
            df = self.load_data()
        
        # Preprocess text
        logger.info("Preprocessing text")
        df['processed_text'] = df['text'].progress_apply( self.preprocess_text )
        # df['processed_text'] = df['text']
        
        # Encode sentiment labels
        logger.info( "Encoding labels" )
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform( df['sentiment'] )
        
        # Save label encoder
        os.makedirs( 'models', exist_ok = True )
        with open( 'models/label_encoder.pkl', 'wb' ) as f:
            pickle.dump( self.label_encoder, f )
        
        # Split into train, validation, and test sets
        logger.info( "Splitting data into train, validation, and test sets" )
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df['processed_text'], encoded_labels, test_size = 0.2, stratify = encoded_labels, random_state = 42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size = 0.25, stratify = y_train_val, random_state = 42
        )
        
        # Build vocabulary
        self.build_vocab( X_train, max_vocab_size )
        
        # Convert text to sequences
        logger.info( "Converting text to sequences" )
        X_train_seq = self.texts_to_sequences( X_train )
        X_val_seq = self.texts_to_sequences( X_val )
        X_test_seq = self.texts_to_sequences( X_test )
        
        # Pad sequences
        logger.info( f"Padding sequences to length {self.max_length}" )
        X_train_pad = self.pad_sequences( X_train_seq )
        X_val_pad = self.pad_sequences( X_val_seq )
        X_test_pad = self.pad_sequences( X_test_seq )
        
        # Convert to PyTorch tensors
        logger.info( "Converting to PyTorch tensors" )
        X_train_tensor = torch.tensor( X_train_pad, dtype = torch.long )
        X_val_tensor = torch.tensor( X_val_pad, dtype = torch.long )
        X_test_tensor = torch.tensor( X_test_pad, dtype = torch.long )
        
        y_train_tensor = torch.tensor( y_train, dtype = torch.long )
        y_val_tensor = torch.tensor( y_val, dtype = torch.long )
        y_test_tensor = torch.tensor(y_test, dtype = torch.long )
        
        # Create TensorDataset and DataLoader
        train_dataset = torch.utils.data.TensorDataset( X_train_tensor, y_train_tensor )
        val_dataset = torch.utils.data.TensorDataset( X_val_tensor, y_val_tensor )
        test_dataset = torch.utils.data.TensorDataset( X_test_tensor, y_test_tensor )
        
        train_loader = DataLoader( train_dataset, batch_size = self.batch_size, shuffle = True )
        val_loader = DataLoader( val_dataset, batch_size = self.batch_size )
        test_loader = DataLoader( test_dataset, batch_size = self.batch_size )
        
        logger.info( f"Training data: {len(train_dataset)} samples" )
        logger.info( f"Validation data: {len(val_dataset)} samples" )
        logger.info( f"Test data: {len(test_dataset)} samples" )
        
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
        train_dataset = YelpBertDataset(X_train.values, y_train, self.tokenizer, self.max_length)
        val_dataset = YelpBertDataset(X_val.values, y_val, self.tokenizer, self.max_length)
        test_dataset = YelpBertDataset(X_test.values, y_test, self.tokenizer, self.max_length)
        
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