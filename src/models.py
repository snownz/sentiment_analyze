import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification
from .layers import EmbeddingLayer, LSTMLayer, AttentionLayer

class LSTMSentimentModel(nn.Module):
    """
    LSTM-based model for sentiment analysis.
    """
    def __init__(
        self, 
        vocab_size, 
        embedding_dim=128, 
        hidden_size=64, 
        num_classes=3, 
        num_layers=1, 
        bidirectional=True, 
        dropout=0.2,
        use_attention=True,
        max_length=None,
        padding_idx=0
    ):
        super(LSTMSentimentModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_attention = use_attention
        self.max_length = max_length
        self.padding_idx = padding_idx

        # Embedding layer
        self.embedding = EmbeddingLayer(
            vocab_size = self.vocab_size,
            embedding_dim = self.embedding_dim,
            padding_idx = self.padding_idx,
            dropout = self.dropout,
            max_length = self.max_length
        )
        
        # LSTM layer
        self.lstm = LSTMLayer(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            bidirectional = self.bidirectional,
            dropout = self.dropout if self.num_layers > 1 else 0,
            num_layers = self.num_layers,
            batch_first = True
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention layer
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer( lstm_output_size )
            
        self.classifier = nn.Sequential(
            nn.Dropout( dropout ),
            nn.Linear( lstm_output_size, num_classes )
        )
    
    def forward(self, x):
        
        # Get embeddings
        embedded = self.embedding( x )  # (batch_size, seq_len, embedding_dim)
        
        # Pass through LSTM
        lstm_output, hidden = self.lstm( embedded )  # (batch_size, seq_len, hidden_size * (2 if bidirectional else 1))
        
        if self.use_attention:
            # Apply attention
            context, _ = self.attention( lstm_output )  # (batch_size, hidden_size * (2 if bidirectional else 1))
        else:            
            context = lstm_output[:, -1, :]  # (batch_size, hidden_size * (2 if bidirectional else 1))
        
        # Classify
        output = self.classifier( context )  # (batch_size, num_classes)
        
        return output

class DistilBERTSentimentModel(nn.Module):
    
    def __init__(self, num_classes=3, dropout=0.1, pretrained_model='distilbert-base-uncased'):
        
        super( DistilBERTSentimentModel, self ).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.pretrained_model = pretrained_model
        
        # Load pretrained DistilBERT model
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels = num_classes,
            output_hidden_states = True,
            output_attentions = True
        )
        
        # Set dropout
        self.distilbert.config.dropout = dropout
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Output tensor with logits of shape (batch_size, num_classes)
        """
        outputs = self.distilbert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )
        
        return outputs