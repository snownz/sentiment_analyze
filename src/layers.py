import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLayer(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional=True, dropout=0.1):
   
        super(LSTMLayer, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            batch_first = batch_first,
            bidirectional = bidirectional,
            dropout = dropout if bidirectional else 0,
            num_layers = num_layers,
        )
        
        self.dropout = nn.Dropout( dropout )
    
    def forward(self, x):

        outputs, ( hidden, cell ) = self.lstm( x )
        
        # Apply dropout
        outputs = self.dropout( outputs )
        
        return outputs, hidden

class AttentionLayer(nn.Module):

    def __init__(self, hidden_size, attention_size=64):
        
        super(AttentionLayer, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear( hidden_size, attention_size ),
            nn.Tanh(),
            nn.Linear( attention_size, 1 )
        )
    
    def forward(self, x):

        # Calculate attention weights
        attention_weights = self.attention( x )  # (batch_size, seq_len, 1)
        attention_weights = F.softmax( attention_weights, dim = 1 )  # (batch_size, seq_len, 1)
        
        # Apply attention weights to get context vector
        context_vector = torch.sum( x * attention_weights, dim = 1 )  # (batch_size, hidden_size)
        
        return context_vector, attention_weights

class EmbeddingLayer(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, padding_idx=0, dropout=0.1, max_length=None):
        
        super(EmbeddingLayer, self).__init__()
        
        self.word_embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = padding_idx
        )
        
        self.position_embeddings = None
        if max_length is not None:
            self.position_embeddings = nn.Embedding( max_length, embedding_dim )
        
        self.dropout = nn.Dropout( dropout )
    
    def forward(self, x):
        
        embeddings = self.word_embeddings( x )
        
        if self.position_embeddings is not None:
            seq_length = x.size( 1 )
            position_ids = torch.arange( seq_length, dtype = torch.long, device = x.device )
            position_ids = position_ids.unsqueeze(0).expand_as( x )
            position_embeddings = self.position_embeddings( position_ids )
            embeddings = embeddings + position_embeddings
        
        embeddings = self.dropout( embeddings )
        
        return embeddings