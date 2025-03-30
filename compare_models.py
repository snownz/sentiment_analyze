import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from src.data import YelpDataProcessor
from src.models import LSTMSentimentModel, DistilBERTSentimentModel
from src.trainer import LSTMTrainer, DistilBERTTrainer
from src.utils import compare_models, set_seed

def main():
    parser = argparse.ArgumentParser(description='Compare LSTM and DistilBERT Models for Yelp Review Sentiment Analysis')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Path to the Yelp dataset file')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for evaluation')
    
    # Model arguments
    parser.add_argument('--lstm_model_path', type=str, default='models/model_best_accuracy.pt',
                        help='Path to the trained LSTM model')
    parser.add_argument('--distilbert_model_path', type=str, default='models/distilbert_best_model.pt',
                        help='Path to the trained DistilBERT model')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data processor
    data_processor = YelpDataProcessor(
        data_path=args.data_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        tokenizer_name='distilbert-base-uncased'
    )
    
    # Load data
    print("Loading data...")
    df = data_processor.load_data()
    
    # Check if models exist
    if not os.path.exists(args.lstm_model_path):
        print(f"LSTM model not found at {args.lstm_model_path}")
        print("Please train the LSTM model first using train_lstm.py")
        return
    
    if not os.path.exists(args.distilbert_model_path):
        print(f"DistilBERT model not found at {args.distilbert_model_path}")
        print("Please train the DistilBERT model first using train_distilbert.py")
        return
    
    # Prepare data for LSTM
    print("Preparing data for LSTM...")
    lstm_train_loader, lstm_val_loader, lstm_test_loader, _ = data_processor.prepare_data_lstm()
    
    # Prepare data for DistilBERT
    print("Preparing data for DistilBERT...")
    bert_train_loader, bert_val_loader, bert_test_loader, _ = data_processor.prepare_data_bert()
    
    # Load LSTM model
    print("Loading LSTM model...")
    with open('models/vocab.json', 'r') as f:
        vocab_info = json.load(f)
    
    lstm_model = LSTMSentimentModel(
        vocab_size=vocab_info['vocab_size'],
        embedding_dim=128,
        hidden_size=64,
        num_classes=len(data_processor.label_encoder.classes_),
        bidirectional=True,
        dropout=0.2,
        use_attention=True,
        max_length=args.max_length
    )
    lstm_model.load_state_dict(torch.load(args.lstm_model_path, map_location=device))
    
    # Load DistilBERT model
    print("Loading DistilBERT model...")
    distilbert_model = DistilBERTSentimentModel(
        num_classes=len(data_processor.label_encoder.classes_),
        dropout=0.1,
        pretrained_model='distilbert-base-uncased'
    )
    distilbert_model.load_state_dict(torch.load(args.distilbert_model_path, map_location=device))
    
    # Initialize trainers
    lstm_trainer = LSTMTrainer(
        model=lstm_model,
        train_loader=lstm_train_loader,
        val_loader=lstm_val_loader,
        test_loader=lstm_test_loader,
        optimizer_name='adam',
        device=device
    )
    
    distilbert_trainer = DistilBERTTrainer(
        model=distilbert_model,
        train_loader=bert_train_loader,
        val_loader=bert_val_loader,
        test_loader=bert_test_loader,
        optimizer_name='adamw',
        device=device
    )
    
    # Evaluate models
    print("Evaluating LSTM model...")
    lstm_results, lstm_preds, lstm_labels = lstm_trainer.evaluate(
        class_names=data_processor.label_encoder.classes_
    )
    
    print("Evaluating DistilBERT model...")
    distilbert_results, distilbert_preds, distilbert_labels = distilbert_trainer.evaluate(
        class_names=data_processor.label_encoder.classes_
    )
    
    # Compare models
    print("Comparing models...")
    comparison = compare_models(
        lstm_results=lstm_results,
        distilbert_results=distilbert_results,
        df=df,
        class_names=data_processor.label_encoder.classes_
    )
    
    print("Comparison completed!")
    
    return comparison

if __name__ == '__main__':
    main()