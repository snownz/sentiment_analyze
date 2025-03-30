import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
import logging
import copy
from src.data import YelpDataProcessor
from src.models import LSTMSentimentModel
from src.trainer import LSTMTrainer
from src.utils import set_seed

# Set up logging
logging.basicConfig( level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s' )
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open( config_path, 'r' ) as f:
        config = yaml.safe_load( f )    
    return config

def save_config(config, config_path):
    with open( config_path, 'w' ) as f:
        yaml.dump( config, f, default_flow_style = False )
    logger.info(f"Configuration saved to {config_path}")

CONFIG_FILE = 'model_configs/lstm_tuning_v1.yaml'
FORCE_TUNING = False

# Load configuration
logger.info( f"Loading configuration from {CONFIG_FILE}" )
config = load_config( CONFIG_FILE )

# Extract configuration values
data_config = config.get( 'data', {} )
model_config = config.get( 'model', {} )
training_config = config.get( 'training', {} )
hp_tuning_config = config.get( 'hyperparameter_tuning', {} )

# Set random seed
seed = training_config.get( 'seed', 42 )
set_seed( seed )

# Create directories
os.makedirs( 'models', exist_ok = True )
os.makedirs( 'results', exist_ok = True )

# Determine if hyperparameter tuning is needed
hp_tuning_enabled = hp_tuning_config.get( 'enabled', False )
best_params = hp_tuning_config.get( 'best_params', None )

need_tuning = ( hp_tuning_enabled and ( best_params is None or FORCE_TUNING ) )

# Set device
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
logger.info( f"Using device: {device}" )

# Initialize data processor
data_processor = YelpDataProcessor(
    data_path = data_config.get( 'path' ),
    max_length = data_config.get( 'max_length', 128 ),
    batch_size = data_config.get( 'batch_size', 32 ),
    tokenization_method = data_config.get( 'tokenization_method', 'bpe' )
)

# Prepare data
logger.info( "Preparing data..." )
train_dataset, train_loader, val_loader, test_loader, df = data_processor.prepare_data_lstm(
    max_vocab_size = data_config.get( 'max_vocab_size', 10000 )
)

# Apply best parameters if they exist, otherwise use defaults from config
effective_model_config = copy.deepcopy( model_config )
effective_training_config = copy.deepcopy( training_config )

if best_params is not None and not FORCE_TUNING:

    logger.info("Using best hyperparameters from config")
    
    # Update model configuration with best parameters
    if 'embedding_dim' in best_params:
        effective_model_config['embedding_dim'] = best_params['embedding_dim']
    if 'hidden_size' in best_params:
        effective_model_config['hidden_size'] = best_params['hidden_size']
    if 'dropout' in best_params:
        effective_model_config['dropout'] = best_params['dropout']
    if 'bidirectional' in best_params:
        effective_model_config['bidirectional'] = best_params['bidirectional']
    if 'attention' in best_params:
        effective_model_config['use_attention'] = best_params['attention']
    
    # Update training configuration with best parameters
    if 'optimizer' in best_params:
        effective_training_config['optimizer'] = best_params['optimizer']
    if 'learning_rate' in best_params:
        effective_training_config['learning_rate'] = best_params['learning_rate']
    if 'weight_decay' in best_params:
        effective_training_config['weight_decay'] = best_params['weight_decay']

# Build model with configuration
logger.info("Building model...")
model = LSTMSentimentModel(
    vocab_size = data_processor.vocab_size,
    embedding_dim = effective_model_config.get( 'embedding_dim', 128 ),
    hidden_size = effective_model_config.get( 'hidden_size', 64 ),
    num_classes = len( data_processor.label_encoder.classes_ ),
    bidirectional = effective_model_config.get( 'bidirectional', True ),
    dropout = effective_model_config.get( 'dropout', 0.2 ),
    use_attention = effective_model_config.get( 'use_attention', True ),
    max_length = data_config.get( 'max_length', 128 ),
    padding_idx = 0,
    num_layers = effective_model_config.get( 'num_layers', 1 )
)

# Initialize trainer
trainer = LSTMTrainer(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    test_loader = test_loader,
    optimizer_name = effective_training_config.get( 'optimizer', 'adam' ),
    learning_rate = effective_training_config.get( 'learning_rate', 0.001 ),
    weight_decay = effective_training_config.get( 'weight_decay', 0.0 ),
    device = device,
    name = config.get( 'name', 'lstm_model' ),
)

# Get training data for hyperparameter tuning if needed
# need_tuning = False
if need_tuning:
    
    logger.info( f"Performing hyperparameter tuning with {hp_tuning_config.get('n_trials', 30)} trials "
                 f"and {hp_tuning_config.get('cv_folds', 5)}-fold cross-validation..." )
    
    # Perform hyperparameter tuning
    with torch.amp.autocast( enabled = True, device_type = device.type ):
        best_params = trainer.hyperparameter_tuning(
            dataset = train_dataset,
            n_epochs = hp_tuning_config.get( 'n_epochs', 10 ),
            n_trials = hp_tuning_config.get( 'n_trials', 30 ),
            cross_validation = hp_tuning_config.get( 'cv_folds', 5 )
        )
    
    # Update configuration with best parameters
    config['hyperparameter_tuning']['best_params'] = best_params
    
    # Save updated configuration
    save_config( config, CONFIG_FILE )
    
    # Rebuild model with best hyperparameters
    logger.info("Rebuilding model with best hyperparameters...")
    model = LSTMSentimentModel(
        vocab_size = data_processor.vocab_size,
        embedding_dim = best_params.get( 'embedding_dim', effective_model_config.get('embedding_dim', 128 ) ),
        hidden_size = best_params.get( 'hidden_size', effective_model_config.get( 'hidden_size', 64 ) ),
        num_classes = len( data_processor.label_encoder.classes_ ),
        bidirectional = best_params.get( 'bidirectional', effective_model_config.get( 'bidirectional', True ) ),
        dropout = best_params.get( 'dropout', effective_model_config.get( 'dropout', 0.2 ) ),
        use_attention = best_params.get( 'attention', effective_model_config.get( 'use_attention', True ) ),
        max_length = data_config.get( 'max_length', 128 ),
        padding_idx = 0,
    )
    
    # Reinitialize trainer with best parameters
    trainer = LSTMTrainer(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        optimizer_name = best_params.get( 'optimizer', effective_training_config.get( 'optimizer', 'adam' ) ),
        learning_rate = best_params.get( 'learning_rate', effective_training_config.get( 'learning_rate', 0.001 ) ),
        weight_decay = best_params.get( 'weight_decay', effective_training_config.get( 'weight_decay', 0.0 ) ),
        device = device,
        name = config.get( 'name', 'lstm_model' ),
    )

# Print model summary
logger.info( model )

# Train the model
logger.info( f"Training LSTM model for {training_config.get( 'epochs', 10 )} epochs..." )

scheduler_name = effective_training_config.get( 'scheduler', 'plateau' )
if scheduler_name == 'none':
    scheduler_name = None

with torch.amp.autocast( enabled = True, device_type = device.type ):

    history = trainer.train(
        trainer.train_loader,
        trainer.val_loader,
        epochs = training_config.get( 'epochs', 10 ),
        scheduler_name = scheduler_name,
        full_train = True
    )

    # Evaluate the model
    logger.info("Evaluating LSTM model...")
    results, test_preds, test_labels = trainer.evaluate(
        trainer.test_loader,
        class_names = data_processor.label_encoder.classes_
    )

# Analyze by review length
trainer.analyze_by_length( df, test_preds, test_labels )

logger.info( "Training and evaluation completed!" )
logger.info( f"Final accuracy: {results['accuracy']:.4f}" )
logger.info( f"Final F1 score: {results['f1_score']:.4f}" )