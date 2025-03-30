import argparse
import os
import yaml
import torch
import numpy as np
import pandas as pd
import logging
import copy
from src.data import YelpDataProcessor
from src.models import DistilBERTSentimentModel
from src.trainer import DistilBERTTrainer
from src.utils import set_seed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_path}")


CONFIG_FILE = 'model_configs/distilbert_tuning_v1.yaml'
FORCE_TUNING = False

# Load configuration
logger.info(f"Loading configuration from {CONFIG_FILE}")
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
force_tuning = FORCE_TUNING

need_tuning = ( hp_tuning_enabled and ( best_params is None or force_tuning ) )

# Set device
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
logger.info( f"Using device: {device}" )

# Initialize data processor
data_processor = YelpDataProcessor(
    data_path = data_config.get( 'path' ),
    max_length = data_config.get( 'max_length', 128 ),
    batch_size = data_config.get( 'batch_size', 16 ),
    tokenizer_name = model_config.get( 'pretrained_model', 'distilbert-base-uncased' )
)

# Prepare data
logger.info( "Preparing data..." )
train_dataset, train_loader, val_loader, test_loader, df = data_processor.prepare_data_bert()

# Apply best parameters if they exist, otherwise use defaults from config
effective_model_config = copy.deepcopy( model_config )
effective_training_config = copy.deepcopy( training_config )

if best_params is not None and not force_tuning:
    logger.info( "Using best hyperparameters from config" )
    
    # Update model configuration with best parameters
    if 'dropout' in best_params:
        effective_model_config['dropout'] = best_params['dropout']
    
    # Update training configuration with best parameters
    if 'optimizer' in best_params:
        effective_training_config['optimizer'] = best_params['optimizer']
    if 'learning_rate' in best_params:
        effective_training_config['learning_rate'] = best_params['learning_rate']
    if 'weight_decay' in best_params:
        effective_training_config['weight_decay'] = best_params['weight_decay']

# Build model with configuration
logger.info( "Building model..." )
model = DistilBERTSentimentModel(
    num_classes = len( data_processor.label_encoder.classes_ ),
    dropout = effective_model_config.get( 'dropout', 0.1 ),
    pretrained_model = effective_model_config.get( 'pretrained_model', 'distilbert-base-uncased' )
)

# Initialize trainer
trainer = DistilBERTTrainer(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    test_loader = test_loader,
    optimizer_name = effective_training_config.get( 'optimizer', 'adamw' ),
    learning_rate = effective_training_config.get( 'learning_rate', 2e-5 ),
    weight_decay = effective_training_config.get( 'weight_decay', 0.01 ),
    device = device,
    name = model_config.get( 'name', 'distilbert_sentiment_model' ),
)

# Get training data for hyperparameter tuning if needed
if need_tuning:
    
    logger.info( f"Performing hyperparameter tuning with {hp_tuning_config.get('n_trials', 20)} trials "
                 f"and {hp_tuning_config.get('cv_folds', 3)}-fold cross-validation..." )
    
    # Perform hyperparameter tuning
    with torch.amp.autocast( enabled = True, device_type = device.type ):
        best_params = trainer.hyperparameter_tuning(
            dataset = train_dataset,
            n_trials = hp_tuning_config.get( 'n_trials', 20 ),
            cross_validation = hp_tuning_config.get( 'cv_folds', 3 )
        )
    
    # Update configuration with best parameters
    config['hyperparameter_tuning']['best_params'] = best_params
    
    # Save updated configuration
    save_config( config, CONFIG_FILE )
    
    # Rebuild model with best hyperparameters
    logger.info( "Rebuilding model with best hyperparameters..." )
    model = DistilBERTSentimentModel(
        num_classes = len( data_processor.label_encoder.classes_ ),
        dropout = best_params.get( 'dropout', effective_model_config.get( 'dropout', 0.1 ) ),
        pretrained_model = effective_model_config.get( 'pretrained_model', 'distilbert-base-uncased' )
    )
    
    # Reinitialize trainer with best parameters
    trainer = DistilBERTTrainer(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        test_loader = test_loader,
        optimizer_name = best_params.get( 'optimizer', effective_training_config.get( 'optimizer', 'adamw' ) ),
        learning_rate = best_params.get( 'learning_rate', effective_training_config.get( 'learning_rate', 2e-5 ) ),
        weight_decay = best_params.get( 'weight_decay', effective_training_config.get( 'weight_decay', 0.01 ) ),
        device = device,
        name = model_config.get( 'name', 'distilbert_sentiment_model' ),
    )

# Print model summary
logger.info( model )

# Train the model
logger.info( f"Training DistilBERT model for {training_config.get('epochs', 4)} epochs..." )

scheduler_name = effective_training_config.get( 'scheduler', 'onecycle' )
if scheduler_name == 'none':
    scheduler_name = None

with torch.amp.autocast( enabled = True, device_type = device.type ):
    history = trainer.train(
        train_loader = trainer.train_loader,
        val_loader = trainer.val_loader,
        epochs = training_config.get('epochs', 4),
        scheduler_name = scheduler_name,
        full_train = True
    )

    # Evaluate the model
    logger.info( "Evaluating DistilBERT model..." )
    results, test_preds, test_labels = trainer.evaluate(
        test_loader = trainer.test_loader,
        class_names = data_processor.label_encoder.classes_
    )
# Analyze by review length
trainer.analyze_by_length( df, test_preds, test_labels )

logger.info( "Training and evaluation completed!" )
logger.info( f"Final accuracy: {results['accuracy']:.4f}" )
logger.info( f"Final F1 score: {results['f1_score']:.4f}" )
