import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import optuna
from torch.utils.data import DataLoader

from src.models import LSTMSentimentModel, DistilBertForSequenceClassification
import gc
import pandas as pd

class ModelTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer_name='adam',
        learning_rate=0.001,
        weight_decay=0.0,
        device=None,
        name=None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            optimizer_name: Name of the optimizer (adam, sgd, rmsprop)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to train on (cpu or cuda)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer_name = optimizer_name.lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.name = name
        
        # Set device
        if device is None:
            self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'lr': []
        }
        
        # Create directories for saving models
        os.makedirs( os.path.join( 'models', self.name ), exist_ok = True )
        os.makedirs( os.path.join( 'results', self.name ), exist_ok = True )
    
    def _get_optimizer(self, name=None, lr=None, wd=None):
        """
        Get optimizer based on name.
        
        Args:
            name: Optimizer name (if None, use self.optimizer_name)
            lr: Learning rate (if None, use self.learning_rate)
            wd: Weight decay (if None, use self.weight_decay)
            
        Returns:
            PyTorch optimizer
        """
        if name is None:
            name = self.optimizer_name
        if lr is None:
            lr = self.learning_rate
        if wd is None:
            wd = self.weight_decay
            
        if name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr = lr,
                momentum = 0.9,
                weight_decay = wd
            )
        elif name == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr = lr,
                weight_decay = wd
            )
        elif name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr = lr,
                weight_decay = wd
            )
        else:  # Default to Adam
            return optim.Adam(
                self.model.parameters(),
                lr = lr,
                weight_decay = wd
            )
    
    def get_scheduler(self, scheduler_name='plateau', epochs=10):
        """
        Get learning rate scheduler.
        
        Args:
            scheduler_name: Name of the scheduler (plateau, onecycle)
            epochs: Number of epochs
            
        Returns:
            PyTorch scheduler
        """
        if scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode = 'min',
                factor = 0.5,
                patience = 2,
                verbose = True
            )
        elif scheduler_name == 'onecycle':
            steps_per_epoch = len( self.train_loader )
            return OneCycleLR(
                self.optimizer,
                max_lr = self.learning_rate,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                anneal_strategy = 'cos'
            )
        else:
            return None
    
    def train_step(self, batch):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss value
        """
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        if isinstance(batch, dict):  # For BERT-based models
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
        else:  # For LSTM-based models
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            loss: Average loss
            accuracy: Accuracy
            f1: F1 score (macro-averaged)
            predictions: Model predictions
            labels: True labels
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Evaluate without gradient calculation
        with torch.no_grad():
            for batch in data_loader:
                # Forward pass
                if isinstance(batch, dict):  # For BERT-based models
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                else:  # For LSTM-based models
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    logits = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                
                # Get predictions
                _, preds = torch.max(logits, dim=1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    def train(self, train_loader, val_loader=None, epochs=10, scheduler_name='plateau', full_train=True):
        """
        Train the model.
        
        Args:
            epochs: Number of epochs
            scheduler_name: Name of the scheduler
            full_train: Whether to train, validate and save the model
            
        Returns:
            Training history
        """
        # Get scheduler
        scheduler = self.get_scheduler( scheduler_name, epochs )
        
        # Initialize best metrics
        best_val_loss = float( 'inf' )
        best_val_f1 = 0.0
        
        # Start training
        start_time = time.time()
        for epoch in range( epochs ):
            
            # Training phase
            train_loss = 0
            
            # Use tqdm for progress bar
            progress_bar = tqdm( train_loader, desc = f"Training Epoch {epoch + 1}/{epochs}", unit = 'batch' )
            
            self.model.train()
            for batch in progress_bar:

                # Perform training step
                batch_loss = self.train_step( batch )
                train_loss += batch_loss
                
                # Update progress bar
                progress_bar.set_postfix( { 'loss': batch_loss } )
                
                # Update learning rate for OneCycleLR
                if scheduler_name == 'onecycle':
                    scheduler.step()
            
            # Calculate average training loss
            avg_train_loss = train_loss / len( train_loader )
            
            # Update learning rate for ReduceLROnPlateau
            if scheduler_name == 'plateau':
                scheduler.step( avg_train_loss )
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if full_train and val_loader is not None:

                # Validation phase
                val_loss, val_accuracy, val_f1, _, _ = self.eval_step( val_loader )
                
                # Update history
                self.history['train_loss'].append( avg_train_loss )
                self.history['val_loss'].append( val_loss )
                self.history['val_accuracy'].append( val_accuracy )
                self.history['val_f1'].append( val_f1 )
                self.history['lr'].append( current_lr )
                
                # Print metrics
                print( f"Train Loss: {avg_train_loss:.4f}" )
                print( f"Val Loss: {val_loss:.4f}" )
                print( f"Val Accuracy: {val_accuracy:.4f}" )
                print( f"Val F1 Score: {val_f1:.4f}" )
                print( f"Learning Rate: {current_lr:.6f}" )
                
                # Save best model
                if full_train:
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        torch.save( self.model.state_dict(), os.path.join( 'models', self.name, 'model_best_f1.pt' ) )
                        print( f"Saved best model with F1 score: {val_f1:.4f}" )
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save( self.model.state_dict(), os.path.join( 'models', self.name, 'model_best_loss.pt' ) )
                        print( f"Saved best model with loss: {val_loss:.4f}" )
        
        # Calculate training time
        training_time = time.time() - start_time
        print( f"\nTraining completed in {training_time:.2f} seconds" )
        
        # Save final model
        if full_train:
            os.makedirs( os.path.join( 'models', self.name ), exist_ok = True )
            torch.save( self.model.state_dict(), 'models/model_final.pt' )
        
            # Plot training history
            self.plot_history()
        
        return self.history
    
    def evaluate(self, loader, class_names=None):
        """
        Evaluate the model on the test set.
        
        Args:
            class_names: Names of the classes
            
        Returns:
            Evaluation metrics
        """
        
        # Load best model
        # if os.path.exists('models/model_best_f1.pt'):
        if os.path.exists( os.path.join( 'models', self.name, 'model_best_f1.pt' ) ):
            self.model.load_state_dict( torch.load( os.path.join( 'models', self.name, 'model_best_f1.pt' ) ) )
            print("Loaded best model based on F1 score")
        
        # Evaluate on test set
        test_loss, test_accuracy, test_f1, test_preds, test_labels = self.eval_step( loader )
        
        # Print metrics
        print( f"\nTest Loss: {test_loss:.4f}" )
        print( f"Test Accuracy: {test_accuracy:.4f}" )
        print( f"Test F1 Score: {test_f1:.4f}" )
        
        # Classification report
        if class_names is None:
            class_names = ['negative', 'neutral', 'positive']
            
        report = classification_report( test_labels, test_preds, target_names = class_names )
        print( "\nClassification Report:" )
        print( report )
        
        # Confusion matrix
        cm = confusion_matrix( test_labels, test_preds )
        
        # Plot confusion matrix
        plt.figure( figsize = ( 8, 6 ) )
        sns.heatmap( cm, annot = True, fmt = 'd', cmap = 'Blues', 
                     xticklabels = class_names, yticklabels = class_names )
        plt.xlabel( 'Predicted' )
        plt.ylabel( 'True' )
        plt.title( 'Confusion Matrix' )
        plt.tight_layout()
        plt.savefig( os.path.join( 'results', self.name, 'confusion_matrix.png' ) )
        plt.close()
        
        # Save evaluation results
        results = {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'loss': test_loss,
            'classification_report': classification_report( test_labels, test_preds, target_names = class_names, output_dict = True ),
            'confusion_matrix': cm.tolist()
        }
        
        return results, test_preds, test_labels
    
    def plot_history(self):
        """
        Plot training history.
        """
        # Create a figure with subplots
        plt.figure(figsize=(20, 5))
        
        # Plot training and validation loss
        plt.subplot( 1, 4, 1 )
        plt.plot( self.history['train_loss'], label = 'Training Loss' )
        plt.plot( self.history['val_loss'], label = 'Validation Loss' )
        plt.title( 'Loss' )
        plt.xlabel( 'Epoch' )
        plt.ylabel( 'Loss' )
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot( 1, 4, 2 )
        plt.plot( self.history['val_accuracy'], label = 'Validation Accuracy' )
        plt.title( 'Accuracy' )
        plt.xlabel( 'Epoch' )
        plt.ylabel( 'Accuracy' )
        plt.legend()
        
        # Plot validation F1 score
        plt.subplot( 1, 4, 3 )
        plt.plot( self.history['val_f1'], label = 'Validation F1 Score' )
        plt.title( 'F1 Score' )
        plt.xlabel( 'Epoch' )
        plt.ylabel( 'F1 Score' )
        plt.legend()
        
        # Plot learning rate
        plt.subplot( 1, 4, 4 )
        plt.plot( self.history['lr'], label = 'Learning Rate' )
        plt.title( 'Learning Rate' )
        plt.xlabel( 'Epoch' )
        plt.ylabel( 'Learning Rate' )
        plt.legend()
        
        plt.tight_layout()
        plt.savefig( os.path.join( 'results', self.name, 'training_history.png' ) )
        plt.close()
    
    def analyze_by_length(self, df, test_preds, test_labels):
        """
        Analyze model performance by review length.
        
        Args:
            df: DataFrame with processed texts
            test_preds: Model predictions
            test_labels: True labels
            
        Returns:
            Accuracy by length category
        """
        
        # Get test set indices
        test_size = len( test_labels )
        test_texts = df['processed_text'].iloc[-test_size:].reset_index( drop = True )
        
        # Calculate text lengths
        text_lengths = [ len( text.split() ) for text in test_texts ]
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame( {
            'text_length': text_lengths,
            'true_label': test_labels,
            'predicted_label': test_preds,
            'correct': np.array( test_labels ) == np.array( test_preds )
        } )
        
        # Define length categories
        length_bins = [ 0, 50, 100, 200, float('inf') ]
        length_labels = [ 'Very Short (0-50)', 'Short (51-100)', 'Medium (101-200)', 'Long (>200)' ]
        
        analysis_df['length_category'] = pd.cut(
            analysis_df['text_length'], 
            bins = length_bins, 
            labels = length_labels, 
            right = False
        )
        
        # Calculate accuracy by length category
        accuracy_by_length = analysis_df.groupby('length_category')['correct'].mean()
        
        # Calculate F1 score by length category
        f1_by_length = {}
        precision_by_length = {}
        for category in length_labels:
            category_df = analysis_df[ analysis_df['length_category'] == category ]
            if len( category_df ) > 0:
                f1_by_length[category] = f1_score(
                    category_df['true_label'], 
                    category_df['predicted_label'], 
                    average = 'macro'
                )
                precision_by_length[category] = precision_score(
                    category_df['true_label'], 
                    category_df['predicted_label'], 
                    average = 'macro'
                )
            else:
                f1_by_length[category] = 0.0
                precision_by_length[category] = 0.0
        
        # Plot results
        plt.figure( figsize = ( 12, 6 ) )
        
        plt.subplot(1, 3, 1)
        sns.barplot( x =accuracy_by_length.index, y = accuracy_by_length.values )
        plt.title( 'Model Accuracy by Review Length' )
        plt.xlabel( 'Review Length Category' )
        plt.ylabel( 'Accuracy' )
        plt.ylim( 0, 1.0 )
        
        plt.subplot( 1, 3, 2 )
        sns.barplot( x = list( f1_by_length.keys() ), y = list( f1_by_length.values() ) )
        plt.title( 'Model F1 Score by Review Length' )
        plt.xlabel( 'Review Length Category' )
        plt.ylabel( 'F1 Score' )
        plt.ylim( 0, 1.0 )

        plt.subplot( 1, 3, 3 )
        sns.barplot( x = list( precision_by_length.keys() ), y = list( precision_by_length.values() ) )
        plt.title( 'Model Precision by Review Length' )
        plt.xlabel( 'Review Length Category' )
        plt.ylabel( 'Precision' )
        plt.ylim( 0, 1.0 )
        
        plt.tight_layout()
        plt.savefig( os.path.join( 'results', self.name, 'performance_by_length.png' ) )
        plt.close()
        
        # Print results
        print("\nPerformance by Review Length:")
        for category in length_labels:
            if category in accuracy_by_length and category in f1_by_length:
                print( f"{category}: Accuracy={accuracy_by_length[category]:.4f}, F1 Score={f1_by_length[category]:.4f}, Precision={precision_by_length[category]:.4f}" )
        
        return accuracy_by_length, f1_by_length
    
    def hyperparameter_tuning(self, X_train, y_train, n_epochs=2, n_trials=30, cross_validation=5):
        """
        Perform hyperparameter tuning using Optuna with cross-validation.
        
        Args:
            X_train: Training data
            y_train: Training labels
            n_epochs: Number of epochs for training
            n_trials: Number of Optuna trials
            cross_validation: Number of cross-validation folds
            
        Returns:
            Best hyperparameters
        """

        # Define objective function for Optuna
        def objective(trial):
            
            # Define hyperparameters to optimize
            params = {
                'optimizer': trial.suggest_categorical( 'optimizer', [ 'adam', 'adamw', 'sgd', 'rmsprop' ] ),
                'learning_rate': trial.suggest_float( 'learning_rate', 1e-5, 1e-2, log = True ),
                'weight_decay': trial.suggest_float( 'weight_decay', 1e-6, 1e-3, log = True ),
                'batch_size': trial.suggest_categorical( 'batch_size', [ 512, 1024, 2048 ] ),
            }
            
            # Add model-specific hyperparameters
            if hasattr(self, 'is_lstm') and self.is_lstm:
                params.update({
                    'hidden_size': trial.suggest_categorical( 'hidden_size', [ 32, 64, 128, 256 ] ),
                    'embedding_dim': trial.suggest_categorical( 'embedding_dim', [ 64, 128, 256 ] ),
                    'dropout': trial.suggest_float( 'dropout', 0.1, 0.5 ),
                    'bidirectional': trial.suggest_categorical( 'bidirectional', [ True, False ] ),
                    'attention': trial.suggest_categorical( 'attention', [ True, False ] ),
                })
            else:  # DistilBERT-specific hyperparameters
                params.update({
                    'dropout': trial.suggest_float( 'dropout', 0.1, 0.3 ),
                })
            
            print("========================================")
            print(f"Trial {trial.number}: \n{params}")
            print("========================================")

            # Implement K-fold cross-validation
            kf = KFold( n_splits = cross_validation, shuffle = True, random_state = 42 )
            
            f1_scores = []
            
            for fold, ( train_idx, val_idx ) in enumerate( kf.split( X_train ) ):
                
                # Reset model for each fold
                self._reset_model(params)
                
                # Get fold data
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                train_loader = DataLoader(
                    list( zip( X_train_fold, y_train_fold ) ),
                    batch_size = params['batch_size'],
                    shuffle = True
                )
                val_loader = DataLoader(
                    list( zip( X_val_fold, y_val_fold ) ),
                    batch_size = params['batch_size'],
                    shuffle = False
                )
                
                # Train for fewer epochs during tuning
                self.train( train_loader, epochs = n_epochs, full_train = False )
                
                # Evaluate on validation fold
                _, _, val_f1, _, _ = self.eval_step( val_loader )
                f1_scores.append( val_f1 )
                
                # Report intermediate objective value
                trial.report( val_f1, fold )
                
                # Handle pruning (early stopping of unpromising trials)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Return average F1 score across folds
            print(f"Trial {trial.number}: F1 score = {np.mean(f1_scores):.4f}")

            return np.mean( f1_scores )
        
        # Create a study and optimize
        study = optuna.create_study( direction = 'maximize' )
        study.optimize( objective, n_trials = n_trials )
        
        # Get best parameters and score
        best_params = study.best_params
        best_score = study.best_value
        
        print( f"\nBest parameters: {best_params}" )
        print( f"Best CV F1 score: {best_score:.4f}" )
        
        # Plot optimization history
        plt.figure( figsize = ( 10, 6 ) )
        optuna.visualization.matplotlib.plot_optimization_history( study )
        plt.tight_layout()
        plt.savefig( os.path.join( 'results', self.name, 'optuna_history.png' ) )
        plt.close()
        
        # Plot parameter importances
        plt.figure( figsize = ( 10, 6 ) )
        optuna.visualization.matplotlib.plot_param_importances( study )
        plt.tight_layout()
        plt.savefig( os.path.join( 'results', self.name, 'optuna_param_importances.png' ) )
        plt.close()
        
        # Reset model with best parameters
        self._reset_model( best_params )
        
        return best_params
    
    def _reset_model(self, params):
        """
        Reset model with new hyperparameters. To be implemented by subclasses.
        
        Args:
            params: New hyperparameters
        """
        raise NotImplementedError("Subclasses must implement _reset_model")

class LSTMTrainer(ModelTrainer):
    
    def __init__(self, *args, **kwargs):
        super( LSTMTrainer, self ).__init__( *args, **kwargs )
        self.is_lstm = True
    
    def train_step(self, batch):
        
        """
        Perform a single training step for LSTM model.
        
        Args:
            batch: Batch of data (inputs, labels)
            
        Returns:
            Loss value
        """
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Get data
        inputs, labels = batch
        inputs, labels = inputs.to( self.device ), labels.to( self.device )
        
        # Forward pass
        outputs = self.model( inputs )
        loss = self.criterion( outputs, labels )
        
        # Backward pass and optimization
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_( self.model.parameters(), max_norm = 1.0 )
        
        self.optimizer.step()
        
        return loss.item()
    
    def _reset_model(self, params):
        
        """
        Reset LSTM model with new hyperparameters.
        
        Args:
            params: New hyperparameters
        """
        # Update model parameters if needed
        if hasattr(self.model, 'hidden_size') and 'hidden_size' in params:
            hidden_size = params['hidden_size']
        
        if hasattr(self.model, 'embedding') and 'embedding_dim' in params:
            embedding_dim = params['embedding_dim']
        
        if hasattr(self.model, 'bidirectional') and 'bidirectional' in params:
            bidirectional = params['bidirectional']
        
        if hasattr(self.model, 'dropout') and 'dropout' in params:
            dropout = params['dropout']
        
        if hasattr(self.model, 'use_attention') and 'attention' in params:
            use_attention = params['attention']
        
        # Update trainer parameters
        if 'optimizer' in params:
            self.optimizer_name = params['optimizer']
        
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        
        if 'weight_decay' in params:
            self.weight_decay = params['weight_decay']

        vocab_size = params.get( 'vocab_size', self.model.vocab_size )
        num_classes = params.get( 'num_classes', self.model.num_classes )
        max_length = params.get( 'max_length', self.model.max_length )
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        self.model = LSTMSentimentModel(
            vocab_size = vocab_size,
            embedding_dim = embedding_dim,
            hidden_size = hidden_size,
            num_classes = num_classes,
            bidirectional = bidirectional,
            dropout = dropout,
            use_attention = use_attention,
            max_length = max_length,
            padding_idx = 0,
            num_layers = params.get( 'num_layers', 1 )
        )
        self.model.to( self.device )
        
        # Re-initialize optimizer
        self.optimizer = self._get_optimizer()

class DistilBERTTrainer(ModelTrainer):
    
    def __init__(self, *args, **kwargs):
        super(DistilBERTTrainer, self).__init__(*args, **kwargs)
        self.is_lstm = False
    
    def train_step(self, batch):
        """
        Perform a single training step for DistilBERT model.
        
        Args:
            batch: Batch of data (dict with input_ids, attention_mask, labels)
            
        Returns:
            Loss value
        """
        # Set model to training mode
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Get data
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Forward pass
        outputs = self.model( input_ids = input_ids, attention_mask = attention_mask )
        
        if hasattr(outputs, 'loss'):
            # Use the loss calculated by the model
            loss = outputs.loss
        else:
            # Calculate loss manually
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
        
        # Backward pass and optimization
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_( self.model.parameters(), max_norm = 1.0 )
        
        self.optimizer.step()
        
        return loss.item()
    
    def _reset_model(self, params):
        """
        Reset DistilBERT model with new hyperparameters.
        
        Args:
            params: New hyperparameters
        """
        # Update model parameters if needed
        if hasattr(self.model, 'dropout') and 'dropout' in params:
            if hasattr(self.model.distilbert, 'config'):
                self.model.distilbert.config.dropout = params['dropout']
        
        # Update trainer parameters
        if 'optimizer' in params:
            self.optimizer_name = params['optimizer']
        
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        
        if 'weight_decay' in params:
            self.weight_decay = params['weight_decay']
        
        pretrained_model = params.get( 'pretrained_model', self.model.pretrained_model )
        num_classes = params.get( 'num_classes', self.model.num_classes )

        # Re-initialize model
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = DistilBertForSequenceClassification(
            num_classes = num_classes,
            dropout = params.get( 'dropout', 0.1 ),
            pretrained_model = pretrained_model
        )

        # Re-initialize optimizer
        self.optimizer = self._get_optimizer()