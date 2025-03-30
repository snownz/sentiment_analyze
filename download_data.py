import argparse
import os
import requests
import logging
import pandas as pd
from src.utils import create_synthetic_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_yelp_dataset(output_dir='data'):
    """
    Download the Yelp dataset from Hugging Face or create synthetic data.
    
    Args:
        output_dir: Directory to save the dataset
        use_huggingface: Whether to try downloading from Hugging Face
        n_samples: Number of samples for synthetic dataset
        
    Returns:
        Path to the dataset file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "yelp_reviews.json")
    
    # If the file already exists, return it
    if os.path.exists(output_file):
        logger.info(f"Dataset already exists at {output_file}")
        return output_file
    
    # If using Hugging Face, try to download it
    try:
        logger.info("Attempting to download Yelp dataset from Hugging Face datasets library...")
        # Try to use datasets library to download Yelp
        try:
            import datasets
            
            # Load the dataset - this will download it if not already cached
            logger.info("Loading Yelp dataset from Hugging Face...")
            dataset = datasets.load_dataset("Yelp/yelp_review_full", split="train")
            
            # Convert to pandas DataFrame
            logger.info("Converting to DataFrame...")
            df = pd.DataFrame(dataset)
            
            # The Yelp dataset has 'text' and 'label' columns (labels 0-4)
            # Map labels to sentiment categories and stars (1-5)
            logger.info("Mapping labels to sentiment categories...")
            
            # Hugging Face Yelp dataset has labels 0-4 (corresponding to stars 1-5)
            def map_stars_to_sentiment(stars):
                if stars <= 2:
                    return "negative"
                elif stars == 3:
                    return "neutral"
                else:
                    return "positive"
            
            df['stars'] = df['label'] + 1  # Convert 0-4 to 1-5
            df['sentiment'] = df['stars'].apply(map_stars_to_sentiment)
            
            # Save to JSON
            logger.info(f"Saving dataset to {output_file}...")
            df.to_json(output_file, orient='records', lines=True)
            
            logger.info(f"Dataset downloaded and saved to {output_file}")
            
            # Print dataset statistics
            sentiment_counts = df['sentiment'].value_counts()
            logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
            
            return output_file
            
        except ImportError:
            logger.warning("Hugging Face datasets library not found. Please install with: pip install datasets")
            logger.info("Falling back to synthetic dataset generation")
    except Exception as e:
        logger.warning(f"Could not download dataset from Hugging Face: {e}")
        logger.info("Falling back to synthetic dataset generation")
    
    # If we get here, either use_huggingface=False or download failed
    n_samples = len(df)
    logger.info(f"Creating synthetic Yelp review dataset with {n_samples} samples...")
    create_synthetic_dataset(output_file, n_samples=n_samples)
    logger.info(f"Synthetic dataset created at {output_file}")
    
    return output_file

def main():

    parser = argparse.ArgumentParser(description='Download or Generate Yelp Dataset')
    parser.add_argument('--output_dir', type=str, default='data', 
                        help='Directory to save the dataset')
    parser.add_argument('--filename', type=str, default='yelp_reviews.json', 
                        help='Filename for the dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Force synthetic data generation (ignores --huggingface)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Full path to the output file
    output_file = os.path.join(args.output_dir, args.filename)
    
    if args.synthetic:
        logger.info(f"Generating synthetic dataset with {args.samples} samples...")
        create_synthetic_dataset(output_file, n_samples=args.samples)
    else:
        logger.info("Attempting to download Yelp data from Hugging Face...")
        try:
            # Try to download from Hugging Face
            download_yelp_dataset( args.output_dir )
        except Exception as e:
            logger.error(f"Error downloading data from Hugging Face: {e}")
            logger.info("Falling back to synthetic data generation.")
    
    # Check the contents of the dataset
    try:
        df = pd.read_json(output_file, lines=True)
        logger.info(f"Dataset ready at: {output_file} with {len(df)} reviews")
        
        # Print statistics
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
    
    logger.info(f"To use this dataset for training, run:")
    logger.info(f"python train_lstm.py --data_path {output_file}")
    logger.info(f"python train_distilbert.py --data_path {output_file}")

if __name__ == '__main__':
    main()