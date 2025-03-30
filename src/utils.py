import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import json

def create_synthetic_dataset(output_file, n_samples=5000):
    """
    Create a synthetic Yelp review dataset for demonstration purposes.
    
    Args:
        output_file: Path to save the synthetic dataset
        n_samples: Number of synthetic reviews to generate
    """
    # Sample positive reviews
    positive_reviews = [
        "The food was amazing! Great service and atmosphere.",
        "Absolutely loved this place. Will definitely come back.",
        "Best restaurant in town. The chef is a genius.",
        "Excellent service, delicious food, and reasonable prices.",
        "The staff was friendly and attentive. Highly recommend!",
        "Great ambiance and even better food. A must-visit!",
        "The dishes were creative and full of flavor. Fantastic experience!",
        "Friendly staff, quick service, and delicious meals. Perfect!",
        "This place exceeded my expectations. Can't wait to return!",
        "Incredible dining experience from start to finish."
    ]
    
    # Sample neutral reviews
    neutral_reviews = [
        "The food was okay, nothing special but not bad either.",
        "Average experience. Might come back, might not.",
        "The service was decent, but the food was just alright.",
        "Not bad for the price, but I've had better.",
        "Some dishes were good, others were mediocre.",
        "It was a standard dining experience, nothing memorable.",
        "The restaurant was clean but the menu was limited.",
        "Food came quickly but lacked flavor.",
        "Decent place for a quick meal, but wouldn't go out of my way.",
        "The atmosphere was nice, but the food was average."
    ]
    
    # Sample negative reviews
    negative_reviews = [
        "Terrible service and the food was cold.",
        "Would not recommend. Overpriced and underwhelming.",
        "The worst dining experience I've ever had.",
        "The staff was rude and the food was inedible.",
        "Avoid this place at all costs. Complete waste of money.",
        "Extremely disappointed with both food and service.",
        "The restaurant was dirty and the food tasted old.",
        "Waited over an hour for mediocre food. Never again.",
        "Overpriced, underwhelming, and poor customer service.",
        "The food made me sick. Definitely not returning."
    ]
    
    # Create review templates with placeholders
    positive_templates = [
        "I {loved} this {place}! The {food} was {amazing} and the {service} was {excellent}.",
        "This {restaurant} is {fantastic}. The {menu} has {great} options and the {prices} are {reasonable}.",
        "Had a {wonderful} experience at this {place}. The {atmosphere} is {perfect} and {staff} is {friendly}.",
        "The {desserts} here are to {die for}. {Definitely} {recommend} this {spot}.",
        "{Best} {meal} I've had in a {long time}. The {chef} really knows what they're {doing}."
    ]
    
    neutral_templates = [
        "This {place} was {okay}. The {food} was {decent} but the {service} was {slow}.",
        "{Average} {restaurant}. The {prices} are {fair} but the {menu} is {limited}.",
        "The {food} here is {hit or miss}. Some {dishes} are {good}, others {not so much}.",
        "{Decent} {spot} for a {quick bite}, but {nothing special}.",
        "The {atmosphere} is {nice} but the {food} is just {alright}."
    ]
    
    negative_templates = [
        "I {hated} this {place}. The {food} was {terrible} and the {service} was {awful}.",
        "This {restaurant} is {overpriced}. The {menu} has {few} options and the {quality} is {poor}.",
        "Had a {horrible} experience at this {place}. The {atmosphere} is {uninviting} and {staff} is {rude}.",
        "The {food} here is {inedible}. {Definitely} {avoid} this {spot}.",
        "{Worst} {meal} I've had in a {long time}. The {management} really needs to {improve}."
    ]
    
    # Words to fill in templates
    positive_words = {
        "loved": ["loved", "enjoyed", "adored", "appreciated", "relished"],
        "place": ["place", "restaurant", "establishment", "spot", "venue"],
        "food": ["food", "cuisine", "dishes", "menu items", "entrees"],
        "amazing": ["amazing", "excellent", "outstanding", "exceptional", "superb"],
        "service": ["service", "staff", "waitstaff", "customer service", "attention"],
        "excellent": ["excellent", "top-notch", "first-rate", "stellar", "impeccable"],
        "restaurant": ["restaurant", "eatery", "dining spot", "bistro", "café"],
        "fantastic": ["fantastic", "wonderful", "marvelous", "terrific", "splendid"],
        "menu": ["menu", "selection", "offerings", "options", "choices"],
        "great": ["great", "excellent", "impressive", "extensive", "diverse"],
        "prices": ["prices", "rates", "costs", "pricing", "value"],
        "reasonable": ["reasonable", "fair", "affordable", "decent", "good value"],
        "wonderful": ["wonderful", "delightful", "pleasant", "enjoyable", "satisfying"],
        "atmosphere": ["atmosphere", "ambiance", "vibe", "environment", "setting"],
        "perfect": ["perfect", "ideal", "charming", "cozy", "welcoming"],
        "staff": ["staff", "servers", "waiters", "employees", "team"],
        "friendly": ["friendly", "courteous", "attentive", "helpful", "accommodating"],
        "desserts": ["desserts", "sweets", "pastries", "treats", "confections"],
        "die for": ["die for", "rave about", "savor", "enjoy thoroughly", "treasure"],
        "definitely": ["definitely", "absolutely", "certainly", "without doubt", "surely"],
        "recommend": ["recommend", "suggest", "endorse", "advocate", "promote"],
        "spot": ["spot", "location", "venue", "establishment", "destination"],
        "best": ["best", "finest", "greatest", "most delicious", "most outstanding"],
        "meal": ["meal", "dinner", "lunch", "breakfast", "dining experience"],
        "long time": ["long time", "while", "ages", "years", "months"],
        "chef": ["chef", "cook", "kitchen staff", "culinary team", "head chef"],
        "doing": ["doing", "creating", "preparing", "cooking", "crafting"]
    }
    
    neutral_words = {
        "okay": ["okay", "alright", "acceptable", "passable", "satisfactory"],
        "food": ["food", "cuisine", "dishes", "menu items", "entrees"],
        "decent": ["decent", "adequate", "reasonable", "fair", "tolerable"],
        "service": ["service", "staff", "waitstaff", "customer service", "attention"],
        "slow": ["slow", "unhurried", "leisurely", "not prompt", "delayed"],
        "average": ["average", "standard", "typical", "ordinary", "mediocre"],
        "restaurant": ["restaurant", "eatery", "dining spot", "establishment", "place"],
        "prices": ["prices", "rates", "costs", "pricing", "charges"],
        "fair": ["fair", "reasonable", "moderate", "acceptable", "appropriate"],
        "menu": ["menu", "selection", "offerings", "options", "choices"],
        "limited": ["limited", "restricted", "small", "narrow", "sparse"],
        "hit or miss": ["hit or miss", "inconsistent", "variable", "unpredictable", "unreliable"],
        "dishes": ["dishes", "items", "options", "selections", "choices"],
        "good": ["good", "decent", "adequate", "satisfactory", "acceptable"],
        "not so much": ["not so much", "disappointing", "underwhelming", "lacking", "subpar"],
        "decent": ["decent", "adequate", "acceptable", "reasonable", "fair"],
        "spot": ["spot", "place", "location", "establishment", "venue"],
        "quick bite": ["quick bite", "fast meal", "casual meal", "quick service", "fast food"],
        "nothing special": ["nothing special", "unremarkable", "unmemorable", "ordinary", "standard"],
        "atmosphere": ["atmosphere", "ambiance", "environment", "setting", "vibe"],
        "nice": ["nice", "pleasant", "agreeable", "decent", "satisfactory"],
        "alright": ["alright", "okay", "acceptable", "passable", "so-so"],
        "place": ["place", "restaurant", "establishment", "eatery", "venue"]
    }
    
    negative_words = {
        "hated": ["hated", "disliked", "despised", "detested", "loathed"],
        "place": ["place", "restaurant", "establishment", "spot", "venue"],
        "food": ["food", "cuisine", "dishes", "menu items", "entrees"],
        "terrible": ["terrible", "awful", "dreadful", "horrible", "atrocious"],
        "service": ["service", "staff", "waitstaff", "customer service", "attention"],
        "awful": ["awful", "poor", "bad", "subpar", "unacceptable"],
        "restaurant": ["restaurant", "eatery", "dining spot", "establishment", "place"],
        "overpriced": ["overpriced", "expensive", "costly", "pricey", "extortionate"],
        "menu": ["menu", "selection", "offerings", "options", "choices"],
        "few": ["few", "limited", "scarce", "insufficient", "minimal"],
        "quality": ["quality", "standard", "caliber", "grade", "condition"],
        "poor": ["poor", "low", "inferior", "substandard", "inadequate"],
        "horrible": ["horrible", "terrible", "dreadful", "awful", "appalling"],
        "atmosphere": ["atmosphere", "ambiance", "environment", "setting", "vibe"],
        "uninviting": ["uninviting", "unwelcoming", "cold", "sterile", "unpleasant"],
        "staff": ["staff", "servers", "waiters", "employees", "personnel"],
        "rude": ["rude", "impolite", "discourteous", "disrespectful", "unfriendly"],
        "inedible": ["inedible", "unpalatable", "disgusting", "revolting", "vile"],
        "definitely": ["definitely", "absolutely", "certainly", "surely", "without doubt"],
        "avoid": ["avoid", "skip", "pass on", "stay away from", "steer clear of"],
        "spot": ["spot", "location", "venue", "establishment", "place"],
        "worst": ["worst", "most terrible", "most awful", "most horrific", "most dreadful"],
        "meal": ["meal", "dinner", "lunch", "breakfast", "dining experience"],
        "long time": ["long time", "while", "ages", "years", "months"],
        "management": ["management", "owner", "staff", "chef", "team"],
        "improve": ["improve", "enhance", "upgrade", "fix", "address"]
    }
    
    # Function to fill a template with random words
    def fill_template(template, word_dict):
        filled = template
        for placeholder in word_dict.keys():
            if "{" + placeholder + "}" in filled:
                filled = filled.replace("{" + placeholder + "}", np.random.choice(word_dict[placeholder]))
        return filled
    
    # Generate synthetic reviews
    reviews = []
    business_ids = [f"business_{i}" for i in range(100)]  # 100 fictional businesses
    user_ids = [f"user_{i}" for i in range(500)]  # 500 fictional users
    
    for _ in range(n_samples):
        # Randomly select sentiment
        sentiment = np.random.choice(["positive", "neutral", "negative"], p=[0.5, 0.3, 0.2])
        
        if sentiment == "positive":
            # 80% generated from templates, 20% from samples
            if np.random.random() < 0.8:
                template = np.random.choice(positive_templates)
                text = fill_template(template, positive_words)
            else:
                text = np.random.choice(positive_reviews)
            stars = np.random.choice([4, 5], p=[0.3, 0.7])
        
        elif sentiment == "neutral":
            if np.random.random() < 0.8:
                template = np.random.choice(neutral_templates)
                text = fill_template(template, neutral_words)
            else:
                text = np.random.choice(neutral_reviews)
            stars = 3
        
        else:  # negative
            if np.random.random() < 0.8:
                template = np.random.choice(negative_templates)
                text = fill_template(template, negative_words)
            else:
                text = np.random.choice(negative_reviews)
            stars = np.random.choice([1, 2], p=[0.7, 0.3])
        
        # Create review entry
        review = {
            "review_id": f"review_{len(reviews)}",
            "user_id": np.random.choice(user_ids),
            "business_id": np.random.choice(business_ids),
            "stars": float(stars),
            "useful": np.random.randint(0, 10),
            "funny": np.random.randint(0, 5),
            "cool": np.random.randint(0, 5),
            "text": text,
            "date": f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
        }
        
        reviews.append(review)
    
    # Convert to DataFrame
    df = pd.DataFrame(reviews)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON file (line by line format)
    df.to_json(output_file, orient='records', lines=True)
    
    print(f"Created synthetic dataset with {len(df)} reviews at {output_file}")

def compare_models(lstm_results, distilbert_results, df, class_names=None):
    """
    Compare LSTM and DistilBERT model performance.
    
    Args:
        lstm_results: Results from LSTM evaluation
        distilbert_results: Results from DistilBERT evaluation
        df: DataFrame with processed texts
        class_names: Names of the classes
    """
    if class_names is None:
        class_names = ['negative', 'neutral', 'positive']
    
    # Create directory for comparisons
    os.makedirs('results/comparison', exist_ok=True)
    
    # Compare accuracy
    accuracies = {
        'LSTM': lstm_results['accuracy'],
        'DistilBERT': distilbert_results['accuracy']
    }
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('results/comparison/accuracy.png')
    plt.close()
    
    # Compare performance by class
    lstm_report = lstm_results['classification_report']
    distilbert_report = distilbert_results['classification_report']
    
    metrics = ['precision', 'recall', 'f1-score']
    
    for metric in metrics:
        metric_data = {
            'Class': [],
            'LSTM': [],
            'DistilBERT': []
        }
        
        for cls in class_names:
            metric_data['Class'].append(cls)
            metric_data['LSTM'].append(lstm_report[cls][metric])
            metric_data['DistilBERT'].append(distilbert_report[cls][metric])
        
        metric_df = pd.DataFrame(metric_data)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y='value', hue='Model', 
                   data=pd.melt(metric_df, id_vars=['Class'], 
                                value_vars=['LSTM', 'DistilBERT'], 
                                var_name='Model', value_name='value'))
        plt.title(f'{metric.capitalize()} by Class')
        plt.xlabel('Class')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f'results/comparison/{metric}.png')
        plt.close()
    
    # Compare confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(lstm_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names, ax=axs[0])
    axs[0].set_title('LSTM Confusion Matrix')
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('True')
    
    sns.heatmap(distilbert_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names, ax=axs[1])
    axs[1].set_title('DistilBERT Confusion Matrix')
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('results/comparison/confusion_matrices.png')
    plt.close()
    
    # Save comparison results
    comparison = {
        'accuracy': {
            'LSTM': lstm_results['accuracy'],
            'DistilBERT': distilbert_results['accuracy']
        },
        'classification_report': {
            'LSTM': lstm_report,
            'DistilBERT': distilbert_report
        }
    }
    
    with open('results/comparison/results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print results
    print("\nModel Comparison:")
    print(f"LSTM Accuracy: {lstm_results['accuracy']:.4f}")
    print(f"DistilBERT Accuracy: {distilbert_results['accuracy']:.4f}")
    
    print("\nLSTM Classification Report:")
    for cls in class_names:
        print(f"{cls}: Precision={lstm_report[cls]['precision']:.4f}, Recall={lstm_report[cls]['recall']:.4f}, F1={lstm_report[cls]['f1-score']:.4f}")
    
    print("\nDistilBERT Classification Report:")
    for cls in class_names:
        print(f"{cls}: Precision={distilbert_report[cls]['precision']:.4f}, Recall={distilbert_report[cls]['recall']:.4f}, F1={distilbert_report[cls]['f1-score']:.4f}")
    
    return comparison

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False