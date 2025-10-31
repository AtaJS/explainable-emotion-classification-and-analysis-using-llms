"""
HuggingFace FLAN-T5 Model Evaluation Script
Evaluates flan-t5-small and flan-t5-base models for emotion classification
Saves metrics and predictions for visualization
"""

from datasets import load_dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                             matthews_corrcoef, cohen_kappa_score, classification_report)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, set_seed
from huggingface_hub import login
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from datetime import datetime
import os
import pickle
import json
import torch
from tqdm import tqdm

# Set seeds for reproducibility
set_seed(42)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create results directory
os.makedirs("results_hf", exist_ok=True)

print("="*70)
print("HUGGINGFACE FLAN-T5 EVALUATION WITH PREDICTIONS SAVED")
print("="*70)

# HuggingFace authentication
hf_token = "Your_HuggingFace_API_Token_Here"  # Replace with your actual token
login(token=hf_token)
print("Logged into HuggingFace\n")

# Configuration
models_config = {
    "google/flan-t5-small": {"local_path": "./flan-t5-small"},
    "google/flan-t5-base": {"local_path": "./flan-t5-base"}
}

shot_configs = [0, 3, 5]
tweet_length_thresholds = [100, 200, 500]
SAMPLE_SIZE = 500

print(f"Models: {list(models_config.keys())}")
print(f"Shot configurations: {shot_configs}")
print(f"Tweet length thresholds: {tweet_length_thresholds}")
print(f"Sample size: {SAMPLE_SIZE} (stratified)")
print("\nWill save: Metrics CSV + All Predictions")
print("="*70 + "\n")

# Load models
print("Loading models from HuggingFace...")
pipelines_dict = {}

for model_name, config in models_config.items():
    local_path = config['local_path']
    print(f"Loading {model_name}...")
    
    # Download model if not exists locally
    if not os.path.exists(local_path):
        print(f"Downloading to {local_path}...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=local_path)
    
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_path)
    
    # Create pipeline with deterministic settings
    pipe = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=10,
        do_sample=False,
        num_beams=1
    )
    pipelines_dict[model_name] = pipe
    print(f"Loaded successfully")

print()

# Load dataset
print("Loading dataset...")
dataset = load_dataset("dair-ai/emotion", "split")
train_data = dataset["train"]
test_data_full = dataset["test"]

label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
emotion_labels = list(label_map.values())

# Stratified sampling to maintain class distribution
labels_full = [label_map[l] for l in test_data_full['label']]
indices = list(range(len(test_data_full)))
_, sample_indices = train_test_split(
    indices, test_size=SAMPLE_SIZE, stratify=labels_full, random_state=42
)

test_data = test_data_full.select(sample_indices)
texts = test_data["text"]
true_labels = [label_map[l] for l in test_data['label']]

print(f"Sampled: {len(test_data)} samples (stratified)")
print(f"Class distribution maintained\n")

# Pre-select balanced examples for few-shot learning
def select_balanced_examples(max_length, max_shots=5):
    """Select balanced examples covering all emotion classes within length constraint"""
    examples = []
    emotions_seen = set()
    train_indices = list(range(len(train_data)))
    random.seed(42)
    random.shuffle(train_indices)
    
    for i in train_indices:
        emotion = label_map[train_data[i]['label']]
        tweet_text = train_data[i]['text']
        if len(tweet_text) <= max_length and emotion not in emotions_seen:
            examples.append({"text": tweet_text, "emotion": emotion})
            emotions_seen.add(emotion)
            if len(examples) == min(max_shots, len(emotion_labels)):
                break
    return examples

print("Pre-selecting few-shot examples...")
few_shot_examples_cache = {}
for threshold in tweet_length_thresholds:
    examples = select_balanced_examples(threshold, max_shots=5)
    few_shot_examples_cache[threshold] = examples
    emotions = set(ex['emotion'] for ex in examples)
    print(f"Threshold {threshold}: {len(examples)} examples, {len(emotions)} emotions")
print()

# Classification functions
def classify_batch(texts_batch, model_pipeline, examples, k_shot):
    """Classify a batch of texts using the specified model and few-shot examples"""
    label_str = ", ".join(emotion_labels)
    prompts = []
    
    for text in texts_batch:
        prompt = ""
        
        # Add few-shot examples to prompt if applicable
        if k_shot > 0:
            for ex in examples[:k_shot]:
                prompt += f"""Classify the predominant emotion in the following text: "{ex["text"]}" as one of the emotions in: {label_str}. Answer with only the emotion word.
{ex["emotion"]}

"""
        
        # Add the main classification query
        prompt += f"""Classify the predominant emotion in the following text: "{text}" as one of the emotions in: {label_str}. Answer with only the emotion word."""
        prompts.append(prompt)
    
    # Generate predictions using the pipeline
    outputs = model_pipeline(prompts, batch_size=8)
    
    # Extract emotion words from model outputs
    predictions = []
    for output in outputs:
        pred_text = output['generated_text'].strip().lower()
        matched = False
        for emotion in emotion_labels:
            if emotion in pred_text:
                predictions.append(emotion)
                matched = True
                break
        if not matched:
            predictions.append(pred_text)
    
    return predictions

def evaluate_configuration(model_name, model_pipeline, k_shot, examples):
    """Evaluate a specific model configuration using batch processing"""
    batch_size = 16
    all_preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), 
                  desc=f"  {model_name.split('/')[-1]} {k_shot}-shot", 
                  ncols=80, leave=False):
        batch_texts = texts[i:i+batch_size]
        batch_preds = classify_batch(batch_texts, model_pipeline, examples, k_shot)
        all_preds.extend(batch_preds)
    
    return all_preds

# Run evaluation and save results
all_results = []
all_predictions = {}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("="*70)
print("RUNNING EVALUATION")
print("="*70 + "\n")

# Zero-shot evaluation (run once per model, reuse for all thresholds)
zero_shot_cache = {}

for model_name, model_pipeline in pipelines_dict.items():
    print(f"Running {model_name} - ZERO-SHOT")
    
    preds = evaluate_configuration(model_name, model_pipeline, 0, [])
    zero_shot_cache[model_name] = preds
    
    # Compute metrics
    acc = accuracy_score(true_labels, preds)
    bal_acc = balanced_accuracy_score(true_labels, preds)
    f1_macro = f1_score(true_labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(true_labels, preds, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(true_labels, preds)
    kappa = cohen_kappa_score(true_labels, preds)
    report = classification_report(true_labels, preds, labels=emotion_labels, 
                                   output_dict=True, zero_division=0)
    
    # Save results for all thresholds
    for threshold in tweet_length_thresholds:
        result = {
            'model': model_name,
            'shot_config': 'zero-shot',
            'k_shot': 0,
            'tweet_length_threshold': threshold,
            'num_test_samples': len(texts),
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'mcc': mcc,
            'cohen_kappa': kappa,
        }
        
        # Add per-emotion F1 scores
        for emotion in emotion_labels:
            if emotion in report:
                result[f'f1_{emotion}'] = report[emotion]['f1-score']
        all_results.append(result)
        
        # Store predictions
        pred_key = f"{model_name}_zero-shot_threshold{threshold}"
        all_predictions[pred_key] = {
            'predictions': preds.copy(),
            'true_labels': true_labels.copy(),
            'config': result.copy()
        }
    
    print(f"Completed: Acc={acc:.4f}, F1={f1_weighted:.4f}\n")

# Few-shot evaluation for each threshold
for threshold in tweet_length_thresholds:
    examples = few_shot_examples_cache[threshold]
    print(f"{'='*70}")
    print(f"THRESHOLD: {threshold} chars")
    print(f"{'='*70}\n")
    
    for model_name, model_pipeline in pipelines_dict.items():
        for k_shot in [k for k in shot_configs if k > 0]:
            shot_name = f"{k_shot}-shot"
            print(f"Running {model_name} - {shot_name} - threshold={threshold}")
            
            preds = evaluate_configuration(model_name, model_pipeline, k_shot, examples)
            
            # Compute metrics
            acc = accuracy_score(true_labels, preds)
            bal_acc = balanced_accuracy_score(true_labels, preds)
            f1_macro = f1_score(true_labels, preds, average='macro', zero_division=0)
            f1_weighted = f1_score(true_labels, preds, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(true_labels, preds)
            kappa = cohen_kappa_score(true_labels, preds)
            report = classification_report(true_labels, preds, labels=emotion_labels, 
                                          output_dict=True, zero_division=0)
            
            result = {
                'model': model_name,
                'shot_config': shot_name,
                'k_shot': k_shot,
                'tweet_length_threshold': threshold,
                'num_test_samples': len(texts),
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'mcc': mcc,
                'cohen_kappa': kappa,
            }
            
            # Add per-emotion F1 scores
            for emotion in emotion_labels:
                if emotion in report:
                    result[f'f1_{emotion}'] = report[emotion]['f1-score']
            all_results.append(result)
            
            # Store predictions
            pred_key = f"{model_name}_{shot_name}_threshold{threshold}"
            all_predictions[pred_key] = {
                'predictions': preds.copy(),
                'true_labels': true_labels.copy(),
                'config': result.copy()
            }
            
            print(f"Completed: Acc={acc:.4f}, F1={f1_weighted:.4f}\n")

# Save results
print(f"{'='*70}")
print("SAVING ALL RESULTS")
print(f"{'='*70}\n")

# Save metrics as CSV
results_df = pd.DataFrame(all_results)
csv_filename = f"results_hf/evaluation_hf_{timestamp}.csv"
results_df.to_csv(csv_filename, index=False)
print(f"Saved metrics CSV: {csv_filename}")

# Save predictions as pickle
predictions_filename = f"results_hf/predictions_hf_{timestamp}.pkl"
with open(predictions_filename, 'wb') as f:
    pickle.dump(all_predictions, f)
print(f"Saved predictions: {predictions_filename}")

# Save metadata
metadata = {
    'timestamp': timestamp,
    'sample_size': SAMPLE_SIZE,
    'models': list(models_config.keys()),
    'shot_configs': shot_configs,
    'thresholds': tweet_length_thresholds,
    'emotion_labels': emotion_labels,
    'csv_file': csv_filename,
    'predictions_file': predictions_filename,
    'framework': 'huggingface',
    'model_type': 'flan-t5'
}
metadata_filename = f"results_hf/metadata_hf_{timestamp}.json"
with open(metadata_filename, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata: {metadata_filename}")

print(f"\n{'='*70}")
print("EVALUATION COMPLETED")
print(f"{'='*70}")
print(f"\nFiles saved:")
print(f"   1. {csv_filename}")
print(f"   2. {predictions_filename}")
print(f"   3. {metadata_filename}")
print(f"\nTotal configurations: {len(all_results)}")
print(f"Total prediction sets: {len(all_predictions)}")
