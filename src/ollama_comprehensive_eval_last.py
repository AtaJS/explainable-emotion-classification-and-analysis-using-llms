"""
Comprehensive Ollama Model Evaluation Script
Evaluates emotion classification performance across different configurations
Saves metrics and predictions for visualization
"""

from datasets import load_dataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                             matthews_corrcoef, cohen_kappa_score, classification_report)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import ollama
from datetime import datetime
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Create results directory
os.makedirs("results", exist_ok=True)

# Configuration
models_config = {
    "mistral:7b": {"color": "Blues"},
    "qwen2.5:7b": {"color": "Greens"}
}

shot_configs = [0, 3, 5]
tweet_length_thresholds = [100, 200, 500]

SAMPLE_SIZE = 500
MAX_WORKERS = 4

print("="*70)
print("OPTIMIZED EVALUATION WITH PREDICTIONS SAVED")
print("="*70)
print(f"Models: {list(models_config.keys())}")
print(f"Shot configurations: {shot_configs}")
print(f"Tweet length thresholds: {tweet_length_thresholds}")
print(f"Sample size: {SAMPLE_SIZE} (stratified)")
print(f"\nWill save: Metrics CSV + All Predictions (for confusion matrices)")
print(f"Run once, visualize unlimited times")
print("="*70 + "\n")

# Load and sample dataset
dataset = load_dataset("dair-ai/emotion", "split")
train_data = dataset["train"]
test_data_full = dataset["test"]

label_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
emotion_labels = list(label_map.values())

# Stratified sampling to maintain class distribution
print("Creating stratified sample...")
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
def classify_single(args):
    """Classify a single text using the specified model and few-shot examples"""
    text, model_name, examples, k_shot = args
    label_str = ", ".join(emotion_labels)
    prompt = ""
    
    # Add few-shot examples to prompt if applicable
    if k_shot > 0:
        for ex in examples[:k_shot]:
            prompt += f"""Classify the predominant emotion in the following text: "{ex["text"]}" as one of the emotions in: {label_str}. Answer with only the emotion word.
{ex["emotion"]}

"""
    
    # Add the main classification query
    prompt += f"""Classify the predominant emotion in the following text: "{text}" as one of the emotions in: {label_str}. Answer with only the emotion word."""
    
    try:
        response = ollama.generate(
            model=model_name, prompt=prompt,
            options={'temperature': 0.0, 'seed': 42, 'top_p': 1.0, 'top_k': 1, 
                    'repeat_penalty': 1.0, 'num_predict': 10}
        )
        output = response['response'].strip().lower()
        
        # Match output to emotion labels
        for emotion in emotion_labels:
            if emotion in output:
                return emotion
        return output
    except:
        return "error"

def evaluate_configuration(model_name, k_shot, examples):
    """Evaluate a specific model configuration using parallel processing"""
    tasks = [(text, model_name, examples, k_shot) for text in texts]
    preds = [None] * len(tasks)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_single, task): i for i, task in enumerate(tasks)}
        for future in tqdm(as_completed(futures), total=len(tasks), 
                          desc=f"  {model_name} {k_shot}-shot", ncols=80, leave=False):
            idx = futures[future]
            preds[idx] = future.result()
    return preds

# Run evaluation and save results
all_results = []
all_predictions = {}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("="*70)
print("RUNNING EVALUATION")
print("="*70 + "\n")

# Zero-shot evaluation (run once per model, reuse for all thresholds)
zero_shot_cache = {}

for model_name in models_config.keys():
    print(f"Running {model_name} - ZERO-SHOT")
    preds = evaluate_configuration(model_name, 0, [])
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
        
        # Store predictions with unique key
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
    
    for model_name in models_config.keys():
        for k_shot in [k for k in shot_configs if k > 0]:
            shot_name = f"{k_shot}-shot"
            print(f"Running {model_name} - {shot_name} - threshold={threshold}")
            
            preds = evaluate_configuration(model_name, k_shot, examples)
            
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
csv_filename = f"results/evaluation_{timestamp}.csv"
results_df.to_csv(csv_filename, index=False)
print(f"Saved metrics CSV: {csv_filename}")

# Save predictions as pickle
predictions_filename = f"results/predictions_{timestamp}.pkl"
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
    'predictions_file': predictions_filename
}
metadata_filename = f"results/metadata_{timestamp}.json"
import json
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
