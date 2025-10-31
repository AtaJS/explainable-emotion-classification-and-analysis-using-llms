"""
Comprehensive Visualization Script
Generates all plots including confusion matrices from saved predictions
Requires no API calls - operates entirely on saved data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle
import json
from datetime import datetime

print("="*70)
print("COMPREHENSIVE VISUALIZATION FROM SAVED DATA")
print("="*70 + "\n")

# Load saved data
print("Loading saved data...")

# Update these with your actual filenames
timestamp = "20251022_160743"
csv_file = f"results/evaluation_{timestamp}.csv"
predictions_file = f"results/predictions_{timestamp}.pkl"
metadata_file = f"results/metadata_{timestamp}.json"

try:
    results_df = pd.read_csv(csv_file)
    print(f"Loaded metrics: {csv_file}")
    
    with open(predictions_file, 'rb') as f:
        all_predictions = pickle.load(f)
    print(f"Loaded predictions: {predictions_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded metadata: {metadata_file}")
    
    emotion_labels = metadata['emotion_labels']
    print(f"\nReady: {len(all_predictions)} prediction sets loaded")
    print(f"Total configurations: {len(results_df)}")
    
except FileNotFoundError as e:
    print(f"\nError: Could not find file!")
    print(f"Update the timestamp variable at the top of this script")
    print(f"Current timestamp: {timestamp}")
    print(f"Your files are probably named with a different timestamp")
    exit(1)

viz_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print()

# Main comparison: Tweet length impact
print("="*70)
print("GENERATING VISUALIZATIONS")
print("="*70 + "\n")

print("[1/6] Main comparison plots...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Impact of Tweet Length Threshold on Model Performance', 
             fontsize=18, fontweight='bold', y=0.995)

metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'mcc', 'cohen_kappa', 'balanced_accuracy']
metric_labels = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'MCC', "Cohen's Kappa", 'Balanced Accuracy']

colors = {'mistral:7b': '#FF6B6B', 'qwen2.5:7b': '#4ECDC4'}
markers = {'zero-shot': 'o', '3-shot': 's', '5-shot': '^'}
linestyles = {'zero-shot': '-', '3-shot': '--', '5-shot': ':'}

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 3, idx % 3]
    
    for model in results_df['model'].unique():
        for shot in results_df['shot_config'].unique():
            data = results_df[(results_df['model'] == model) & 
                             (results_df['shot_config'] == shot)]
            if len(data) > 0:
                ax.plot(data['tweet_length_threshold'], data[metric],
                       marker=markers[shot], linestyle=linestyles[shot],
                       color=colors[model], linewidth=2.5, markersize=9,
                       label=f"{model.split(':')[0]} {shot}", alpha=0.85)
    
    ax.set_xlabel('Tweet Length Threshold (chars)', fontsize=11, fontweight='bold')
    ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f"viz_main_comparison_{viz_timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: viz_main_comparison_{viz_timestamp}.png")
plt.close()

# Confusion matrices - all configurations
print("[2/6] Confusion matrices (all configurations)...")

# Create 3x3 grid for each model (3 thresholds × 3 shots)
for model in ['mistral:7b', 'qwen2.5:7b']:
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f'{model} - Confusion Matrices Across All Configurations', 
                 fontsize=16, fontweight='bold')
    
    plot_idx = 0
    for threshold in sorted(results_df['tweet_length_threshold'].unique()):
        for shot_config in ['zero-shot', '3-shot', '5-shot']:
            ax = axes[plot_idx // 3, plot_idx % 3]
            
            # Get predictions
            pred_key = f"{model}_{shot_config}_threshold{threshold}"
            if pred_key in all_predictions:
                pred_data = all_predictions[pred_key]
                preds = pred_data['predictions']
                true_labels = pred_data['true_labels']
                config = pred_data['config']
                
                # Generate normalized confusion matrix
                cm = confusion_matrix(true_labels, preds, labels=emotion_labels)
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Plot heatmap
                sns.heatmap(cm_norm, annot=True, fmt='.2f', 
                           cmap='Reds' if model == 'mistral:7b' else 'Blues',
                           xticklabels=emotion_labels, yticklabels=emotion_labels,
                           ax=ax, vmin=0, vmax=1, cbar=False)
                
                ax.set_title(f'{shot_config}, T={threshold}\n'
                            f'Acc={config["accuracy"]:.3f}, F1={config["f1_weighted"]:.3f}',
                            fontsize=10, fontweight='bold')
                ax.set_xlabel('Predicted', fontsize=9)
                ax.set_ylabel('True', fontsize=9)
                ax.tick_params(labelsize=8)
            
            plot_idx += 1
    
    plt.tight_layout()
    model_safe = model.replace(':', '_')
    plt.savefig(f"confusion_matrices_{model_safe}_all_{viz_timestamp}.png", 
                dpi=300, bbox_inches='tight')
    print(f"Saved: confusion_matrices_{model_safe}_all_{viz_timestamp}.png")
    plt.close()

# Best configuration comparison
print("[3/6] Best configuration comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Best Configurations - Detailed Confusion Matrices', 
             fontsize=16, fontweight='bold')

for idx, model in enumerate(['mistral:7b', 'qwen2.5:7b']):
    model_data = results_df[results_df['model'] == model]
    best_row = model_data.loc[model_data['f1_weighted'].idxmax()]
    
    # Get best predictions
    pred_key = f"{model}_{best_row['shot_config']}_threshold{int(best_row['tweet_length_threshold'])}"
    pred_data = all_predictions[pred_key]
    
    cm = confusion_matrix(pred_data['true_labels'], pred_data['predictions'], 
                         labels=emotion_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', 
                cmap='Reds' if model == 'mistral:7b' else 'Blues',
                xticklabels=emotion_labels, yticklabels=emotion_labels,
                ax=axes[idx], vmin=0, vmax=1)
    
    axes[idx].set_title(f'{model}\n{best_row["shot_config"]}, T={int(best_row["tweet_length_threshold"])}\n'
                       f'Acc={best_row["accuracy"]:.3f}, F1={best_row["f1_weighted"]:.3f}',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=11)
    axes[idx].set_ylabel('True', fontsize=11)

plt.tight_layout()
plt.savefig(f"confusion_matrices_best_comparison_{viz_timestamp}.png", 
            dpi=300, bbox_inches='tight')
print(f"Saved: confusion_matrices_best_comparison_{viz_timestamp}.png")
plt.close()

# Per-emotion F1 scores
print("[4/6] Per-emotion F1 scores...")

emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
emotion_cols = [f'f1_{emotion}' for emotion in emotions]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Per-Emotion F1 Scores Across Configurations', 
             fontsize=16, fontweight='bold')

plot_idx = 0
for model in ['mistral:7b', 'qwen2.5:7b']:
    for shot in ['zero-shot', '3-shot', '5-shot']:
        ax = axes[plot_idx // 3, plot_idx % 3]
        
        model_shot_data = results_df[(results_df['model'] == model) & 
                                     (results_df['shot_config'] == shot)]
        
        if len(model_shot_data) > 0:
            best_row = model_shot_data.loc[model_shot_data['f1_weighted'].idxmax()]
            emotion_scores = [best_row[col] for col in emotion_cols]
            
            bars = ax.bar(emotions, emotion_scores, 
                         color=colors[model], alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Emotion', fontsize=11, fontweight='bold')
            ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
            ax.set_title(f'{model.split(":")[0]} | {shot}\n'
                        f'T={int(best_row["tweet_length_threshold"])}, '
                        f'F1={best_row["f1_weighted"]:.3f}',
                        fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        plot_idx += 1

plt.tight_layout()
plt.savefig(f"per_emotion_f1_{viz_timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: per_emotion_f1_{viz_timestamp}.png")
plt.close()

# Performance heatmaps
print("[5/6] Performance heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Accuracy Heatmaps Across All Configurations', 
             fontsize=16, fontweight='bold')

for idx, model in enumerate(['mistral:7b', 'qwen2.5:7b']):
    model_data = results_df[results_df['model'] == model]
    pivot = model_data.pivot_table(values='accuracy', 
                                   index='tweet_length_threshold',
                                   columns='shot_config', aggfunc='first')
    pivot = pivot[['zero-shot', '3-shot', '5-shot']]
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn',
                cbar_kws={'label': 'Accuracy'}, ax=axes[idx], 
                vmin=0.3, vmax=0.7, linewidths=2, linecolor='white')
    
    axes[idx].set_title(f'{model}', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Shot Configuration', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Tweet Length Threshold', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(f"heatmaps_{viz_timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: heatmaps_{viz_timestamp}.png")
plt.close()

# Improvement analysis
print("[6/6] Improvement analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Few-Shot Learning Improvement Over Zero-Shot', 
             fontsize=16, fontweight='bold')

metrics_imp = ['accuracy', 'f1_weighted', 'f1_macro', 'mcc']
labels_imp = ['Accuracy', 'F1 Weighted', 'F1 Macro', 'MCC']

for idx, (metric, label) in enumerate(zip(metrics_imp, labels_imp)):
    ax = axes[idx // 2, idx % 2]
    
    thresholds = sorted(results_df['tweet_length_threshold'].unique())
    x = np.arange(len(thresholds))
    width = 0.2
    
    for m_idx, model in enumerate(['mistral:7b', 'qwen2.5:7b']):
        improvements_3 = []
        improvements_5 = []
        
        for threshold in thresholds:
            # Get zero-shot baseline
            zero = results_df[(results_df['model'] == model) & 
                             (results_df['shot_config'] == 'zero-shot') &
                             (results_df['tweet_length_threshold'] == threshold)][metric].values[0]
            # Get 3-shot performance
            shot3 = results_df[(results_df['model'] == model) & 
                              (results_df['shot_config'] == '3-shot') &
                              (results_df['tweet_length_threshold'] == threshold)][metric].values[0]
            # Get 5-shot performance
            shot5 = results_df[(results_df['model'] == model) & 
                              (results_df['shot_config'] == '5-shot') &
                              (results_df['tweet_length_threshold'] == threshold)][metric].values[0]
            
            improvements_3.append(shot3 - zero)
            improvements_5.append(shot5 - zero)
        
        offset = m_idx * width * 2
        ax.bar(x + offset, improvements_3, width, 
               label=f'{model.split(":")[0]} 3-shot', 
               color=colors[model], alpha=0.7)
        ax.bar(x + offset + width, improvements_5, width,
               label=f'{model.split(":")[0]} 5-shot', 
               color=colors[model], alpha=0.4)
    
    ax.set_xlabel('Tweet Length Threshold', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{label} Improvement', fontsize=11, fontweight='bold')
    ax.set_title(f'{label} Improvement', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(thresholds)
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"improvement_analysis_{viz_timestamp}.png", dpi=300, bbox_inches='tight')
print(f"Saved: improvement_analysis_{viz_timestamp}.png")
plt.close()

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70 + "\n")

print("Best Configuration for Each Model:\n")
for model in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model]
    best_config = model_data.loc[model_data['f1_weighted'].idxmax()]
    
    print(f"{model}:")
    print(f"  Shot config: {best_config['shot_config']}")
    print(f"  Threshold: {int(best_config['tweet_length_threshold'])} chars")
    print(f"  Accuracy: {best_config['accuracy']:.4f}")
    print(f"  F1 Weighted: {best_config['f1_weighted']:.4f}")
    print(f"  F1 Macro: {best_config['f1_macro']:.4f}")
    print()

print("\n" + "="*70)
print("VISUALIZATION COMPLETED")
print("="*70)
print(f"\nvisualization files:")
print(f"  1. viz_main_comparison_{viz_timestamp}.png")
print(f"  2. confusion_matrices_mistral_7b_all_{viz_timestamp}.png")
print(f"  3. confusion_matrices_qwen2_5_7b_all_{viz_timestamp}.png")
print(f"  4. confusion_matrices_best_comparison_{viz_timestamp}.png")
print(f"  5. per_emotion_f1_{viz_timestamp}.png")
print(f"  6. heatmaps_{viz_timestamp}.png")
print(f"  7. improvement_analysis_{viz_timestamp}.png")
