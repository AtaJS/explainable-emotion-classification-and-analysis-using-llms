# Explainable Emotion Classification Using Large Language Models

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![University of Oulu](https://img.shields.io/badge/University-Oulu-003479)](https://www.oulu.fi/en)

> Research project for Affective Computing course at University of Oulu, 2025(Lecturer: Dr. Haoyu Chen)

## Abstract

Large Language Models (LLMs) have demonstrated strong performance in natural language understanding tasks, including emotion recognition, but their "black box" nature reduces trust and usability in affective computing applications. This research addresses this challenge by employing locally executed open-source LLMs for interpretable emotion analysis. The key innovation is exploiting the generative power of models not only for emotion classification but also to generate human-understandable justifications for predictions using Chain-of-Thought (CoT) prompting. We evaluate performance across different few-shot learning configurations (zero-shot, 3-shot, 5-shot) and tweet length thresholds, assessing models both quantitatively and qualitatively.

**Key Contributions:**
- Explainable emotion classification using Chain-of-Thought reasoning
- Comprehensive few-shot learning evaluation across multiple model architectures
- Comparison of Ollama models (Mistral 7B, Qwen2.5 7B) and HuggingFace FLAN-T5 models (small/base)
- Analysis of tweet length impact on classification performance
- Extensive performance metrics including accuracy, F1-scores, Cohen's Kappa, and MCC

## Team Members

| Name | Role | Email |
|------|------|-------|
| **Salwa Mostafa** | Lead Implementation & Presentation | Salwa.Mostafa@oulu.fi |
| **Seyedata Jodeiri Seyedian** | Few-Shot Learning Implementation | Ata.Seyedian@oulu.fi |
| **Severi Raunama** | Chain-of-Thought Implementation | Severi.Raunama@student.oulu.fi |
| **Amir Mollazadeh** | Technical Report & Presentation | amir.mollazadeh@student.oulu.fi |
| **Tommi Niemi** | Technical Report & Presentation | Tommi.Niemi@student.oulu.fi |
| **Tomi Luukkonen** | Technical Report & Presentation | Tomi.Luukkonen@student.oulu.fi |

*University of Oulu, Finland*

## Research Objectives

1. **Emotion Classification**: Classify text into 6 basic emotions (sadness, joy, love, anger, fear, surprise)
2. **Explainability**: Generate interpretable reasoning using Chain-of-Thought prompting
3. **Few-Shot Learning**: Evaluate model performance with varying numbers of examples (0, 3, 5 shots)
4. **Model Comparison**: Assess performance across different model architectures and parameter scales
5. **Tweet Length Analysis**: Investigate impact of example verbosity on model learning

## Dataset

**DAIR.AI Emotion Dataset** (split version)
- **Source**: Twitter messages labeled with emotions
- **Total samples**: 20,000 English texts
- **Emotion categories**: 6 (anger, fear, joy, love, sadness, surprise)
- **Splits**: Training (16,000), Validation (2,000), Test (2,000)
- **Test sample**: 500 stratified samples used for evaluation

**Class Distribution:**

| Emotion | Count | Percentage |
|---------|-------|------------|
| Joy | 5,362 | 33.5% |
| Sadness | 4,666 | 29.2% |
| Anger | 2,159 | 13.5% |
| Fear | 1,937 | 12.1% |
| Love | 1,304 | 8.2% |
| Surprise | 572 | 3.6% |

## Key Results

### Best Performing Models

| Model | Configuration | Accuracy | F1-Weighted | F1-Macro |
|-------|--------------|----------|-------------|----------|
| **FLAN-T5 Base (250M)** | 3-shot, 100 chars | **70.6%** | **0.697** | **0.550** |
| Qwen2.5 7B | 5-shot, 100 chars | 59.8% | 0.604 | 0.148 |
| Mistral 7B | 3/5-shot, 100 chars | 53.0% | 0.567 | 0.046 |
| FLAN-T5 Small (80M) | Zero-shot | 41.0% | 0.419 | 0.370 |

### Key Findings

1. **Architecture Over Scale**: FLAN-T5 Base (250M parameters) significantly outperformed models with 7B parameters, demonstrating that encoder-decoder architecture and task-specific training matter more than raw parameter count

2. **Optimal Tweet Length**: 100-character examples consistently yielded best performance across all models. Shorter examples enabled better focus on emotion-relevant features

3. **Few-Shot Learning Effects**: 
   - Qwen2.5 7B: Improved from 54.8% (zero-shot) to 59.8% (5-shot)
   - FLAN-T5 Small: Degraded in few-shot settings, suggesting insufficient capacity
   - FLAN-T5 Base: Modest improvement with few-shot examples

4. **Challenging Emotions**: "Love" was consistently the most difficult emotion to classify across all models

5. **Chain-of-Thought**: While quantitative improvements were modest, generated explanations significantly enhanced model transparency and trust

## Installation

### Prerequisites
- Python 3.8 or higher
- GPU recommended but not required
- 8GB+ RAM
- Internet connection for model downloads

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/affective-computing-llm.git
cd affective-computing-llm
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install and Setup Ollama

**For Ollama Models (Mistral 7B, Qwen2.5 7B):**

1. **Install Ollama:**
   - Visit [https://ollama.ai](https://ollama.ai)
   - Download and install for your operating system
   - Follow installation instructions

2. **Download Required Models:**
```bash
ollama pull mistral:7b
ollama pull qwen2.5:7b
```

3. **Verify Installation:**
```bash
ollama list
```

### Step 4: Setup HuggingFace (for FLAN-T5 Models)

1. **Create HuggingFace Account:**
   - Visit [https://huggingface.co](https://huggingface.co)
   - Sign up for free account

2. **Generate Access Token:**
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Copy your token

3. **Set Token in Code:**
   - Open `src/huggingface_evaluation.py`
   - Replace the token placeholder with your token:
```python
   hf_token = "YOUR_TOKEN_HERE"
```

**Note:** FLAN-T5 models (flan-t5-small and flan-t5-base) will be automatically downloaded when you run the evaluation script for the first time.


## Evaluation Metrics

Our evaluation employs comprehensive metrics to assess model performance:

- **Accuracy**: Overall proportion of correct predictions
- **Balanced Accuracy**: Accuracy adjusted for class distribution
- **F1-Score (Weighted)**: Harmonic mean of precision and recall, weighted by class support
- **F1-Score (Macro)**: Unweighted average F1 across all classes
- **Matthews Correlation Coefficient (MCC)**: Robust metric considering all confusion matrix elements
- **Cohen's Kappa**: Agreement measure accounting for chance
- **Per-Emotion F1-Scores**: Class-specific performance for each of six emotions

## Methodology

### Models Evaluated

**Ollama Models:**
- **Mistral 7B**: 7 billion parameter autoregressive decoder model
- **Qwen2.5 7B**: 7 billion parameter model from Alibaba Cloud

**HuggingFace Models:**
- **FLAN-T5 Small**: 80 million parameter encoder-decoder model
- **FLAN-T5 Base**: 250 million parameter encoder-decoder model

### Few-Shot Learning Configurations

- **Zero-shot**: Model receives only task instruction, no examples
- **3-shot**: Model receives 3 balanced examples covering different emotions
- **5-shot**: Model receives 5 balanced examples

### Tweet Length Thresholds

Few-shot examples were filtered by maximum character length:
- **100 characters**: Short, concise examples
- **200 characters**: Medium-length examples
- **500 characters**: Longer, more detailed examples

### Chain-of-Thought Prompting

Models generate explicit reasoning before predictions. Example:

**Input Text:** "I realized my mistake and I'm really feeling terrible and thinking that i shouldn't do that"

**Generated Reasoning:** "The phrases 'really feeling terrible' and 'i shouldn't do that' strongly indicate regret and remorse. The speaker is acknowledging a negative action and experiencing a negative emotional response to it. This aligns most closely with sadness and potentially fear (of repeating the mistake)."

**Prediction:** sadness (Correct)

## Insights and Discussion

### Why FLAN-T5 Base Outperformed Larger Models

The FLAN-T5 Base model (250M parameters) achieved 70.6% accuracy, significantly outperforming both Mistral 7B (53.0%) and Qwen2.5 7B (59.8%) despite having far fewer parameters. Key factors:

1. **Encoder-Decoder Architecture**: Better suited for classification tasks than autoregressive decoders
2. **Task-Specific Training**: FLAN-T5 was trained on diverse instruction-following tasks
3. **Text-to-Text Framework**: Natural fit for emotion classification

### Few-Shot Learning Capacity Threshold

Our results reveal a capacity threshold for effective few-shot learning:
- **Below threshold (FLAN-T5 Small)**: Performance degraded with examples (41.0% → 32.4%)
- **Above threshold (Qwen2.5 7B, FLAN-T5 Base)**: Performance improved with examples

This suggests smaller models suffer from "early ascent" where limited examples retrieve incorrect patterns before sufficient task learning occurs.

### Optimal Tweet Length

100-character examples consistently yielded best results because:
- Reduced cognitive load on models
- Higher information density
- Less noise from filler words
- Better focus on emotion-relevant features

### Chain-of-Thought Value

While CoT prompting showed modest quantitative improvements, it provided significant qualitative benefits:
- Increased transparency and interpretability
- Revealed model reasoning process
- Enhanced trust in predictions
- Enabled error analysis


## Acknowledgments

- **Course**: Affective Computing, University of Oulu, 2025
- **Dataset**: DAIR.AI Emotion Dataset ([Saravia et al., 2018](https://www.aclweb.org/anthology/D18-1404))
- **Models**: Mistral AI, Alibaba Cloud (Qwen), Google Research (FLAN-T5)
- **Frameworks**: Ollama, HuggingFace Transformers

## References

1. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.
2. Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.
3. Saravia, E., et al. (2018). CARER: Contextualized Affect Representations for Emotion Recognition. EMNLP 2018.
4. Wei, J., et al. (2023). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.

