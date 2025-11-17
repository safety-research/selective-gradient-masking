# Selective Gradient Masking (SGTM)

Code accompanying the paper **"Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs"**.

This repository provides the implementation of **Selective Gradient Masking (SGTM)**, a method for targeted capability removal in language models through parameter-level gradient manipulation. The code includes training scripts, model implementations, and analysis notebooks used in the paper's experiments.

## Overview

SGTM enables selective forgetting of specific capabilities in language models by:
- Dedicating parameters to the target capability
- Masking gradients during training to update only relevant parameters
- Removing (ablating) dedicated parameters after training

## Installation

```bash
uv pip install -e .
```

## Running Tests

Tests are implemented using pytest:

```bash
pytest ./src -sv
```

## Code Structure

```
├── src/sgtm/
│   ├── model/          # SGTM model implementations and masking strategies
│   │   └── tests/      # Unit tests for model components
│   ├── train/          # Training scripts for different experiments
│   └── data/           # Data processing and tokenization scripts
├── configs/
│   ├── tinystories/    # Model hyperparameters for TinyStories experiments (8M, 18M, 34M, 64M)
│   └── wiki/           # Model hyperparameters for Wikipedia experiments (34M, 64M, 125M, 254M)
├── scripts/
│   ├── tinystories/    # Bash scripts for TinyStories experiments
│   ├── wiki/           # Bash scripts for Wikipedia experiments
│   └── data/           # Data preparation scripts
└── notebooks/          # Analysis and visualization notebooks
```

## Masking Strategies

The repository implements three complementary approaches to selective gradient masking:

### 1. **Parameter Masking** (SGTM in the paper)
The primary method used in the paper. Parameters are split into "forget" and "retain" dimensions. When training on retain data, gradients for forget dimensions are zeroed after the backward pass, preventing updates to those parameters.

**Implementation**: `parameter_masking.py`

### 2. **Gradient Routing** ([Cloud et al., 2024](https://arxiv.org/abs/2410.04332))
Based on Cloud et al.'s approach, this method operates at the activation level rather than parameters. Uses `.detach()` during forward pass to stop gradient flow through specific activation dimensions, preventing gradients from reaching the corresponding parameters.

**Implementation**: `gradient_routing.py`

### 3. **Activation Masking**
Zeros out activations corresponding to forget dimensions during the forward pass on retain data, effectively routing information through retain parameters only.

**Implementation**: `activation_masking.py`

## Model Implementation

### Initializing a Model with SGTM

SGTM extends the GPT-Neo architecture from HuggingFace Transformers. Here's how to create a model with selective gradient masking:

```python
from transformers import GPTNeoConfig
from sgtm.model import GPTNeoForCausalLMSGTM

# Configure the model with SGTM parameters
config = GPTNeoConfig(
    vocab_size=50257,
    hidden_size=512,
    num_layers=12,
    num_heads=32,
    max_position_embeddings=2048,
    # SGTM-specific parameters
    retain_mlp_dim=1984,           # MLP dimensions for retain data (out of 2048 total)
    retain_attn_heads=31,          # Attention heads for retain data (out of 32 total)
    masking_strategy="parameter_masking",  # Strategy to use
    split_masked_weights=True,    # Use SplitLinearOut for keeping retain and forget weights in separate parameters (no effect on traininig)
    sgtm_mask_embeddings=False,   # Whether to mask embedding gradients during training
)

# Initialize the model
model = GPTNeoForCausalLMSGTM(config)

# During training, specify which data split you're training on
outputs = model(
    input_ids=batch["input_ids"],
    labels=batch["labels"],
    sgtm_mode="forget"  # or "retain" or "default"
)

# After backward pass, adjust gradients based on the data split
loss.backward()
model.adjust_gradients(sgtm_mode="forget")
optimizer.step()
```

### Key Classes

- **`GPTNeoForCausalLMSGTM`**: Main model class with SGTM support
- **`GPTNeoModelSGTM`**: Base transformer with selective masking
- **`GPTNeoBlockSGTM`**: Individual transformer block with masking
- **Configuration parameters**:
  - `retain_mlp_dim`: Number of MLP dimensions dedicated to retain data
  - `retain_attn_heads`: Number of attention heads dedicated to retain data
  - `masked_layers`: List of layer indices to apply masking
  - `masking_strategy`: Which masking strategy to use
  - `split_masked_weights`: Whether to physically split parameters

## Experiments

### TinyStories Experiments

The TinyStories experiments aim at removing a knowledge of a language from a model trained on bilingual TinyStories.

#### Dataset

The bilingual English-Spanish TinyStories dataset is available on HuggingFace:
- **Dataset**: [`ffuuugor/tinystories-spanish`](https://huggingface.co/datasets/ffuuugor/tinystories_spanish) (derived from [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories))
- **Translation**: The dataset was originally in English. The Spanish translation was created using `src/sgtm/data/translate_to_spanish.py`

#### Data Preparation

Before running experiments, tokenize and split the data:

```bash
bash scripts/data/tinystories_tokenize.sh
```

This will:
1. Download the dataset from HuggingFace
2. Separate English and Spanish versions
3. Tokenize both using GPT-2 tokenizer
4. Save to `data/datasets/tinystories_split/en` (retain) and `data/datasets/tinystories_split/es` (forget)

#### Running Experiments

Key experiments from the paper can be launched using scripts in `scripts/tinystories/`:

```bash
# Train a model with SGTM on TinyStories with varying rates of undiscovered forget data
bash scripts/tinystories/mislabeling.sh

# Train a model with SGTM on TinyStories while tracking calibrated loss on every checkpoint for trade-off analysis.
bash scripts/tinystories/tradeoff.sh
```

#### Analysis Notebooks

- **`notebooks/tinystories/retain_forget_tradeoff.ipynb`**: Produces the 2D forget/retain trade-off graph comparing SGTM, Gradient Routing, and data filtering baselines
- **`notebooks/tinystories/undiscovered_rate.ipynb`**: Analyzes the effect of undiscovered forget data (mislabeling) on model performance
- **`notebooks/tinystories/grad_norms.ipynb`**: Computes and visualizes gradient norms for forget and retain parameters across different data splits

### Wikipedia Experiments


#### Data Preparation

[ORES topic data](https://www.mediawiki.org/wiki/ORES/Articletopic) is available for download [here](https://analytics.wikimedia.org/published/datasets/topics/).

We then filter English articles only by

```bash
cat topicsForAllWikipediaPages2020-08-24.csv | grep "enwiki," > enwiki_topics2020.csv
```

To prepare the Wikipedia dataset for experiments:

```bash
bash scripts/data/prepare_wikipedia.sh
```

This script will:
1. Download the Wikipedia dataset from HuggingFace
2. Match Wikipedia articles with ORES topic data
3. Tokenize the data using GPT-2 tokenizer
4. Split articles into chunks of context size (1024 tokens)
5. Split data into three separate datasets:
   - **Forget**: Stem.Biology articles (target capability to remove)
   - **Forget-adjacent**: Chemistry, Medicine, and Environment articles (related topics)
   - **General knowledge**: All other topics
6. 5000 articles from each category are reserved for the test set


#### Running Experiments

Key experiments from the paper can be launched using:

```bash
bash scripts/wiki/run.sh
```

This script launches 4 runs:
- SGTM
- Weak filter
- Strict filter
- No filter

Additional baseline methods:

```bash
# RMU (Representation Misdirection for Unlearning)
bash scripts/wiki/rmu.sh

# Finetuning after unlearning
bash scripts/wiki/finetune.sh
```

The `rmu.sh` script implements the RMU baseline method, which performs unlearning by steering model representations. The `finetune.sh` script can be used to finetune models after unlearning to evaluate robustness.

#### Analysis Notebooks

- **`notebooks/wiki/retain_forget_tradeoff.ipynb`**: Produces 2D trade-off plots comparing forget loss (Biology) against retain loss (Culture/Geography/History) and forget-adjacent loss (Medicine/Chemistry/Environment)
- **`notebooks/wiki/finetuning.ipynb`**: Analyzes model performance during finetuning, comparing how different unlearning methods (RMU, filters, SGTM) hold up under continued training

## Citation

```bibtex
(Citation will be added upon paper publication)
```