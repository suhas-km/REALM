# REALM: Reward Enhanced Alignment Learning Method

REALM is a comprehensive framework for fine-tuning language models using Reinforcement Learning from Human Feedback (RLHF) techniques. It combines state-of-the-art reward modeling with both PPO (Proximal Policy Optimization) and DPO (Direct Preference Optimization) approaches.

## Features

- **Multiple reward strategies**:
  - QRM direct reward scoring using Llama 3.1 reward model
  - Harmonic blending of semantic similarity and reward scores

- **Multiple fine-tuning methods**:
  - PPO (Proximal Policy Optimization)
  - DPO (Direct Preference Optimization)
  - QRM-PPO (Direct QRM reward with PPO)
  - QRM-DPO (Direct QRM reward with DPO)

- **Evaluation capabilities**:
  - Truthfulness evaluation using the TruthfulQA benchmark
  - Reward prediction for prompt-response pairs

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/REALM.git
   cd REALM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Authenticate with Hugging Face:
   ```bash
   huggingface-cli login
   ```

4. Configure your environment:
   - Edit `config/config.yaml` to set appropriate model paths and parameters

## Usage

REALM supports several modes of operation:

### Prediction Mode

Predict the reward for a given prompt-response pair using the harmonic blending approach:

```bash
python main.py --mode predict --prompt "Tell me about reinforcement learning." --response "Reinforcement learning is a machine learning technique where an agent learns to make decisions by taking actions in an environment to maximize a reward signal."
```

### Training with PPO (Harmonic Blend)

Fine-tune a model using Proximal Policy Optimization with harmonic blend reward:

```bash
python main.py --mode ppo --output_dir "./models/ppo_finetuned"
```

### Training with DPO (Harmonic Blend)

Fine-tune a model using Direct Preference Optimization with harmonic blend reward:

```bash
python main.py --mode dpo
```

### Training with QRM-PPO (Direct QRM Reward)

Fine-tune a model using PPO with direct QRM reward scoring:

```bash
python main.py --mode qrm_ppo --output_dir "./models/qrm_ppo_finetuned"
```

### Training with QRM-DPO (Direct QRM Reward)

Fine-tune a model using DPO with direct QRM reward scoring:

```bash
python main.py --mode qrm_dpo
```

### Evaluation on TruthfulQA

Evaluate a fine-tuned model on the TruthfulQA benchmark:

```bash
python main.py --mode evaluate --model_type qrm_ppo --max_new_tokens 128
```

## Advanced Configuration

### Reward Model Configuration

The project uses the QRM-Llama3.1-8B-v2 reward model by default, but you can configure different reward models in the `config.yaml` file:

```yaml
qrm_reward:
  model_id: "nicolinho/QRM-Llama3.1-8B-v2"
```

### Embedding Model Configuration

Semantic similarity is calculated using the Lajavaness bilingual embedding model:

```yaml
embedding:
  model_id: "Lajavaness/bilingual-embedding-large"
```

### Harmonic Blend Configuration

You can adjust the weight parameter (alpha) for the harmonic blend:

```yaml
reward:
  method: "harmonic_blend"
  alpha: 0.5  # Weight parameter for harmonic blend
```

### PPO and DPO Configurations

Fine-tuning parameters can be adjusted in the config file:

```yaml
rlhf:
  ppo:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: 8
    # other parameters...
  
  dpo:
    model_name: "meta-llama/Llama-3.1-8B-Instruct"
    learning_rate: 5e-7
    # other parameters...
```

## Project Structure

```
REALM/
├── config/               # Configuration files
├── data/                 # Data processing and datasets
├── inference/            # Inference and prediction code
├── models/               # Model implementations
├── rlhf/                 # RLHF implementations (PPO, DPO)
├── utils/                # Utility functions
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## References & Acknowledgements

This project builds upon several key papers and libraries:

- [Llama 3.1](https://ai.meta.com/blog/meta-llama-3/) by Meta AI
- [QRM-Llama](https://github.com/nicolinho/QRM-Llama) by Nicolinho
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) by Rafailov et al.
- [TruthfulQA](https://arxiv.org/abs/2109.07958) by Lin et al.