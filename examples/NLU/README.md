# GP-LoRA for Natural Language Understanding

This folder contains experiments applying **GP-LoRA (Gauge-Projected Low-Rank Adaptation)** to RoBERTa and DeBERTa for the GLUE benchmark.

## Overview

GP-LoRA extends LoRA by applying a gauge-fixing projection after each optimizer step. The projection enforces the imbalance constraint AA^⊤ = μB^⊤B while preserving Δ = BA exactly, exploiting gauge symmetry to improve optimization dynamics without changing the effective weight update.

<p align="center">
<img src="figures/LoRA_NLU.PNG" width="700">
</p>

## GLUE Benchmark Results

We evaluate on all GLUE tasks:

<p align="center">
<img src="figures/deberta_lora_glue.jpg" width="700">
</p>

## Available Experiments

### GP-LoRA Scripts

| Model | Task | Script |
|-------|------|--------|
| RoBERTa-base | MNLI | `roberta_base_mnli_gplora.sh` |
| RoBERTa-base | SST-2 | `roberta_base_sst2_gplora.sh` |
| RoBERTa-base | MRPC | `roberta_base_mrpc_gplora.sh` |
| RoBERTa-base | CoLA | `roberta_base_cola_gplora.sh` |
| RoBERTa-base | QNLI | `roberta_base_qnli_gplora.sh` |
| RoBERTa-base | QQP | `roberta_base_qqp_gplora.sh` |
| RoBERTa-base | RTE | `roberta_base_rte_gplora.sh` |
| RoBERTa-base | STS-B | `roberta_base_stsb_gplora.sh` |
| RoBERTa-large | All tasks | `roberta_large_*_gplora.sh` |
| DeBERTa-XXL | All tasks | `deberta_v2_xxlarge_*.sh` |

### Standard LoRA Baselines

Corresponding standard LoRA scripts are provided for comparison:
- `roberta_base_mnli.sh`, `roberta_large_mnli.sh`, etc.

## Setup

### 1. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate gplora
```

### 2. Install the GP-LoRA library

```bash
# From the NLU directory
pip install -e ../..   # Install loralib with GP-LoRA
pip install -e .        # Install the transformers fork
```

## Running Experiments

### GP-LoRA Training

**Example: RoBERTa-base on MNLI**

```bash
bash roberta_base_mnli_gplora.sh
```

This runs:
```bash
python -m torch.distributed.launch --nproc_per_node=8 \
    examples/text-classification/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mnli \
    --apply_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --gp_lora \
    --gp_mu auto \
    --gp_eps 1e-4 \
    ...
```

The GP-LoRA flags are:
- `--gp_lora`: Enable gauge projection after each optimizer step
- `--gp_mu auto`: Dimension-calibrated imbalance ratio μ = r/m
- `--gp_eps 1e-4`: Gram matrix regularization for numerical stability

### Running All Experiments

**RoBERTa-base (all GLUE tasks)**:
```bash
# GP-LoRA
for task in mnli sst2 mrpc cola qnli qqp rte stsb; do
    bash roberta_base_${task}_gplora.sh
done

# Standard LoRA baselines
for task in mnli sst2 mrpc cola qnli qqp rte stsb; do
    bash roberta_base_${task}.sh
done
```

**RoBERTa-large (all GLUE tasks)**:
```bash
bash run_roberta_large_experiments.sh
```

**Complete suite**:
```bash
bash run_all_experiments.sh
```

## Evaluation

### Evaluate a trained checkpoint

```bash
python -m torch.distributed.launch --nproc_per_node=1 \
    examples/text-classification/run_glue.py \
    --model_name_or_path roberta-base \
    --lora_path ./mnli_gplora/model/checkpoint-best/lora.bin \
    --task_name mnli \
    --do_eval \
    --output_dir ./output \
    --apply_lora \
    --lora_r 8 \
    --lora_alpha 16
```

### Download pretrained checkpoints

We provide LoRA checkpoints for quick evaluation:

| Model | Task | Checkpoint |
|-------|------|------------|
| RoBERTa-base | MNLI | `roberta_base_lora_mnli.bin` |
| RoBERTa-large | MNLI | `roberta_large_lora_mnli.bin` |

## GP-LoRA Configuration

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Enable GP-LoRA | `--gp_lora` | False | Apply gauge projection |
| Imbalance ratio | `--gp_mu` | `auto` | μ for constraint AA^⊤ = μB^⊤B |
| Regularization | `--gp_eps` | `1e-4` | Gram matrix regularization |

### Understanding μ (mu)

The imbalance ratio μ controls the target factorization balance:
- `μ = "auto"`: Uses r/m (dimension-calibrated, recommended)
- `μ = 1.0`: Balanced factorization AA^⊤ = B^⊤B
- `μ > 1`: Larger A relative to B
- `μ < 1`: Smaller A relative to B

The dimension-calibrated choice μ = r/m is motivated by variance preservation in the forward pass.

## Transfer Learning for Low-Resource Tasks

For MRPC, RTE, and STS-B (which have smaller training sets), we recommend:

1. First train on MNLI with GP-LoRA
2. Initialize from the MNLI checkpoint for the target task

```bash
# Train on MNLI first
bash roberta_base_mnli_gplora.sh

# Then fine-tune on RTE (update the script to point to MNLI checkpoint)
bash roberta_base_rte_gplora.sh
```

## Data Augmentation

We also support data augmentation techniques:

**Cutoff augmentation**:
```bash
bash mnli.cutoff.sh
```

**R-Drop regularization**:
```bash
bash mnli.rdrop.sh
```

## Adapter Baselines

For comparison with adapter methods:
```bash
bash adapter_houlsby_roberta_large_mnli.sh
bash adapter_pfeiffer_roberta_large_mnli.sh
```

## Citation

If you use this code, please cite:

```bibtex
@misc{gplora2024,
    title={Gauge-Projected Low-Rank Adaptation},
    author={[Your Name]},
    year={2024}
}

@inproceedings{hu2022lora,
    title={{LoRA}: Low-Rank Adaptation of Large Language Models},
    author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
    booktitle={International Conference on Learning Representations},
    year={2022}
}
```

## Acknowledgments

The NLU experiment infrastructure and Hugging Face Transformers fork are adapted from the [Microsoft LoRA repository](https://github.com/microsoft/LoRA). We thank the original authors for their comprehensive codebase.
