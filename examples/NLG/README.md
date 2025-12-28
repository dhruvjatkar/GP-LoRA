# GP-LoRA for Natural Language Generation

This folder contains experiments applying **GP-LoRA (Gauge-Projected Low-Rank Adaptation)** to GPT-2 for natural language generation tasks.

## Overview

GP-LoRA extends LoRA by applying a gauge-fixing projection after each optimizer step, enforcing the imbalance constraint AA^⊤ = μB^⊤B while preserving the weight update Δ = BA exactly. This exploits the gauge symmetry of the low-rank factorization to improve optimization dynamics.

<p align="center">
<img src="figures/LoRA_GPT2.PNG" width="700">
</p>

## Datasets

We evaluate on three NLG benchmarks:
- **E2E NLG Challenge**: Restaurant domain dialogue generation
- **WebNLG**: RDF-to-text generation
- **DART**: Open-domain structured data-to-text

## Directory Structure

```
NLG/
├── src/                    # Training and evaluation code
├── data/                   # Dataset files
├── eval/                   # Evaluation scripts
├── vocab/                  # GPT-2 vocabulary
├── run_e2e_gplora.sh       # GP-LoRA on E2E
├── run_dart_gplora.sh      # GP-LoRA on DART
├── run_webnlg_gplora.sh    # GP-LoRA on WebNLG
├── run_e2e_lora.sh         # Standard LoRA baseline
├── run_dart_lora.sh        # Standard LoRA baseline
└── run_webnlg_lora.sh      # Standard LoRA baseline
```

## Setup

1. **Environment**: Start with a PyTorch-enabled environment (e.g., `nvcr.io/nvidia/pytorch:20.03-py3`)

2. **Install dependencies**:
```bash
pip install -r requirement.txt
```

3. **Install the GP-LoRA library**:
```bash
cd ../..  # Navigate to repo root
pip install -e .
```

4. **Download pretrained checkpoints and prepare data**:
```bash
bash download_pretrained_checkpoints.sh
bash create_datasets.sh
cd eval && bash download_evalscript.sh && cd ..
```

## Running Experiments

### GP-LoRA Training

**E2E Dataset (GP-LoRA)**:
```bash
bash run_e2e_gplora.sh
```

This runs:
```bash
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_ft.py \
    --train_data ./data/e2e/train.jsonl \
    --valid_data ./data/e2e/valid.jsonl \
    --model_card gpt2.md \
    --lora_dim 4 \
    --lora_alpha 32 \
    --gp_lora \
    --gp_mu auto \
    --gp_eps 1e-4 \
    ...
```

The key GP-LoRA flags are:
- `--gp_lora`: Enable gauge projection
- `--gp_mu auto`: Use dimension-calibrated μ = r/m
- `--gp_eps 1e-4`: Regularization for numerical stability

**Other Datasets**:
```bash
bash run_dart_gplora.sh    # DART
bash run_webnlg_gplora.sh  # WebNLG
```

### Standard LoRA Baselines

For comparison, we also provide standard LoRA scripts:
```bash
bash run_e2e_lora.sh
bash run_dart_lora.sh
bash run_webnlg_lora.sh
```

## Evaluation Pipeline

### E2E Evaluation

1. **Generate outputs** using beam search:
```bash
python -m torch.distributed.launch --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/e2e/test.jsonl \
    --model_card gpt2.md \
    --init_checkpoint ./trained_models/GPT2_M/e2e_gplora/model.XXXX.pt \
    --lora_dim 4 \
    --lora_alpha 32 \
    --beam 10 \
    --length_penalty 0.8 \
    --work_dir ./trained_models/GPT2_M/e2e_gplora \
    --output_file predict.XXXX.jsonl
```

2. **Decode outputs**:
```bash
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/e2e_gplora/predict.XXXX.jsonl \
    --input_file ./data/e2e/test_formatted.jsonl \
    --output_ref_file e2e_ref.txt \
    --output_pred_file e2e_pred.txt
```

3. **Compute metrics**:
```bash
python eval/e2e/measure_scores.py e2e_ref.txt e2e_pred.txt -p
```

### WebNLG Evaluation

```bash
# Decode
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/webnlg_gplora/predict.XXXX.jsonl \
    --input_file ./data/webnlg_challenge_2017/test_formatted.jsonl \
    --ref_type webnlg --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_webnlg \
    --output_pred_file eval/GenerationEval/data/hypothesis_webnlg \
    --tokenize --lower

# Evaluate
cd eval/GenerationEval/
python eval.py -R data/references_webnlg/reference -H data/hypothesis_webnlg -nr 6 -m bleu,meteor,ter
```

### DART Evaluation

```bash
# Decode
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./trained_models/GPT2_M/dart_gplora/predict.XXXX.jsonl \
    --input_file ./data/dart/test_formatted.jsonl \
    --ref_type dart --ref_num 6 \
    --output_ref_file eval/GenerationEval/data/references_dart \
    --output_pred_file eval/GenerationEval/data/hypothesis_dart \
    --tokenize --lower

# Evaluate
cd eval/GenerationEval/
python eval.py -R data/references_dart/reference -H data/hypothesis_dart -nr 6 -m bleu,meteor,ter
```

## GP-LoRA Configuration

The gauge projection is controlled by:

| Flag | Description | Recommended |
|------|-------------|-------------|
| `--gp_lora` | Enable gauge projection after each step | Required |
| `--gp_mu` | Imbalance ratio μ | `auto` (uses r/m) |
| `--gp_eps` | Gram matrix regularization | `1e-4` |

## Run All Experiments

To run the complete experiment suite:
```bash
bash run_all_experiments.sh
```

## Citation

If you use this code, please cite GP-LoRA and the original LoRA paper:

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

The NLG experiment infrastructure is adapted from the [Microsoft LoRA repository](https://github.com/microsoft/LoRA). We thank the original authors for their excellent codebase.
