# IWSLT Prepare

Baseline models for the [IWSLT](https://iwslt.org/) (International Workshop on Spoken Language Translation) competition

## Project Structure

```
iwslt_prepare/
├── configs/                    # YAML configuration files
│   ├── transformer_ende.yaml   # Transformer en-de config
│   └── seq2seq_ende.yaml       # Seq2Seq en-de config
├── scripts/                    # Training & evaluation scripts
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation & BLEU scoring
│   └── prepare_data.py         # Data & tokenizer preparation
├── src/
│   ├── models/
│   │   ├── transformer.py      # Transformer encoder-decoder
│   │   └── seq2seq_attention.py # Seq2Seq with Bahdanau attention
│   ├── data/
│   │   ├── preprocessing.py    # Tokenizer & data loading utilities
│   │   └── dataset.py          # PyTorch Dataset & collate function
│   └── utils/
│       ├── metrics.py          # BLEU score computation (sacrebleu)
│       ├── training.py         # Train/eval loops & checkpointing
│       └── scheduler.py        # LR schedulers (warmup cosine, inverse sqrt)
├── requirements.txt
└── README.md
```

## Models

### 1. Transformer (Vaswani et al., 2017)
Standard encoder-decoder Transformer architecture with:
- Multi-head self-attention and cross-attention
- Sinusoidal positional encoding
- Xavier uniform parameter initialization
- Greedy decoding for inference

### 2. Seq2Seq with Bahdanau Attention
RNN-based sequence-to-sequence model with:
- Bidirectional GRU encoder
- Bahdanau (additive) attention mechanism
- Teacher forcing during training
- Greedy decoding for inference

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data and tokenizers

```bash
python scripts/prepare_data.py --config configs/transformer_ende.yaml
```

This downloads the IWSLT dataset via HuggingFace and trains SentencePiece BPE tokenizers.

### 3. Train a model

```bash
# Train Transformer baseline
python scripts/train.py --config configs/transformer_ende.yaml

# Train Seq2Seq baseline
python scripts/train.py --config configs/seq2seq_ende.yaml
```

Training supports:
- Automatic checkpoint saving (best model + periodic)
- TensorBoard logging
- Resume from checkpoint: `--resume checkpoints/transformer/checkpoint_epoch10.pt`

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/transformer_ende.yaml \
    --checkpoint checkpoints/transformer/best_model.pt \
    --split test \
    --output translations.txt
```

### 5. Monitor training

```bash
tensorboard --logdir logs/
```

## Configuration

All hyperparameters are controlled via YAML config files in `configs/`. Key settings:

| Parameter | Transformer | Seq2Seq |
|-----------|------------|---------|
| d_model / embed_dim | 512 | 256 |
| Layers | 6 enc + 6 dec | 2 (bidir) |
| Hidden dim | 2048 (FFN) | 512 |
| Attention heads | 8 | Additive |
| Dropout | 0.1 | 0.3 |
| LR Schedule | Inverse sqrt | None |
| Vocab size | 8000 (BPE) | 8000 (BPE) |

## Dependencies

- **PyTorch** ≥ 2.0 — model implementation and training
- **SentencePiece** — BPE tokenization
- **sacrebleu** — BLEU score evaluation
- **HuggingFace Datasets** — IWSLT data loading
- **TensorBoard** — training visualization
- **PyYAML** — configuration management

## License

This project is for research and competition purposes.
