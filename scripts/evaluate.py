"""Evaluation script for IWSLT baseline models.

Usage:
    python scripts/evaluate.py --config configs/transformer_ende.yaml \
        --checkpoint checkpoints/transformer/best_model.pt
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    create_dataloaders,
    decode_tokens,
    load_iwslt_data,
    load_tokenizer,
)
from src.utils.metrics import compute_bleu
from src.utils.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate IWSLT baseline model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Data split to evaluate on"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save translations"
    )
    parser.add_argument(
        "--max_len", type=int, default=256, help="Maximum decoding length"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_cfg = cfg["data"]
    print(f"Loading IWSLT {data_cfg['year']} {data_cfg['src_lang']}-{data_cfg['tgt_lang']}...")
    data = load_iwslt_data(
        src_lang=data_cfg["src_lang"],
        tgt_lang=data_cfg["tgt_lang"],
        year=data_cfg["year"],
        dataset_name=data_cfg.get("dataset_name"),
    )

    split_data = data.get(args.split, [])
    if not split_data:
        print(f"No data found for split '{args.split}'")
        sys.exit(1)

    print(f"  {args.split}: {len(split_data)} pairs")

    # Load tokenizers
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_sp = load_tokenizer(
        os.path.join(tokenizer_dir, f"src_{data_cfg['src_lang']}.model")
    )
    tgt_sp = load_tokenizer(
        os.path.join(tokenizer_dir, f"tgt_{data_cfg['tgt_lang']}.model")
    )

    src_vocab_size = src_sp.get_piece_size()
    tgt_vocab_size = tgt_sp.get_piece_size()

    # Build model and load checkpoint
    # Import here to avoid circular deps at module level
    from scripts.train import build_model

    model = build_model(cfg, src_vocab_size, tgt_vocab_size).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Create dataloader
    eval_loader = create_dataloaders(
        split_data, src_sp, tgt_sp,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    # Decode and evaluate
    hypotheses = []
    references = []

    print("Generating translations...")
    for src, tgt in eval_loader:
        src = src.to(device)
        decoded = model.greedy_decode(src, BOS_IDX, EOS_IDX, args.max_len)

        for i in range(decoded.size(0)):
            hyp = decode_tokens(tgt_sp, decoded[i].tolist())
            ref = decode_tokens(tgt_sp, tgt[i].tolist())
            hypotheses.append(hyp)
            references.append(ref)

    # Compute BLEU
    result = compute_bleu(hypotheses, references)
    print(f"\n{'=' * 50}")
    print(f"BLEU Score: {result['score']:.2f}")
    print(f"Brevity Penalty: {result['bp']:.4f}")
    print(f"Precisions: {[f'{p:.1f}' for p in result['precisions']]}")
    print(f"Sys/Ref length: {result['sys_len']}/{result['ref_len']}")
    print(f"{'=' * 50}")

    # Save translations
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for hyp in hypotheses:
                f.write(hyp + "\n")
        print(f"Translations saved to {args.output}")

    # Print sample translations
    print(f"\nSample translations (first 5):")
    for i in range(min(5, len(hypotheses))):
        print(f"\n  Source: {split_data[i][0]}")
        print(f"  Reference: {references[i]}")
        print(f"  Hypothesis: {hypotheses[i]}")


if __name__ == "__main__":
    main()
