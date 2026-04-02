"""Script to prepare tokenizers from IWSLT data.

Usage:
    python scripts/prepare_data.py --config configs/transformer_ende.yaml
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import build_tokenizer, load_iwslt_data


def main():
    parser = argparse.ArgumentParser(description="Prepare data and tokenizers")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Load data
    print(f"Loading IWSLT {data_cfg['year']} {data_cfg['src_lang']}-{data_cfg['tgt_lang']}...")
    data = load_iwslt_data(
        src_lang=data_cfg["src_lang"],
        tgt_lang=data_cfg["tgt_lang"],
        year=data_cfg["year"],
        dataset_name=data_cfg.get("dataset_name"),
    )

    train_pairs = data["train"]
    print(f"  Training pairs: {len(train_pairs)}")

    # Train tokenizers
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_sentences = [p[0] for p in train_pairs]
    tgt_sentences = [p[1] for p in train_pairs]

    print(f"Training source tokenizer ({data_cfg['src_lang']})...")
    src_sp = build_tokenizer(
        src_sentences,
        f"src_{data_cfg['src_lang']}",
        vocab_size=data_cfg["vocab_size"],
        model_type=data_cfg["model_type"],
        output_dir=tokenizer_dir,
    )
    print(f"  Source vocab size: {src_sp.get_piece_size()}")

    print(f"Training target tokenizer ({data_cfg['tgt_lang']})...")
    tgt_sp = build_tokenizer(
        tgt_sentences,
        f"tgt_{data_cfg['tgt_lang']}",
        vocab_size=data_cfg["vocab_size"],
        model_type=data_cfg["model_type"],
        output_dir=tokenizer_dir,
    )
    print(f"  Target vocab size: {tgt_sp.get_piece_size()}")

    # Print sample tokenization
    print("\nSample tokenizations:")
    for i in range(min(3, len(train_pairs))):
        src, tgt = train_pairs[i]
        src_tokens = src_sp.encode(src, out_type=str)
        tgt_tokens = tgt_sp.encode(tgt, out_type=str)
        print(f"\n  Source: {src}")
        print(f"  Tokens: {src_tokens}")
        print(f"  Target: {tgt}")
        print(f"  Tokens: {tgt_tokens}")

    print(f"\nTokenizers saved to {tokenizer_dir}/")


if __name__ == "__main__":
    main()
