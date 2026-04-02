"""Training script for IWSLT baseline models.

Usage:
    python scripts/train.py --config configs/transformer_ende.yaml
    python scripts/train.py --config configs/seq2seq_ende.yaml
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    build_tokenizer,
    create_dataloaders,
    decode_tokens,
    load_iwslt_data,
    load_tokenizer,
)
from src.models.seq2seq_attention import Seq2SeqAttention
from src.models.transformer import TransformerModel
from src.utils.metrics import compute_bleu
from src.utils.scheduler import get_scheduler
from src.utils.training import evaluate, save_checkpoint, train_one_epoch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, src_vocab_size: int, tgt_vocab_size: int) -> nn.Module:
    """Instantiate a model based on configuration."""
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]

    if model_type == "transformer":
        return TransformerModel(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=model_cfg["d_model"],
            nhead=model_cfg["nhead"],
            num_encoder_layers=model_cfg["num_encoder_layers"],
            num_decoder_layers=model_cfg["num_decoder_layers"],
            dim_feedforward=model_cfg["dim_feedforward"],
            dropout=model_cfg["dropout"],
            max_len=model_cfg.get("max_len", 5000),
            pad_idx=PAD_IDX,
        )
    elif model_type == "seq2seq_attention":
        return Seq2SeqAttention(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            embed_dim=model_cfg["embed_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            attention_dim=model_cfg["attention_dim"],
            num_layers=model_cfg["num_layers"],
            dropout=model_cfg["dropout"],
            pad_idx=PAD_IDX,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_bleu_score(
    model: nn.Module,
    dataloader,
    tgt_sp,
    device: torch.device,
    max_len: int = 256,
) -> float:
    """Decode and compute BLEU on a dataset split."""
    model.eval()
    hypotheses = []
    references = []

    for src, tgt in dataloader:
        src = src.to(device)
        decoded = model.greedy_decode(src, BOS_IDX, EOS_IDX, max_len)

        for i in range(decoded.size(0)):
            hyp = decode_tokens(tgt_sp, decoded[i].tolist())
            ref = decode_tokens(tgt_sp, tgt[i].tolist())
            hypotheses.append(hyp)
            references.append(ref)

    result = compute_bleu(hypotheses, references)
    return result["score"]


def main():
    parser = argparse.ArgumentParser(description="Train IWSLT baseline model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Set seed
    set_seed(cfg["training"]["seed"])

    # Device
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

    train_pairs = data["train"]
    val_pairs = data.get("validation", data.get("valid", []))
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs")

    # Build or load tokenizers
    tokenizer_dir = data_cfg["tokenizer_dir"]
    src_model_path = os.path.join(tokenizer_dir, f"src_{data_cfg['src_lang']}.model")
    tgt_model_path = os.path.join(tokenizer_dir, f"tgt_{data_cfg['tgt_lang']}.model")

    if os.path.exists(src_model_path) and os.path.exists(tgt_model_path):
        print("Loading existing tokenizers...")
        src_sp = load_tokenizer(src_model_path)
        tgt_sp = load_tokenizer(tgt_model_path)
    else:
        print("Training tokenizers...")
        src_sentences = [p[0] for p in train_pairs]
        tgt_sentences = [p[1] for p in train_pairs]
        src_sp = build_tokenizer(
            src_sentences,
            f"src_{data_cfg['src_lang']}",
            vocab_size=data_cfg["vocab_size"],
            model_type=data_cfg["model_type"],
            output_dir=tokenizer_dir,
        )
        tgt_sp = build_tokenizer(
            tgt_sentences,
            f"tgt_{data_cfg['tgt_lang']}",
            vocab_size=data_cfg["vocab_size"],
            model_type=data_cfg["model_type"],
            output_dir=tokenizer_dir,
        )

    src_vocab_size = src_sp.get_piece_size()
    tgt_vocab_size = tgt_sp.get_piece_size()
    print(f"  Source vocab size: {src_vocab_size}")
    print(f"  Target vocab size: {tgt_vocab_size}")

    # Create dataloaders
    train_cfg = cfg["training"]
    train_loader = create_dataloaders(
        train_pairs, src_sp, tgt_sp,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = create_dataloaders(
        val_pairs, src_sp, tgt_sp,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    # Build model
    model = build_model(cfg, src_vocab_size, tgt_vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['type']} ({n_params:,} trainable parameters)")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX,
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    total_steps = len(train_loader) * train_cfg["num_epochs"]
    scheduler = get_scheduler(
        train_cfg.get("scheduler", "none"),
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 4000),
        total_steps=total_steps,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_bleu = 0.0
    if args.resume:
        from src.utils.training import load_checkpoint

        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        best_bleu = ckpt.get("best_bleu", 0.0)
        print(f"Resumed from epoch {start_epoch}, best BLEU: {best_bleu:.2f}")

    # Tensorboard
    log_dir = cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    ckpt_cfg = cfg["checkpoint"]
    os.makedirs(ckpt_cfg["save_dir"], exist_ok=True)
    decode_max_len = cfg.get("decode", {}).get("max_len", 256)

    print(f"\nStarting training for {train_cfg['num_epochs']} epochs...")
    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=train_cfg["clip_grad"],
            scheduler=scheduler,
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        # Compute BLEU on validation set
        bleu_score = compute_bleu_score(
            model, val_loader, tgt_sp, device, max_len=decode_max_len
        )

        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("BLEU/val", bleu_score, epoch)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{train_cfg['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"BLEU: {bleu_score:.2f} | LR: {lr:.6f}"
        )

        # Save checkpoint
        is_best = bleu_score > best_bleu
        if is_best:
            best_bleu = bleu_score
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(ckpt_cfg["save_dir"], "best_model.pt"),
                best_bleu=best_bleu,
            )
            print(f"  ✓ New best model (BLEU: {best_bleu:.2f})")

        if (epoch + 1) % ckpt_cfg["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(ckpt_cfg["save_dir"], f"checkpoint_epoch{epoch + 1}.pt"),
                best_bleu=best_bleu,
            )

    # Save final model
    save_checkpoint(
        model, optimizer, train_cfg["num_epochs"] - 1, val_loss,
        os.path.join(ckpt_cfg["save_dir"], "final_model.pt"),
        best_bleu=best_bleu,
    )

    writer.close()
    print(f"\nTraining complete! Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()
