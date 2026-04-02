"""BLEU score computation using sacrebleu."""

import sacrebleu


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
    tokenize: str = "13a",
) -> dict:
    """Compute BLEU score using sacrebleu.

    Args:
        hypotheses: List of system output strings.
        references: List of reference strings.
        tokenize: Tokenization method for sacrebleu.

    Returns:
        dict with keys: score, counts, totals, precisions, bp, sys_len, ref_len.
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize)
    return {
        "score": bleu.score,
        "counts": bleu.counts,
        "totals": bleu.totals,
        "precisions": bleu.precisions,
        "bp": bleu.bp,
        "sys_len": bleu.sys_len,
        "ref_len": bleu.ref_len,
    }
