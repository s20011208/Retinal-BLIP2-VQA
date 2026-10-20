"""Frozen answer scoring and strict generation parsing, independent of model weights."""
import re
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

SCORING_VERSION = "answer_only_sum_with_eos_v1"
PARSER_VERSION = "whole_output_exact_alias_v1"


def answer_log_scores(logits: torch.Tensor, labels: torch.Tensor,
                      reduction: str = "sum") -> torch.Tensor:
    """Next-token conditional log likelihood, with -100 ignored.

    Logits must be aligned with the *text input*, as returned by BLIP-2 when
    labels are supplied. In the legacy prefix implementation official BLIP-2
    removes the prepended visual positions before returning these logits.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must have matching batch/text dimensions")
    if reduction not in ("sum", "mean"):
        raise ValueError("reduction must be sum or mean")
    target = labels[:, 1:]
    valid = target.ne(-100)
    counts = valid.sum(-1)
    if (counts == 0).any():
        raise ValueError("every sample needs at least one supervised next token")
    token_logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    chosen = token_logp.gather(-1, target.masked_fill(~valid, 0).unsqueeze(-1)).squeeze(-1)
    score = (chosen * valid).sum(-1)
    return score if reduction == "sum" else score / counts


def normalize_answer(text: str) -> str:
    """Only case/whitespace and one terminal punctuation mark are normalized."""
    return re.sub(r"[.!?]$", "", " ".join(str(text).strip().lower().split())).strip()


def parse_answer(text: str, candidates: Sequence[str],
                 aliases: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """Parse the whole generated answer; extra assertions remain invalid.

    Synonyms must be supplied by the frozen training-derived protocol. Never
    extract the first word or a yes/no substring from an unsupported assertion.
    """
    norm = normalize_answer(text)
    if not norm:
        return None
    matches = []
    for candidate in candidates:
        allowed = [candidate] + (aliases or {}).get(candidate, [])
        if norm in {normalize_answer(item) for item in allowed}:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def restricted_probabilities(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(torch.tensor(scores, dtype=torch.float64) / temperature, -1).tolist()


def decode_generated_answers(processor, sequences, prompt_input_ids=None, includes_prompt=False):
    """Decode all answer tokens, optionally remove an API-declared input prefix.

    transformers 4.46.3 BLIP-2 returns new tokens plus BOS, so its caller uses
    includes_prompt=False. The explicit prefix mode covers APIs that return
    prompt+continuation without ever extracting a first word or assertion.
    """
    if includes_prompt:
        if prompt_input_ids is None or sequences.shape[0] != prompt_input_ids.shape[0]:
            raise ValueError("prompt-returning API needs aligned original input IDs")
        length = prompt_input_ids.shape[1]
        if sequences.shape[1] < length or not torch.equal(sequences[:, :length], prompt_input_ids):
            raise ValueError("returned sequence does not begin with the exact known input prefix")
        sequences = sequences[:, length:]
    return [text.strip() for text in processor.batch_decode(sequences, skip_special_tokens=True)]
