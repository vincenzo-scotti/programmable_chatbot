import torch
import torch.nn.functional as F

from typing import Optional, Literal

from .const import IGNORE_INDEX


def nll(
        logits,
        labels,
        reduction: Optional[Literal['mean', 'sum', 'batchmean', 'batchsum', 'seqmean', 'seqsum']] = None
) -> torch.tensor:
    # Shift left labels
    shift_labels: torch.tensor = labels[..., 1:].contiguous()
    # Shift right logits
    shift_logits: torch.tensor = logits[..., :-1, :].contiguous()

    # Compute element-wise loss
    loss: torch.tensor = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none',
        ignore_index=IGNORE_INDEX
    ).view(shift_labels.size())
    # Apply reduction if required
    if reduction is not None:
        # Compute loss reduction depending on the required approach
        if reduction == 'mean':
            # Get number of tokens
            n_tokens: torch.tensor = (shift_labels != IGNORE_INDEX).float().sum()
            # Apply reduction
            loss = loss.sum() / n_tokens
        elif reduction == 'sum':
            # Apply reduction
            loss = loss.sum()
        elif reduction == 'batchmean':
            # Get number of tokens
            n_tokens: torch.tensor = (shift_labels != IGNORE_INDEX).float().sum(1)
            # Apply reduction
            loss = loss.sum(1) / n_tokens
            loss = loss.mean()
        elif reduction == 'batchsum':
            # Get number of tokens
            n_tokens: torch.tensor = (shift_labels != IGNORE_INDEX).float().sum(1)
            # Apply reduction
            loss = loss.sum(1) / n_tokens
            loss = loss.sum()
        elif reduction == 'seqmean':
            # Get number of tokens
            n_tokens: torch.tensor = (shift_labels != IGNORE_INDEX).float().sum(1)
            # Apply reduction
            loss = loss.sum(1) / n_tokens
        elif reduction == 'seqsum':
            # Apply reduction
            loss = loss.sum(1)
        else:
            raise ValueError(f'Unsupported reduction approach: \'{reduction}\'')

    return loss
