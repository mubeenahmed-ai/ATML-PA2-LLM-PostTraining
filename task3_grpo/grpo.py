from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy

INFORMATIVE_EPS = 1e-6  # tolerance used by group_relative_advantages (std <= eps => uninformative group)


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = INFORMATIVE_EPS):
    """Return one scalar advantage per sampled completion.

    A_k = (r_k - mean_group(r)) / (std_group(r) + eps-clamp), computed WITHIN each prompt group.
    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.

    Correction of the released starter: the starter used the mean / std of the WHOLE batch,
    which compares completions of different prompts (so an easy prompt's completions all get
    positive advantages and a hard prompt's all negative) and ignores group_ids entirely.
    A group whose rewards are identical gets advantage 0 for every completion.
    """
    adv = torch.zeros_like(rewards)
    for g in torch.unique(group_ids):
        idx = group_ids == g
        r = rewards[idx]
        mean = r.mean()
        std = r.std(unbiased=False).clamp_min(eps)
        adv[idx] = (r - mean) / std
    return adv


def group_statistics(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = INFORMATIVE_EPS):
    """Per-group reward mean / population std and the informative flag (std > eps)."""
    out = []
    for g in torch.unique(group_ids):
        r = rewards[group_ids == g].float()
        std = float(r.std(unbiased=False)) if r.numel() > 1 else 0.0
        out.append({"group": int(g), "mean": float(r.mean()), "std": std, "informative": std > eps, "n": int(r.numel())})
    return out


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def grpo_microbatch_loss(new_logp, old_logp, seq_adv, token_mask, ref_logp, eps, beta, loss_type,
                         max_completion_length, n_seq_total: int, n_tok_total: float):
    """Exact contribution of a micro-batch of completions to grpo_policy_loss on the full group
    batch (so the K completions can be processed in smaller pieces with gradient accumulation).
    Summing this over micro-batches equals grpo_policy_loss on the whole batch (checked in validate.py):
    the policy term is a mean over ALL sequences and the KL term a mean over ALL valid tokens."""
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    objective = torch.minimum(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv)
    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        per_sequence = token_sum / token_mask.sum(-1).clamp_min(1.0)
    elif loss_type == "dr_grpo":
        per_sequence = token_sum / float(max_completion_length)
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")
    policy_part = -per_sequence.sum() / float(n_seq_total)
    lr = ref_logp - new_logp
    per_token_kl = torch.exp(lr) - lr - 1.0
    kl_part = (per_token_kl * token_mask).sum() / max(float(n_tok_total), 1.0)
    affected = (((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float() * token_mask).sum()
    return policy_part + float(beta) * kl_part, {
        "policy_part": policy_part.detach(), "kl_part": kl_part.detach(), "clipped_tokens": affected.detach(),
        "ratio_sum": (ratio.detach() * token_mask).sum(),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
