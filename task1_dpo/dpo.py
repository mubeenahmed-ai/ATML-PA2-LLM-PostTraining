from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    L_DPO = -E[ log sigmoid( beta * m ) ],
    m     = [log pi(y+|x) - log ref(y+|x)] - [log pi(y-|x) - log ref(y-|x)]
          = (log pi(y+|x) - log pi(y-|x)) - (log ref(y+|x) - log ref(y-|x)).

    All inputs are SEQUENCE log-probabilities (sum over response tokens only).

    Correction of the released starter: the starter computed
        logits = beta * (policy_margin + ref_margin)
    but the objective requires the policy margin MINUS the reference margin.
    """
    chosen_logratio = policy_chosen_logp - ref_chosen_logp
    rejected_logratio = policy_rejected_logp - ref_rejected_logp

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    margin = chosen_logratio - rejected_logratio  # == policy_margin - ref_margin
    logits = beta * margin

    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "ref_margin_mean": ref_margin.detach().mean(),
        "margin_mean": margin.detach().mean(),
        "chosen_logratio_mean": chosen_logratio.detach().mean(),
        "rejected_logratio_mean": rejected_logratio.detach().mean(),
        "preference_accuracy": (margin.detach() > 0).float().mean(),
    }
