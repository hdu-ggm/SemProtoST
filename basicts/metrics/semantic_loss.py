import torch.nn.functional as F
import torch



def semantic_consistency_loss(attn, meta, unseen_idx, seen_idx, temperature=0.1):
    attn_u = F.normalize(attn[:, unseen_idx, :], dim=-1)  # [B, Nu, K]
    attn_s = F.normalize(attn[:, seen_idx, :], dim=-1)  # [B, Ns, K]

    meta_u = F.normalize(meta[:, unseen_idx, :], dim=-1)
    meta_s = F.normalize(meta[:, seen_idx, :], dim=-1)

    # cross similarity
    sim_meta = torch.matmul(meta_u, meta_s.transpose(1, 2))  # [B, Nu, Ns]
    sim_attn = torch.matmul(attn_u, attn_s.transpose(1, 2))  # [B, Nu, Ns]

    # mask
    pos_mask = (sim_meta > 0.7).float()
    neg_mask = (sim_meta < 0.3).float()

    logits = sim_attn / temperature
    exp_logits = torch.exp(logits)

    pos_exp = exp_logits * pos_mask
    neg_exp = exp_logits * neg_mask

    denom = pos_exp.sum(dim=-1) + neg_exp.sum(dim=-1) + 1e-8
    loss = -torch.log((pos_exp.sum(dim=-1) + 1e-8) / denom)

    valid = (pos_mask.sum(dim=-1) > 0).float()
    loss = (loss * valid).sum() / (valid.sum() + 1e-6)

    return loss



def semantic_consistency_loss_sample(
    attn,
    meta,
    unseen_idx,
    seen_idx,
    temperature=0.1,
    pos_threshold=0.7,
    neg_threshold=0.3,
    num_candidates=256,
    num_pos=8,
    num_neg=32,
    max_anchors=None,
):
    """
    Sampled Semantic Consistency Constraint.

    Args:
        attn: [B, N, K], prototype attention distributions.
        meta: [B, N, C] or [N, C], node metadata.
        unseen_idx: anchor-node indices.
        seen_idx: candidate seen-node indices.
        num_candidates: sampled seen candidates per iteration.
        num_pos: maximum positives retained for each anchor.
        num_neg: maximum negatives retained for each anchor.
        max_anchors: optionally sample only a subset of anchors.
    """
    device = attn.device

    unseen_idx = torch.as_tensor(
        unseen_idx, dtype=torch.long, device=device
    )
    seen_idx = torch.as_tensor(
        seen_idx, dtype=torch.long, device=device
    )

    if unseen_idx.numel() == 0 or seen_idx.numel() == 0:
        return attn.sum() * 0.0

    # Optionally sample a fixed number of anchors.
    if max_anchors is not None and unseen_idx.numel() > max_anchors:
        anchor_perm = torch.randperm(
            unseen_idx.numel(), device=device
        )[:max_anchors]
        anchor_idx = unseen_idx[anchor_perm]
    else:
        anchor_idx = unseen_idx

    # Randomly sample a candidate bank from seen nodes.
    candidate_size = min(num_candidates, seen_idx.numel())
    candidate_perm = torch.randperm(
        seen_idx.numel(), device=device
    )[:candidate_size]
    candidate_idx = seen_idx[candidate_perm]

    # Metadata is static across the batch, so compute it only once.
    meta_static = meta[0] if meta.dim() == 3 else meta
    meta_static = meta_static.float()

    meta_anchor = F.normalize(
        meta_static[anchor_idx], dim=-1
    )  # [A, C]

    meta_candidate = F.normalize(
        meta_static[candidate_idx], dim=-1
    )  # [M, C]

    # Only A x M metadata similarities, rather than Nu x Ns.
    sim_meta = torch.matmul(
        meta_anchor, meta_candidate.transpose(0, 1)
    )  # [A, M]

    positive_scores = sim_meta.masked_fill(
        sim_meta <= pos_threshold, -torch.inf
    )

    # Negating similarity allows topk to select the least similar nodes.
    negative_scores = (-sim_meta).masked_fill(
        sim_meta >= neg_threshold, -torch.inf
    )

    pos_size = min(num_pos, candidate_size)
    neg_size = min(num_neg, candidate_size)

    pos_values, pos_local_idx = torch.topk(
        positive_scores, k=pos_size, dim=-1
    )

    neg_values, neg_local_idx = torch.topk(
        negative_scores, k=neg_size, dim=-1
    )

    pos_valid = torch.isfinite(pos_values)  # [A, P]
    neg_valid = torch.isfinite(neg_values)  # [A, Q]

    # Keep anchors that have at least one positive and one negative.
    valid_anchor = (
        pos_valid.any(dim=-1) & neg_valid.any(dim=-1)
    )

    if not valid_anchor.any():
        return attn.sum() * 0.0

    attn_anchor = F.normalize(
        attn[:, anchor_idx, :], dim=-1
    )  # [B, A, K]

    attn_candidate = F.normalize(
        attn[:, candidate_idx, :], dim=-1
    )  # [B, M, K]

    # Advanced indexing gives [B, A, P/Q, K].
    positive_attn = attn_candidate[:, pos_local_idx, :]
    negative_attn = attn_candidate[:, neg_local_idx, :]

    positive_logits = (
        attn_anchor.unsqueeze(2) * positive_attn
    ).sum(dim=-1) / temperature  # [B, A, P]

    negative_logits = (
        attn_anchor.unsqueeze(2) * negative_attn
    ).sum(dim=-1) / temperature  # [B, A, Q]

    positive_logits = positive_logits.masked_fill(
        ~pos_valid.unsqueeze(0), -torch.inf
    )
    negative_logits = negative_logits.masked_fill(
        ~neg_valid.unsqueeze(0), -torch.inf
    )

    # Multi-positive InfoNCE, consistent with the original SCC.
    positive_logsumexp = torch.logsumexp(
        positive_logits, dim=-1
    )

    denominator_logsumexp = torch.logsumexp(
        torch.cat([positive_logits, negative_logits], dim=-1),
        dim=-1,
    )

    loss = -(
        positive_logsumexp[:, valid_anchor]
        - denominator_logsumexp[:, valid_anchor]
    ).mean()

    return loss