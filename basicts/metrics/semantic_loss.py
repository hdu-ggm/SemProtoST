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

    # B, N, K = attn.shape
    #
    # # -------- normalize --------
    # meta = F.normalize(meta_node, dim=-1)
    # attn = F.normalize(attn, dim=-1)
    #
    # # -------- similarity --------
    # sim_meta = torch.matmul(meta, meta.transpose(1, 2))  # [B, N, N]
    # sim_attn = torch.matmul(attn, attn.transpose(1, 2))  # [B, N, N]
    #
    # # -------- remove self --------
    # eye = torch.eye(N, device=attn.device).unsqueeze(0)
    # sim_meta = sim_meta * (1 - eye)
    # sim_attn = sim_attn * (1 - eye)
    #
    # # -------- define masks --------
    # pos_mask = (sim_meta > 0.7).float()
    # neg_mask = (sim_meta < 0.3).float()
    #
    # # ❗关键：去掉“假负样本”（高相似的不要当负）
    # neg_mask = neg_mask * (sim_meta < 0.9).float()
    #
    # # -------- logits --------
    # logits = sim_attn / temperature
    #
    # # -------- InfoNCE --------
    # exp_logits = torch.exp(logits)
    #
    # # 正样本
    # pos_exp = exp_logits * pos_mask
    #
    # # 负样本
    # neg_exp = exp_logits * neg_mask
    #
    # # denominator（只包含有效对比项）
    # denom = pos_exp.sum(dim=-1) + neg_exp.sum(dim=-1) + 1e-8
    #
    # # loss
    # loss = -torch.log((pos_exp.sum(dim=-1) + 1e-8) / denom)
    #
    # # 只对有正样本的节点计算
    # valid = (pos_mask.sum(dim=-1) > 0).float()
    # loss = (loss * valid).sum() / (valid.sum() + 1e-6)
    #
    # return loss