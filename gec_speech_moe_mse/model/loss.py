"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

zero_tensor=0.0

# def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0,args=None):
#     compute = NMTLossCompute(
#         generator, symbols, vocab_size,
#         label_smoothing=label_smoothing if train else 0.0, args=args)
#     compute.to(device)
#     return compute


def contrastiveLoss(emb_i, emb_j, device='cuda',temperature=0.5):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到

    batch_size = emb_i.shape[0]
    negatives_mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float()
    temperature = torch.tensor(temperature).to(device)

    z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
    z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

    representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

    sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
    sim_ji = torch.diag(similarity_matrix, batch_size)  # bs
    positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

    nominator = torch.exp(positives / temperature)  # 2*bs
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)  # 2*bs, 2*bs

    # import pdb
    # pdb.set_trace()


    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
    loss = torch.sum(loss_partial) / (2 * batch_size)
    return loss

@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, logits, labels):
        # logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)


        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


@dataclass
class LabelSmoother_wo_log_softmax:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, logits, labels):
        # logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        # log_probs = -nn.functional.log_softmax(logits, dim=-1)
        log_probs = -logits
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
