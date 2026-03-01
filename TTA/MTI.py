import time
import numpy as np
from copy import deepcopy
from torch.functional import F
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast, GradScaler
import math

def quantile_rank(x, descending=False):
    if x.numel() == 0:
        return x

    n = x.size(0)
    sorted_x, sorted_indices = torch.sort(x, descending=descending)
    ranks = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
    unique_vals, inverse_indices = torch.unique(sorted_x, return_inverse=True, sorted=True)
    rank_sums = torch.zeros(unique_vals.size(0), device=x.device, dtype=torch.float32)
    rank_sums.scatter_add_(0, inverse_indices, ranks)
    counts = torch.bincount(inverse_indices).float()
    avg_ranks = rank_sums / counts
    adjusted_ranks = avg_ranks[inverse_indices]
    final_ranks = torch.empty_like(x, dtype=torch.float32)
    final_ranks[sorted_indices] = adjusted_ranks
    normalized_ranks = final_ranks / n

    return normalized_ranks

def SoftCELoss(outputs, targets, reduction=None):
    Lx = -torch.sum(torch.log_softmax(outputs, dim=-1) * torch.softmax(targets, dim=-1), dim=1)
    return Lx


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class MTI(nn.Module):
    def __init__(self, model, optimizer, device, args):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = 1
        self.args = args
        self.scaler = GradScaler()
        self.device = device

        self.source_model = deepcopy(model)
        self.source_model.eval()
        for p in self.source_model.parameters():
            p.requires_grad = False

    def forward(self, x, adapt_flag, ):
        for step in range(self.steps):
            if adapt_flag:
                outputs, loss = forward_and_adapt_TSA(x, self.model, self.optimizer, self.args, self.scaler, self.source_model)
            else:
                outputs, attn, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode, TSA=self.args.TSA)
                loss = (0, 0)
                outputs = (outputs, outputs)
                return outputs, loss, attn
        return outputs, loss


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_TSA(x, model, optimizer, args, scaler, source_model):
    """Forward and adapt model on batch of data.
    Compute loss function (Eq. 7) based on the model prediction, take gradients, and update params.
    """
    GAP_THRESHOLD = 0.01
    TOP_K = 10

    bs = x[0].shape[0]

    with autocast():
        outputs, _, router_weights, feat_a, feat_v = model.module.forward_adapt(a=x[0], v=x[1], mode=args.testmode, TSA=True)
        p_sum = outputs.softmax(dim=-1).sum(dim=-2)

        pred = outputs.softmax(dim=-1)
        pred_max = pred.max(dim=-1)[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probs = outputs.softmax(dim=1)
        conf_values, pred_labels = probs.max(dim=1)

        ent_standard = softmax_entropy(outputs)
        ent_topk = top_k_entropy(outputs, k=TOP_K)
        ent_tail = tail_entropy(outputs, k=TOP_K)
        batch_size = outputs.size(0)

        unique_classes, counts = torch.unique(pred_labels, return_counts=True)
        class_freq_dict = {cls.item(): count.item() / batch_size for cls, count in zip(unique_classes, counts)}

        z_values = torch.tensor([class_freq_dict[label.item()] for label in pred_labels], device=device)
        z_bar = z_values.mean()
        bias_levels = z_bar - z_values
        positive_mask = bias_levels >= 0
        negative_mask = bias_levels < 0

        top2_probs, _ = probs.topk(2, dim=1)
        prob_gap = top2_probs[:, 0] - top2_probs[:, 1]
        high_conf_mask = prob_gap > GAP_THRESHOLD
        low_conf_mask = ~high_conf_mask

        final_entropy_term = torch.zeros(batch_size, device=device)
        weights = torch.zeros(batch_size, device=device)

        pos_indices = torch.where(positive_mask)[0]
        if len(pos_indices) > 0:
            q_z = quantile_rank(bias_levels[pos_indices])
            q_k = quantile_rank(conf_values[pos_indices])
            weights[pos_indices] = q_z * q_k

            pos_high_mask = positive_mask & high_conf_mask
            pos_low_mask = positive_mask & low_conf_mask

            if pos_high_mask.any():
                final_entropy_term[pos_high_mask] = ent_standard[pos_high_mask]

            if pos_low_mask.any():
                final_entropy_term[pos_low_mask] = ent_topk[pos_low_mask]
        neg_indices = torch.where(negative_mask)[0]
        if len(neg_indices) > 0:
            weights[neg_indices] = -0.5

            neg_high_mask = negative_mask & high_conf_mask
            neg_low_mask = negative_mask & low_conf_mask

            if neg_high_mask.any():
                final_entropy_term[neg_high_mask] = ent_standard[neg_high_mask]

            if neg_low_mask.any():
                final_entropy_term[neg_low_mask] = ent_tail[neg_low_mask]

        loss_ent = (weights * final_entropy_term).mean()
        loss = loss_ent

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        with autocast():
            outputs2, _, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode, TSA=True, Traverse=True)
            outputs2 = outputs2.chunk(2)
            outputs2 = outputs2[0] + outputs2[1]

    return (outputs, outputs2), (loss.item(), loss.item())


def top_k_entropy(logits: torch.Tensor, k: int = 5) -> torch.Tensor:
    topk_logits, _ = logits.topk(k, dim=1)
    topk_probs = topk_logits.softmax(dim=1)
    return -(topk_probs * torch.log(topk_probs + 1e-8)).sum(dim=1)


def tail_entropy(logits: torch.Tensor, k: int = 5) -> torch.Tensor:
    num_classes = logits.size(1)
    if k >= num_classes:
        return torch.zeros(logits.size(0), device=logits.device)

    _, topk_indices = logits.topk(k, dim=1)

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)

    tail_logits = logits.clone()
    tail_logits[mask] = float('-inf')

    tail_probs = tail_logits.softmax(dim=1)
    return -(tail_probs * torch.log(tail_probs + 1e-8)).sum(dim=1)

def collect_params(model, args):
    extra_params = []
    extra_params.append(model.module.gate_a_adaptor)
    extra_params.append(model.module.gate_v_adaptor)
    extra_params.append(model.module.a_adaptor)
    extra_params.append(model.module.v_adaptor)
    for p in model.module.v_router.parameters():
        p.requires_grad = True
        extra_params.append(p)
    for p in model.module.a_router.parameters():
        p.requires_grad = True
        extra_params.append(p)
    return extra_params



def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, p in model.named_parameters():
        if 'adaptor' in nm or 'router' in nm or 'gate' in nm:
            p.requires_grad_(True)
    return model