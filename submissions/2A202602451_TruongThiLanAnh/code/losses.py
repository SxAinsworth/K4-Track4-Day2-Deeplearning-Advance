"""Loss functions and Mixup/CutMix."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    if kind == "ce": return nn.CrossEntropyLoss(weight=kw.get("weight"))
    if kind == "ls": return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal": return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None: raise ValueError("ce_weighted cần weight")
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"Loss không hỗ trợ: {kind}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1: raise ValueError("smoothing phải thuộc [0,1)")
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0: raise ValueError("gamma phải không âm")
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - log_pt.exp()) ** self.gamma) * log_pt
        if self.alpha is not None: loss = loss * self.alpha.to(logits)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or len(values) != 9 or torch.any(values <= 0):
        raise ValueError("counts phải gồm 9 số dương từ train")
    if beta == 0: weights = values.reciprocal()
    elif 0 < beta < 1: weights = (1-beta) / (1-torch.pow(torch.tensor(beta), values))
    else: raise ValueError("beta phải bằng 0 hoặc thuộc (0,1)")
    return (weights * len(values) / weights.sum()).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha <= 0 or mode not in {"mixup", "cutmix"}: raise ValueError("alpha/mode không hợp lệ")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(x), device=x.device)
    if mode == "mixup": return lam*x + (1-lam)*x[perm], (y, y[perm], lam)
    _, _, h, w = x.shape
    ratio = math.sqrt(1-lam); cw, ch = int(w*ratio), int(h*ratio)
    cx, cy = int(np.random.randint(w)), int(np.random.randint(h))
    x1, x2 = max(0, cx-cw//2), min(w, cx+cw//2)
    y1, y2 = max(0, cy-ch//2), min(h, cy+ch//2)
    mixed = x.clone(); mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1 - (x2-x1)*(y2-y1)/float(w*h)
    return mixed, (y, y[perm], float(lam))


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam*criterion(logits, y_a) + (1-lam)*criterion(logits, y_b)
