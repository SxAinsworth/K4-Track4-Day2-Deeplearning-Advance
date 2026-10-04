"""timm model creation, freezing, parameter groups and complexity."""
from __future__ import annotations
import torch
from torch import nn

SUGGESTED_BACKBONES = {"resnet50":"resnet50", "resnext50":"resnext50_32x4d",
 "convnext_tiny":"convnext_tiny", "deit_small":"deit_small_patch16_224",
 "swin_tiny":"swin_tiny_patch4_window7_224", "efficientnet_b0":"efficientnet_b0",
 "mobilenetv3":"mobilenetv3_large_100"}


def _head_ids(model):
    return {id(p) for p in model.get_classifier().parameters()}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    import timm
    if init not in {"scratch", "frozen", "finetune"}: raise ValueError("init không hợp lệ")
    net = timm.create_model(name, pretrained=False if init == "scratch" else pretrained,
                            num_classes=num_classes, drop_rate=drop_rate)
    cfg = dict(getattr(net, "pretrained_cfg", {}) or {})
    net.pretrained_tag = cfg.get("tag") or cfg.get("url") or cfg.get("hf_hub_id")
    init_head(net)
    if init == "frozen": freeze_backbone(net)
    return net


def init_head(model, std: float = 0.01) -> None:
    """Same small init for every backbone's new head so the initial CE is close to ln(9).

    timm's default Linear init on 1280-d features (EfficientNet, MobileNetV3) gives logit std
    near 3 and an initial loss of 4.7-5.9 instead of 2.197.
    """
    for module in model.get_classifier().modules():
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=std)
            if module.bias is not None: nn.init.zeros_(module.bias)


def freeze_backbone(model) -> None:
    head_ids = _head_ids(model)
    for parameter in model.parameters(): parameter.requires_grad = id(parameter) in head_ids
    frozen_bn = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
                p.requires_grad for p in module.parameters(recurse=False)):
            module.eval(); frozen_bn.append(module)
    model._frozen_bn_modules = frozen_bn


def keep_frozen_bn_eval(model) -> None:
    for module in getattr(model, "_frozen_bn_modules", []): module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_ids = _head_ids(model); decay, no_decay, head = [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad: continue
        if id(parameter) in head_ids: head.append(parameter)
        elif parameter.ndim <= 1: no_decay.append(parameter)
        else: decay.append(parameter)
    groups = []
    if decay: groups.append({"params":decay, "lr":lr_backbone, "weight_decay":weight_decay})
    if no_decay: groups.append({"params":no_decay, "lr":lr_backbone, "weight_decay":0.0})
    if head: groups.append({"params":head, "lr":lr_head, "weight_decay":weight_decay})
    if not groups: raise ValueError("Không có tham số trainable")
    return groups


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Cài thop để đếm GMAC") from exc
    device = next(model.parameters()).device; was_training = model.training; model.eval()
    with torch.inference_mode():
        macs, _ = profile(model, inputs=(torch.zeros(1,3,img_size,img_size,device=device),), verbose=False)
    model.train(was_training)
    return float(macs / 1e9)
