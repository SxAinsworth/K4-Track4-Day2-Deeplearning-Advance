"""Inference views, aggregation, calibration, ensembling and BN fusion."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None, amp: bool = False):
    model.eval(); names=[]; targets=[]; outputs=[]; target=torch.device(device)
    with torch.inference_mode():
        for images,labels,batch_names in loader:
            images=images.to(target,non_blocking=True)
            if view is not None: images=view(images)
            with torch.autocast(device_type=target.type,enabled=amp and target.type=="cuda"):
                logits=model(images)
            names.extend(batch_names); targets.append(labels.numpy()); outputs.append(logits.float().cpu().numpy())
    return names,np.concatenate(targets),np.concatenate(outputs)


def view_identity(x): return x


def view_hflip(x): return torch.flip(x,dims=(-1,))


def views_multicrop(x, crop: int):
    if x.ndim!=4: raise ValueError("x phải có dạng NCHW")
    height,width=x.shape[-2:]
    if crop>height or crop>width: raise ValueError("crop lớn hơn ảnh")
    top,bottom,left,right=0,height-crop,0,width-crop
    center_y,center_x=(height-crop)//2,(width-crop)//2
    return [x[...,top:top+crop,left:left+crop],x[...,top:top+crop,right:right+crop],
            x[...,bottom:bottom+crop,left:left+crop],x[...,bottom:bottom+crop,right:right+crop],
            x[...,center_y:center_y+crop,center_x:center_x+crop]]


def views_multiscale(x, sizes):
    return [F.interpolate(x,size=(int(size),int(size)),mode="bilinear",align_corners=False,antialias=True) for size in sizes]


def _numpy(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def aggregate_views(logits_per_view, space: str = "prob"):
    if not logits_per_view: raise ValueError("Cần ít nhất một view")
    stack=np.stack([_numpy(value) for value in logits_per_view])
    if space=="prob":
        shifted=stack-stack.max(axis=2,keepdims=True); probs=np.exp(shifted); probs/=probs.sum(axis=2,keepdims=True)
        return probs.mean(axis=0)
    if space=="logit":
        mean=stack.mean(axis=0); mean-=mean.max(axis=1,keepdims=True); probs=np.exp(mean); return probs/probs.sum(axis=1,keepdims=True)
    raise ValueError("space phải là prob hoặc logit")


def ensemble_probs(list_of_probs):
    if not list_of_probs: raise ValueError("Cần ít nhất một mô hình")
    arrays=[_numpy(value) for value in list_of_probs]
    if len({array.shape for array in arrays})!=1: raise ValueError("Các prediction phải cùng shape/thứ tự")
    result=np.mean(np.stack(arrays),axis=0)
    return result/result.sum(axis=1,keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    logits=torch.as_tensor(val_logits,dtype=torch.float64)
    labels=torch.as_tensor(val_labels,dtype=torch.long)
    log_temperature=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_temperature],lr=0.1,max_iter=100,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(); temperature=log_temperature.exp().clamp(0.05,20.0)
        loss=F.cross_entropy(logits/temperature,labels); loss.backward(); return loss
    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05,20.0))


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T<=0: raise ValueError("T phải hữu hạn và dương")
    values=torch.as_tensor(logits,dtype=torch.float64)
    return F.softmax(values/T,dim=1).cpu().numpy()


def fuse_conv_bn(model):
    """Return a deep-copied eval model with adjacent Conv2d+BatchNorm2d fused."""
    fused=copy.deepcopy(model).eval()
    def recurse(parent):
        for child in parent.children(): recurse(child)
        children=list(parent.named_children())
        for (conv_name,conv),(bn_name,bn) in zip(children,children[1:]):
            if isinstance(conv,nn.Conv2d) and isinstance(bn,nn.BatchNorm2d):
                setattr(parent,conv_name,torch.nn.utils.fusion.fuse_conv_bn_eval(conv,bn))
                setattr(parent,bn_name,nn.Identity())
    recurse(fused)
    return fused
