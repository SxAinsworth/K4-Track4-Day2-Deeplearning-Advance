"""Reproducible inference latency measurement."""
from __future__ import annotations
import time
import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    if warmup < 0 or iters < 1: raise ValueError("warmup >= 0 và iters >= 1")
    for _ in range(warmup): fn()
    if sync: sync()
    samples=[]
    for _ in range(iters):
        if sync: sync()
        start=time.perf_counter(); fn()
        if sync: sync()
        samples.append((time.perf_counter()-start)*1000)
    values=np.asarray(samples,dtype=np.float64)
    return {"p50":float(np.percentile(values,50)),"p95":float(np.percentile(values,95)),
            "p99":float(np.percentile(values,99)),"mean":float(values.mean()),"n":iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    if dtype not in {"fp32","amp","fp16"}: raise ValueError("dtype: fp32 | amp | fp16")
    target=torch.device(device)
    if target.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA không khả dụng")
    model=model.to(target).eval(); x=torch.randn(batch_size,3,img_size,img_size,device=target)
    if dtype=="fp16": model=model.half(); x=x.half()
    amp_enabled=dtype=="amp" and target.type=="cuda"
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=target.type,enabled=amp_enabled): model(x)
    sync=torch.cuda.synchronize if target.type=="cuda" else None
    measured=bench(forward,warmup,iters,sync)
    gpu=torch.cuda.get_device_name(target) if target.type=="cuda" else "CPU"
    return {"gpu":gpu,"dtype":dtype,"batch":batch_size,"img_size":img_size,**measured,
            "images_per_s":batch_size/(measured["p50"]/1000),"torch":torch.__version__,
            "includes_preprocessing":False}


def tta_latency(model, k_views: int, **kw) -> dict:
    if k_views < 1: raise ValueError("k_views phải >= 1")
    batch_size=kw.pop("batch_size"); img_size=kw.pop("img_size")
    device=torch.device(kw.pop("device","cuda")); dtype=kw.pop("dtype","fp32")
    warmup=kw.pop("warmup",10); iters=kw.pop("iters",100)
    if kw: raise TypeError(f"Tham số không hỗ trợ: {sorted(kw)}")
    model=model.to(device).eval(); x=torch.randn(batch_size,3,img_size,img_size,device=device)
    if dtype=="fp16": model=model.half(); x=x.half()
    amp=dtype=="amp" and device.type=="cuda"
    def forward_views():
        with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=amp):
            for _ in range(k_views): model(x)
    sync=torch.cuda.synchronize if device.type=="cuda" else None
    measured=bench(forward_views,warmup,iters,sync)
    return {"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
            "dtype":dtype,"batch":batch_size,"img_size":img_size,"k_views":k_views,**measured,
            "images_per_s":batch_size/(measured["p50"]/1000),"torch":torch.__version__,
            "includes_preprocessing":False}
