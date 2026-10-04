"""One reusable training pipeline for every DeepWeeds experiment."""
from __future__ import annotations
import argparse, copy, json, math, random, sys, time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import dataset, losses, model as model_utils
import benchmark

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"; seed: int = 0; fold: int = 0
    backbone: str = "resnet50"; init: str = "finetune"; drop_rate: float = 0.0
    img_size: int = 224; aug: str = "basic"; sampler: str | None = None
    mix: str | None = None; mix_alpha: float = 1.0
    loss: str = "ce"; label_smoothing: float = 0.0; focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12; batch_size: int = 64; lr_backbone: float = 1e-4
    lr_head: float = 1e-3; weight_decay: float = 0.05; warmup_epochs: float = 1.0
    ema_decay: float | None = None; amp: bool = True; num_workers: int = 2
    grad_accum_steps: int = 1
    images_dir: str = "data/images"; labels_dir: str = "data/labels"
    out_dir: str = "runs"; pred_dir: str = "predictions"; curves_dir: str = "curves"
    save_test_predictions: bool = False
    debug_train_samples: int | None = None; debug_val_samples: int | None = None
    measure_latency: bool = False; latency_dtype: str = "amp"
    latency_warmup: int = 10; latency_iters: int = 30
    resume_completed: bool = True
    curve_label: str | None = None


def run_dir(cfg): return Path(cfg.out_dir)/cfg.exp_id/f"seed{cfg.seed}"
def pred_path(cfg, split): return Path(cfg.pred_dir)/f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def build_optimizer(net, cfg):
    return torch.optim.AdamW(model_utils.param_groups(net,cfg.lr_backbone,cfg.lr_head,cfg.weight_decay))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    total=max(1,cfg.epochs*steps_per_epoch); warm=max(0,int(cfg.warmup_epochs*steps_per_epoch))
    def factor(step):
        if warm and step < warm: return (step+1)/warm
        progress=(step-warm)/max(1,total-warm)
        return 0.5*(1+math.cos(math.pi*min(1,max(0,progress))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


class EMA:
    def __init__(self, net, decay):
        if not 0 < decay < 1: raise ValueError("EMA decay phải thuộc (0,1)")
        self.decay=decay; self.model=copy.deepcopy(net).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, net):
        source=net.state_dict()
        for name,value in self.model.state_dict().items():
            current=source[name].detach()
            if value.is_floating_point(): value.mul_(self.decay).add_(current,alpha=1-self.decay)
            else: value.copy_(current)
    @torch.no_grad()
    def copy_to(self, net): net.load_state_dict(self.model.state_dict())


def train_one_epoch(net, loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=None):
    net.train()
    if cfg.init == "frozen": model_utils.keep_frozen_bn_eval(net)
    total=items=0; start=time.perf_counter(); amp=cfg.amp and device.type=="cuda"
    optimizer.zero_grad(set_to_none=True)
    for step,(images,labels,_) in enumerate(loader):
        images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
        targets=labels
        if cfg.mix: images,targets=losses.mix_batch(images,labels,cfg.mix_alpha,cfg.mix)
        with torch.autocast(device_type=device.type,enabled=amp):
            logits=net(images); loss=losses.mixed_loss(criterion,logits,targets) if cfg.mix else criterion(logits,labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss train không hữu hạn; kiểm tra AMP, LR và batch")
        scaler.scale(loss/cfg.grad_accum_steps).backward()
        if (step+1)%cfg.grad_accum_steps==0 or step+1==len(loader):
            old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            step_succeeded=scaler.get_scale() >= old_scale
            optimizer.zero_grad(set_to_none=True)
            if step_succeeded:
                scheduler.step()
                if ema: ema.update(net)
        total += float(loss.detach())*len(images); items += len(images)
    return {"train_loss":total/max(1,items),"lr":max(g["lr"] for g in optimizer.param_groups),
            "train_seconds":time.perf_counter()-start}


def evaluate(net, loader, criterion, device):
    net.eval(); names=[]; targets=[]; outputs=[]; total=items=0
    with torch.inference_mode():
        for images,labels,batch_names in loader:
            images=images.to(device,non_blocking=True); labels_dev=labels.to(device,non_blocking=True)
            logits=net(images); loss=criterion(logits,labels_dev)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Logit/loss validation không hữu hạn; kiểm tra AMP, LR và dữ liệu")
            names.extend(batch_names); targets.append(labels.numpy()); outputs.append(logits.cpu().numpy())
            total += float(loss)*len(images); items += len(images)
    return names,np.concatenate(targets),np.concatenate(outputs),total/items


def plot_curves(history, path, title):
    import matplotlib.pyplot as plt
    df=pd.DataFrame(history); fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(df.epoch,df.train_loss,label="train"); axes[0].plot(df.epoch,df.val_loss,label="val")
    axes[0].set(xlabel="Epoch",ylabel="Loss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(df.epoch,df.val_macro_f1,label="macro-F1"); axes[1].plot(df.epoch,df.val_top1,label="top-1")
    axes[1].set(xlabel="Epoch",ylabel="Score",ylim=(0,1)); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle(title); fig.tight_layout(); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def _criterion(cfg, train_df, device):
    weight=None
    if cfg.loss=="ce_weighted":
        counts=train_df.Label.value_counts().reindex(range(9),fill_value=0).to_numpy()
        weight=losses.class_weights(counts,0 if cfg.class_weight_beta is None else cfg.class_weight_beta).to(device)
    return losses.build_criterion(cfg.loss,smoothing=cfg.label_smoothing,gamma=cfg.focal_gamma,weight=weight,alpha=weight)


def _probs(logits): return F.softmax(torch.from_numpy(logits),dim=1).numpy()


def run(cfg):
    if cfg.grad_accum_steps < 1: raise ValueError("grad_accum_steps phải >= 1")
    set_seed(cfg.seed); out=run_dir(cfg); out.mkdir(parents=True,exist_ok=True)
    summary_path=out/"summary.json"; config_path=out/"config.json"
    if cfg.resume_completed and summary_path.is_file() and config_path.is_file() and pred_path(cfg,"val").is_file():
        saved=json.loads(config_path.read_text(encoding="utf-8"))
        if saved==asdict(cfg):
            summary=json.loads(summary_path.read_text(encoding="utf-8")); summary["skipped_completed"]=True
            print(f"{cfg.exp_id}: đã hoàn tất, bỏ qua huấn luyện lại")
            return summary
    config_path.write_text(json.dumps(asdict(cfg),indent=2),encoding="utf-8")
    train_df,val_df,test_df=dataset.load_split(cfg.labels_dir,cfg.fold)
    report=dataset.check_split(train_df,val_df,test_df,cfg.images_dir)
    (out/"split_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    # Optional smoke subsets are applied only after validating the complete official split.
    if cfg.debug_train_samples is not None: train_df=train_df.head(cfg.debug_train_samples).copy()
    if cfg.debug_val_samples is not None: val_df=val_df.head(cfg.debug_val_samples).copy()
    train_loader=dataset.make_loader(train_df,cfg.images_dir,dataset.build_transforms(True,cfg.img_size,cfg.aug),cfg.batch_size,True,cfg.sampler,cfg.num_workers)
    val_loader=dataset.make_loader(val_df,cfg.images_dir,dataset.build_transforms(False,cfg.img_size),cfg.batch_size,False,None,cfg.num_workers)
    test_loader=dataset.make_loader(test_df,cfg.images_dir,dataset.build_transforms(False,cfg.img_size),cfg.batch_size,False,None,cfg.num_workers) if cfg.save_test_predictions else None
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=model_utils.build_model(cfg.backbone,init=cfg.init,drop_rate=cfg.drop_rate).to(device)
    criterion=_criterion(cfg,train_df,device); optimizer=build_optimizer(net,cfg)
    updates_per_epoch=math.ceil(len(train_loader)/cfg.grad_accum_steps)
    scheduler=build_scheduler(optimizer,cfg,updates_per_epoch)
    scaler=torch.amp.GradScaler("cuda",enabled=cfg.amp and device.type=="cuda")
    ema=EMA(net,cfg.ema_decay) if cfg.ema_decay else None
    checkpoint=out/"best.pt"; history=[]; best=-float("inf"); best_epoch=-1
    for epoch in range(1,cfg.epochs+1):
        row={"epoch":epoch}; row.update(train_one_epoch(net,train_loader,criterion,optimizer,scheduler,scaler,cfg,device,ema))
        evaluated=ema.model if ema else net
        _,y,logits,val_loss=evaluate(evaluated,val_loader,criterion,device); probs=_probs(logits)
        metrics=compute_metrics(y,probs.argmax(1),probs)
        row.update(val_loss=val_loss,val_macro_f1=metrics["macro_f1"],val_top1=metrics["top1"]); history.append(row)
        pd.DataFrame(history).to_csv(out/"history.csv",index=False)
        if metrics["macro_f1"] > best:
            best=metrics["macro_f1"]; best_epoch=epoch
            torch.save({"model":evaluated.state_dict(),"epoch":epoch,"val_macro_f1":best},checkpoint)
    state=torch.load(checkpoint,map_location=device,weights_only=True); net.load_state_dict(state["model"])
    names,y,logits,val_loss=evaluate(net,val_loader,criterion,device); probs=_probs(logits)
    save_predictions(pred_path(cfg,"val"),names,y,probs); np.save(out/"val_logits.npy",logits)
    metrics=compute_metrics(y,probs.argmax(1),probs)
    if test_loader is not None:
        names,yt,lt,_=evaluate(net,test_loader,criterion,device)
        save_predictions(pred_path(cfg,"test"),names,yt,_probs(lt)); np.save(out/"test_logits.npy",lt)
    architecture=cfg.backbone.split(".")[0]
    curve_label=cfg.curve_label or architecture
    curve_path=Path(cfg.curves_dir)/f"{cfg.exp_id}_{curve_label}.png"
    plot_curves(history,curve_path,f"{cfg.exp_id} — {cfg.backbone}")
    gmac=model_utils.count_gmacs(net,cfg.img_size)
    latency={}
    if cfg.measure_latency:
        latency=benchmark.latency_report(net,1,cfg.img_size,cfg.latency_dtype,str(device),cfg.latency_warmup,cfg.latency_iters)
    summary={"best_epoch":best_epoch,"val_macro_f1":metrics["macro_f1"],"val_top1":metrics["top1"],
             "val_loss":val_loss,"params_m":model_utils.count_params(net),
             "val_f1_per_class":metrics["f1"].tolist(),"val_recall_per_class":metrics["recall"].tolist(),
             "gmac":gmac,"mean_train_seconds":float(np.mean([r["train_seconds"] for r in history])),
             "exp_id":cfg.exp_id,"backbone":architecture,"pretrained_tag":cfg.backbone,
             "img_size":cfg.img_size,"epochs":cfg.epochs,"seed":cfg.seed,
             "physical_batch":cfg.batch_size,"effective_batch":cfg.batch_size*cfg.grad_accum_steps,
             "curve":str(curve_path),"history":str(out/"history.csv"),"latency":latency}
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8"); return summary


def _coerce(value,default):
    low=value.lower()
    if low in {"none","null"}: return None
    if isinstance(default,bool): return low in {"true","1","yes"}
    if isinstance(default,int): return int(value)
    if isinstance(default,float): return float(value)
    return value


def parse_overrides(pairs):
    defaults=Config(); valid={f.name for f in fields(Config)}; result={}
    for pair in pairs:
        if "=" not in pair: raise ValueError(f"Cần KEY=VALUE: {pair}")
        key,value=pair.split("=",1)
        if key not in valid: raise ValueError(f"Không có Config.{key}")
        if key in {"ema_decay","class_weight_beta"} and value.lower() not in {"none","null"}: result[key]=float(value)
        elif key in {"debug_train_samples","debug_val_samples"} and value.lower() not in {"none","null"}: result[key]=int(value)
        else: result[key]=_coerce(value,getattr(defaults,key))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--set",nargs="*",default=[])
    print(json.dumps(run(Config(**parse_overrides(parser.parse_args().set))),indent=2))


if __name__=="__main__": main()
