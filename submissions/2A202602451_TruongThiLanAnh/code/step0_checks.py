"""Generate EDA artifacts and run inexpensive pipeline checks."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
import dataset, losses, model
from train import set_seed


def run_eda(labels_dir,images_dir,out_dir="eda"):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    train,val,test=dataset.load_split(labels_dir,0); report=dataset.check_split(train,val,test,images_dir)
    full=pd.concat([train,val,test],ignore_index=True)
    counts=full.Label.value_counts().reindex(range(9),fill_value=0)
    report["full_per_class"]={str(k):int(v) for k,v in counts.items()}
    report["imbalance_ratio"]=float(counts.max()/counts.min())
    fig,ax=plt.subplots(figsize=(11,5)); ax.bar(dataset.CLASS_NAMES,counts.to_numpy())
    ax.set(title="DeepWeeds — full class distribution",ylabel="Images"); ax.tick_params(axis="x",rotation=40)
    fig.tight_layout(); fig.savefig(out/"class_distribution.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(9,3,figsize=(10,25)); sizes=set(); channels=set()
    for label in range(9):
        rows=train.loc[train.Label==label].head(3)
        for col,(_,row) in enumerate(rows.iterrows()):
            with Image.open(Path(images_dir)/str(row.Filename)) as image:
                sizes.add(image.size); channels.add(len(image.getbands())); axes[label,col].imshow(image.convert("RGB"))
            axes[label,col].set_title(f"{dataset.CLASS_NAMES[label]}\n{row.Filename}",fontsize=8); axes[label,col].axis("off")
    fig.tight_layout(); fig.savefig(out/"samples_3_per_class.png",dpi=160); plt.close(fig)
    # Visual check after stochastic augmentation, de-normalized for display.
    transform=dataset.build_transforms(True,224,"basic")
    fig,axes=plt.subplots(3,3,figsize=(10,10))
    mean=torch.tensor(dataset.IMAGENET_MEAN)[:,None,None]; std=torch.tensor(dataset.IMAGENET_STD)[:,None,None]
    for label,ax in enumerate(axes.flat):
        row=train.loc[train.Label==label].iloc[0]
        with Image.open(Path(images_dir)/str(row.Filename)) as image: tensor=transform(image.convert("RGB"))
        shown=(tensor*std+mean).clamp(0,1).permute(1,2,0).numpy()
        ax.imshow(shown); ax.set_title(f"{dataset.CLASS_NAMES[label]} — label {label}",fontsize=8); ax.axis("off")
    fig.tight_layout(); fig.savefig(out/"augmented_samples.png",dpi=160); plt.close(fig)
    report["sample_sizes"]=[list(x) for x in sorted(sizes)]; report["sample_channels"]=sorted(channels)
    (out/"eda_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return report


def math_checks():
    logits=torch.zeros(32,9); labels=torch.arange(32)%9
    ce=float(F.cross_entropy(logits,labels)); err=float(abs(losses.FocalLoss(0)(logits,labels)-F.cross_entropy(logits,labels)))
    x=torch.arange(8*3*16*16,dtype=torch.float32).reshape(8,3,16,16)
    mixed,(a,b,lam)=losses.mix_batch(x,torch.arange(8),1.0,"cutmix")
    result={"uniform_ce":ce,"expected_ln9":math.log(9),"focal_gamma0_abs_error":err,
            "cutmix_lambda":lam,"cutmix_pixels_changed":not torch.equal(mixed,x),"cutmix_labels_permuted":not torch.equal(a,b)}
    if abs(ce-math.log(9))>1e-6 or err>=1e-6: raise AssertionError(result)
    return result


def initial_loss_check(backbone,labels_dir,images_dir,n=64,img_size=224,seed=0):
    """CE of the real model (pretrained backbone + new 9-class head) on real val images, before any update."""
    set_seed(seed); _,val,_=dataset.load_split(labels_dir,0)
    loader=dataset.make_loader(val.sample(n,random_state=seed),images_dir,dataset.build_transforms(False,img_size),n,False,None,0)
    images,labels,_=next(iter(loader)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=model.build_model(backbone).to(device).eval()
    with torch.inference_mode(): logits=net(images.to(device)).float().cpu()
    loss=float(F.cross_entropy(logits,labels))
    return {"backbone":backbone,"tag":net.pretrained_tag,"initial_loss":loss,"expected_ln9":math.log(9),
            "logit_std":float(logits.std()),"n_images":n}


def mode_checks():
    """train_one_epoch puts the net in train mode (frozen BN stays eval); evaluate puts it in eval mode."""
    import train as train_mod
    from torch.utils.data import DataLoader, TensorDataset
    class Named(TensorDataset):
        def __getitem__(self,i): x,y=super().__getitem__(i); return x,int(y),f"img{i}.jpg"
    loader=DataLoader(Named(torch.randn(8,3,64,64),torch.arange(8)%9),batch_size=4)
    seen={}
    def record(name): return lambda m,_i,_o: seen.__setitem__(name,m.training)
    result={}
    for init in ("finetune","frozen"):
        net=model.build_model("resnet18",pretrained=False,init=init); bn=net.bn1; head=net.get_classifier()
        bn.register_forward_hook(record("bn")); head.register_forward_hook(record("head"))
        cfg=train_mod.Config(init=init,amp=False,epochs=1); dev=torch.device("cpu")
        opt=train_mod.build_optimizer(net,cfg); sched=train_mod.build_scheduler(opt,cfg,len(loader))
        train_mod.train_one_epoch(net,loader,torch.nn.CrossEntropyLoss(),opt,sched,torch.amp.GradScaler("cuda",enabled=False),cfg,dev)
        result[f"{init}_train"]={"bn_training":seen["bn"],"head_training":seen["head"]}
        train_mod.evaluate(net,loader,torch.nn.CrossEntropyLoss(),dev)
        result[f"{init}_eval"]={"bn_training":seen["bn"],"head_training":seen["head"]}
    expected={"finetune_train":{"bn_training":True,"head_training":True},"finetune_eval":{"bn_training":False,"head_training":False},
              "frozen_train":{"bn_training":False,"head_training":True},"frozen_eval":{"bn_training":False,"head_training":False}}
    if result!=expected: raise AssertionError(result)
    return result


def overfit_small_batch(backbone,labels_dir,images_dir,img_size=128,steps=100,seed=0):
    set_seed(seed); train,_,_=dataset.load_split(labels_dir,0)
    loader=dataset.make_loader(train.head(16),images_dir,dataset.build_transforms(False,img_size),16,False,None,0)
    images,labels,_=next(iter(loader)); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=model.build_model(backbone,pretrained=False,init="scratch").to(device); images,labels=images.to(device),labels.to(device)
    opt=torch.optim.Adam(net.parameters(),lr=3e-3); seen=[]; net.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); loss=F.cross_entropy(net(images),labels); loss.backward(); opt.step(); seen.append(float(loss.detach()))
    return {"initial_loss":seen[0],"final_loss":seen[-1],"steps":steps}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--labels-dir",default="data/labels"); p.add_argument("--images-dir",default="data/images")
    p.add_argument("--out-dir",default="submissions/2A202602451_TruongThiLanAnh/eda"); p.add_argument("--overfit",action="store_true")
    p.add_argument("--backbone",default="resnet18"); args=p.parse_args()
    result={"math":math_checks(),"eda":run_eda(args.labels_dir,args.images_dir,args.out_dir)}
    if args.overfit: result["overfit"]=overfit_small_batch(args.backbone,args.labels_dir,args.images_dir)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
