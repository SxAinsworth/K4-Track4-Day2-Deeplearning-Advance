"""Validation-only inference comparison and correct latency benchmarking."""
from __future__ import annotations
import json
from copy import copy
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
import benchmark, dataset, inference, model as model_utils
from eval import compute_metrics, save_predictions


def _softmax(logits):
    values=np.asarray(logits,dtype=np.float64); values-=values.max(axis=1,keepdims=True)
    result=np.exp(values); return result/result.sum(axis=1,keepdims=True)


def _load_model(backbone,checkpoint,device,init="finetune"):
    net=model_utils.build_model(backbone,pretrained=False,init=init).to(device)
    state=torch.load(checkpoint,map_location=device,weights_only=True); net.load_state_dict(state["model"]); return net.eval()


def _multicrop_logits(model,loader,device,crop=224):
    names=[]; labels=[]; outputs=[[] for _ in range(5)]; model.eval()
    with torch.inference_mode():
        for images,target,batch_names in loader:
            images=images.to(device,non_blocking=True); views=inference.views_multicrop(images,crop)
            for index,view in enumerate(views): outputs[index].append(model(view).float().cpu().numpy())
            names.extend(batch_names); labels.append(target.numpy())
    return names,np.concatenate(labels),[np.concatenate(parts) for parts in outputs]


def _measure_method(model,device,batch_size,img_size,kind="single",dtype="fp32",other_models=None,T=1.0):
    x=torch.randn(batch_size,3,img_size,img_size,device=device)
    models=[model]+list(other_models or []); [item.eval() for item in models]
    amp=dtype=="amp" and device.type=="cuda"
    def fn():
        with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=amp):
            if kind=="hflip_prob":
                torch.stack([torch.softmax(model(x),1),torch.softmax(model(inference.view_hflip(x)),1)]).mean(0)
            elif kind=="hflip_logit":
                torch.softmax(torch.stack([model(x),model(inference.view_hflip(x))]).mean(0),1)
            elif kind=="multicrop":
                torch.stack([torch.softmax(model(view),1) for view in inference.views_multicrop(x,224)]).mean(0)
            elif kind=="ensemble":
                torch.stack([torch.softmax(item(x),1) for item in models]).mean(0)
            elif kind=="temperature": torch.softmax(model(x)/T,1)
            else: torch.softmax(model(x),1)
    measured=benchmark.bench(fn,warmup=10,iters=50,sync=torch.cuda.synchronize if device.type=="cuda" else None)
    return {"batch":batch_size,"p50_ms":measured["p50"],"p95_ms":measured["p95"],
            "p99_ms":measured["p99"],"mean_ms":measured["mean"],
            "images_per_s":batch_size/(measured["p50"]/1000),"dtype":dtype,
            "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
            "img_size":img_size,"warmup":10,"iterations":50,"torch":torch.__version__,
            "includes_preprocessing":kind in {"hflip_prob","hflip_logit","multicrop"}}


def _write_sheets(inference_frame,latency_frame,path):
    mode="a" if path.exists() else "w"; kwargs={"engine":"openpyxl","mode":mode}
    if mode=="a": kwargs["if_sheet_exists"]="replace"
    with pd.ExcelWriter(path,**kwargs) as writer:
        inference_frame.to_excel(writer,sheet_name="Inference",index=False,startrow=2)
        latency_frame.to_excel(writer,sheet_name="Latency",index=False,startrow=2)
        for sheet_name,title in (("Inference","Validation inference comparison"),("Latency","Latency benchmark conditions")):
            sheet=writer.book[sheet_name]; frame=inference_frame if sheet_name=="Inference" else latency_frame
            sheet["A1"]=title; sheet.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(frame.columns)); sheet.freeze_panes="A4"
            for cell in sheet[3]:
                font=copy(cell.font); font.bold=True; font.color="FFFFFF"; cell.font=font
                cell.fill=__import__("openpyxl").styles.PatternFill("solid",fgColor="1F4E78")
            for idx,name in enumerate(frame.columns,1):
                width=max(12,min(38,max(len(name)+2,*(len(str(v))+2 for v in frame[name].fillna("")))))
                sheet.column_dimensions[__import__("openpyxl").utils.get_column_letter(idx)].width=width
            for row in range(4,4+len(frame)):
                for name in set(frame.columns)&{"val_macro_f1","val_top1","ece","relative_cost","p50_ms","p95_ms","p99_ms","mean_ms","images_per_s"}:
                    sheet.cell(row,frame.columns.get_loc(name)+1).number_format="0.0000"


def run_inference_comparison(artifact_root: str | Path, labels_dir="data/labels", images_dir="data/images"):
    root=Path(artifact_root); recipe_path=root/"training_selection.json"
    if not recipe_path.is_file(): raise FileNotFoundError("Chưa có training_selection.json; chạy Bước 2 trước")
    recipe=json.loads(recipe_path.read_text(encoding="utf-8")); backbone=recipe["backbone"]; exp_id=recipe["exp_id"]
    overrides=recipe.get("overrides",{}); init=overrides.get("init","finetune")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint=root/"runs"/exp_id/"seed0"/"best.pt"
    model=_load_model(backbone,checkpoint,device,init)
    # Step 3 is validation-only: deliberately do not open the test split.
    val_df=pd.read_csv(Path(labels_dir)/"val_subset0.csv")
    required={"Filename","Label"}
    if not required.issubset(val_df.columns): raise ValueError(f"val_subset0.csv thiếu cột: {required-set(val_df.columns)}")
    names=val_df.Filename.astype(str).tolist(); y=val_df.Label.to_numpy(dtype=np.int64)
    logits=np.load(root/"runs"/exp_id/"seed0"/"val_logits.npy"); base_probs=_softmax(logits)
    if len(logits)!=len(val_df): raise ValueError("Số logit val không khớp val_subset0.csv")
    standard_loader=dataset.make_loader(val_df,images_dir,dataset.build_transforms(False,224),64,False,None,2)

    methods=[]; probability_map={}; latency_rows=[]
    def add(code,name,probs,k,latency_kind="single",dtype="fp32",models=None,T=1.0,input_size=224,note="",primary_model=None):
        metric=compute_metrics(y,probs.argmax(1),probs); probability_map[code]=probs
        latencies=[]
        for batch in (1,32):
            row=_measure_method(primary_model or model,device,batch,input_size,latency_kind,dtype,models,T)
            row.update(exp_id=code,method=name,fused_bn=code=="I09",
                       measurement_scope="GPU tensor transforms + model + aggregation; excludes image decode/CPU loader")
            latency_rows.append(row)
            if batch==1: latencies=row
        methods.append({"exp_id":code,"method":name,"checkpoint":exp_id,"K":k,
                        "val_macro_f1":metric["macro_f1"],"val_top1":metric["top1"],"ece":metric["ece"],
                        "latency_p50_ms":latencies["p50_ms"],"latency_p95_ms":latencies["p95_ms"],
                        "latency_p99_ms":latencies["p99_ms"],"dtype":dtype,"note":note})
        save_predictions(root/"predictions"/f"{code}_seed0_val.csv",names,y,probs)

    add("I00","1-view FP32",base_probs,1)
    _,y_flip,flip_logits=inference.predict_logits(model,standard_loader,device,inference.view_hflip)
    if not np.array_equal(y,y_flip): raise ValueError("Thứ tự validation thay đổi ở hflip")
    add("I01","TTA horizontal flip — probability mean",inference.aggregate_views([logits,flip_logits],"prob"),2,"hflip_prob")

    raw_transform=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),
                                      transforms.Normalize(dataset.IMAGENET_MEAN,dataset.IMAGENET_STD)])
    crop_loader=dataset.make_loader(val_df,images_dir,raw_transform,32,False,None,2)
    crop_names,crop_y,crop_logits=_multicrop_logits(model,crop_loader,device,224)
    if crop_names!=names or not np.array_equal(crop_y,y): raise ValueError("Thứ tự validation thay đổi ở 5-crop")
    add("I02","TTA 5-crop — probability mean",inference.aggregate_views(crop_logits,"prob"),5,"multicrop",input_size=256)
    add("I03","TTA horizontal flip — logit mean",inference.aggregate_views([logits,flip_logits],"logit"),2,"hflip_logit")

    backbone_table=pd.read_csv(root/"backbones.csv").sort_values("val_macro_f1",ascending=False).head(2)
    ensemble_models=[]; ensemble_probabilities=[]
    for row in backbone_table.itertuples():
        ensemble_probabilities.append(_softmax(np.load(root/"runs"/row.exp_id/"seed0"/"val_logits.npy")))
        ensemble_models.append(_load_model(row.pretrained_tag,root/"runs"/row.exp_id/"seed0"/"best.pt",device))
    add("I05","Ensemble top-2 backbones",inference.ensemble_probs(ensemble_probabilities),2,"ensemble",
        models=ensemble_models[1:],note="Bước 1 T00 checkpoints",primary_model=ensemble_models[0])
    del ensemble_models
    if device.type=="cuda": torch.cuda.empty_cache()

    temperature=inference.fit_temperature(logits,y); calibrated=inference.apply_temperature(logits,temperature)
    add("I07","Temperature scaling",calibrated,1,"temperature",T=temperature,note=f"T={temperature:.6f}, fit on val")
    _,y_amp,amp_logits=inference.predict_logits(model,standard_loader,device,amp=True)
    if not np.array_equal(y,y_amp): raise ValueError("Thứ tự validation thay đổi ở AMP")
    add("I08","1-view AMP",_softmax(amp_logits),1,"single",dtype="amp")

    bn_before=sum(isinstance(module,nn.BatchNorm2d) for module in model.modules())
    fusion={"applicable":bn_before>0,"bn_before":bn_before}
    if bn_before:
        fused=inference.fuse_conv_bn(model).to(device); bn_after=sum(isinstance(module,nn.BatchNorm2d) for module in fused.modules())
        sample=torch.randn(2,3,224,224,device=device)
        with torch.inference_mode(): max_error=float((model(sample)-fused(sample)).abs().max())
        fusion.update(bn_after=bn_after,max_abs_error=max_error)
        _,yf,fused_logits=inference.predict_logits(fused,standard_loader,device)
        if max_error>1e-4: raise AssertionError(f"Sai số gộp BN quá lớn: {max_error}")
        original=model; model=fused
        add("I09","Fused Conv-BN",_softmax(fused_logits),1,"single",note=f"max error={max_error:.2e}")
        model=original
    (root/"bn_fusion_check.json").write_text(json.dumps(fusion,indent=2),encoding="utf-8")

    inference_frame=pd.DataFrame(methods); latency_frame=pd.DataFrame(latency_rows)
    baseline_latency=float(inference_frame.loc[inference_frame.exp_id=="I00","latency_p50_ms"].iloc[0])
    inference_frame["relative_cost"]=inference_frame.latency_p50_ms/baseline_latency
    _write_sheets(inference_frame,latency_frame,root/"results.xlsx")
    inference_frame.to_csv(root/"inference.csv",index=False); latency_frame.to_csv(root/"latency.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(inference_frame.latency_p95_ms,inference_frame.val_macro_f1)
    for row in inference_frame.itertuples(): ax.annotate(row.exp_id,(row.latency_p95_ms,row.val_macro_f1),xytext=(4,4),textcoords="offset points")
    ax.set(xlabel="Latency p95, batch 1 (ms)",ylabel="Validation macro-F1",title="Accuracy–latency trade-off"); ax.grid(alpha=.25)
    fig.tight_layout(); (root/"curves").mkdir(parents=True,exist_ok=True); fig.savefig(root/"curves"/"I_accuracy_latency.png",dpi=180); plt.close(fig)

    offline=inference_frame.sort_values(["val_macro_f1","latency_p95_ms"],ascending=[False,True]).iloc[0]
    realtime_candidates=inference_frame[inference_frame.latency_p95_ms<=100]
    realtime=None if realtime_candidates.empty else realtime_candidates.sort_values(["val_macro_f1","latency_p95_ms"],ascending=[False,True]).iloc[0]
    selection={"temperature":temperature,"offline_method":offline.exp_id,
               "realtime_method":None if realtime is None else realtime.exp_id,
               "realtime_p95_ms":None if realtime is None else float(realtime.latency_p95_ms),
               "base_checkpoint_exp_id":exp_id,"backbone":backbone}
    (root/"inference_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    before=float(inference_frame.loc[inference_frame.exp_id=="I00","ece"].iloc[0]); after=float(inference_frame.loc[inference_frame.exp_id=="I07","ece"].iloc[0])
    analysis=(f"# Phân tích suy luận\n\n- ECE trước/sau temperature scaling: {before:.4f} → {after:.4f}, T={temperature:.6f}.\n"
              f"- Phương pháp macro-F1 cao nhất trên val: {offline.exp_id} ({offline.val_macro_f1:.4f}), p95={offline.latency_p95_ms:.2f} ms.\n"
              f"- Cấu hình thời gian thực p95≤100 ms: {selection['realtime_method'] or 'không có'}.\n"
              "- TTA/ensemble được đánh giá cùng chi phí tương đối; phương pháp offline chỉ được chọn khi mức tăng chất lượng biện minh cho độ trễ.\n"
              f"- Gộp Conv-BN: {json.dumps(fusion,ensure_ascii=False)}.\n")
    (root/"inference_analysis.md").write_text(analysis,encoding="utf-8")
    return {"inference":inference_frame,"latency":latency_frame,"selection":selection,"results_xlsx":str(root/"results.xlsx")}
