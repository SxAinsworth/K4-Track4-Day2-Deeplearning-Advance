"""Controlled training-recipe ablations on the Step-1 winner."""
from __future__ import annotations
import json
from copy import copy
from pathlib import Path
import numpy as np
import pandas as pd
from train import Config, run

VARIANTS = [
    ("T01","A","frozen backbone",{"init":"frozen"}),
    ("T02","A","training from scratch",{"init":"scratch"}),
    ("T03","B","ColorJitter",{"aug":"color"}),
    ("T04","B","RandAugment",{"aug":"randaug"}),
    ("T05","B","Mixup alpha=1",{"mix":"mixup","mix_alpha":1.0}),
    ("T06","B","CutMix alpha=1",{"mix":"cutmix","mix_alpha":1.0}),
    ("T07","C","label smoothing 0.1",{"loss":"ls","label_smoothing":0.1}),
    ("T08","C","focal gamma=2",{"loss":"focal","focal_gamma":2.0}),
    ("T09","C","inverse-frequency weighted CE",{"loss":"ce_weighted","class_weight_beta":0.0}),
    ("T10","F","EMA decay=0.999",{"ema_decay":0.999}),
]


def _record(summary,axis,change,baseline_seed0=None,noise_std=None):
    delta=None if baseline_seed0 is None else summary["val_macro_f1"]-baseline_seed0
    verdict="baseline" if delta is None else ("phân biệt được" if abs(delta)>noise_std else "không phân biệt được")
    return {"exp_id":summary["exp_id"],"backbone":summary["pretrained_tag"],"axis":axis,
            "change_vs_T00":change,"seed":summary["seed"],"val_macro_f1":summary["val_macro_f1"],
            "val_top1":summary["val_top1"],"delta_vs_T00_seed0":delta,"baseline_macro_f1_std":noise_std,
            "verdict_vs_noise":verdict,"chinee_apple_f1":summary["val_f1_per_class"][0],
            "snake_weed_f1":summary["val_f1_per_class"][7],"best_epoch":summary["best_epoch"],
            "train_seconds_epoch":summary["mean_train_seconds"],"curve":summary["curve"]}


def _write_sheet(frame,path):
    mode="a" if path.exists() else "w"; kwargs={"engine":"openpyxl","mode":mode}
    if mode=="a": kwargs["if_sheet_exists"]="replace"
    with pd.ExcelWriter(path,**kwargs) as writer:
        frame.to_excel(writer,sheet_name="Training",index=False,startrow=2)
        sheet=writer.book["Training"]; sheet["A1"]="DeepWeeds training-recipe ablations — fold 0"
        sheet.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(frame.columns)); sheet.freeze_panes="A4"
        for cell in sheet[3]:
            font=copy(cell.font); font.bold=True; font.color="FFFFFF"; cell.font=font
            cell.fill=__import__("openpyxl").styles.PatternFill("solid",fgColor="1F4E78")
        for idx,name in enumerate(frame.columns,1):
            width=max(12,min(40,max(len(name)+2,*(len(str(v))+2 for v in frame[name].fillna("")))))
            sheet.column_dimensions[__import__("openpyxl").utils.get_column_letter(idx)].width=width
        for row in range(4,4+len(frame)):
            for name in ("val_macro_f1","val_top1","delta_vs_T00_seed0","baseline_macro_f1_std",
                         "chinee_apple_f1","snake_weed_f1"):
                sheet.cell(row,frame.columns.get_loc(name)+1).number_format="0.0000"


def _cfg(root,backbone,exp_id,seed,epochs,batch_size,overrides):
    return Config(exp_id=exp_id,backbone=backbone,seed=seed,epochs=epochs,batch_size=batch_size,
                  amp=True,measure_latency=False,curve_label=overrides.pop("curve_label"),
                  out_dir=str(root/"runs"),pred_dir=str(root/"predictions"),curves_dir=str(root/"curves"),**overrides)


def run_training_ablations(artifact_root: str | Path, epochs: int = 12,
                           batch_size: int = 64) -> dict:
    root=Path(artifact_root); selection_path=root/"backbone_selection.json"
    if not selection_path.is_file(): raise FileNotFoundError("Chưa có backbone_selection.json; chạy Bước 1 trước")
    backbone=json.loads(selection_path.read_text(encoding="utf-8"))["best_quality_backbone"]

    baseline=[]
    for seed in (0,1,2):
        cfg=_cfg(root,backbone,"T00",seed,epochs,batch_size,{"curve_label":f"baseline_seed{seed}"})
        baseline.append(run(cfg))
    base0=baseline[0]["val_macro_f1"]
    noise_std=float(np.std([x["val_macro_f1"] for x in baseline],ddof=1))
    records=[_record(item,"Baseline","T00",None,noise_std) for item in baseline]

    summaries={}
    for exp_id,axis,change,overrides in VARIANTS:
        values=dict(overrides); values["curve_label"]=change.lower().replace(" ","_").replace("=","")
        summary=run(_cfg(root,backbone,exp_id,0,epochs,batch_size,values)); summaries[exp_id]=summary
        records.append(_record(summary,axis,change,base0,noise_std))
        frame=pd.DataFrame(records); frame.to_csv(root/"training.csv",index=False); _write_sheet(frame,root/"results.xlsx")

    # Best augmentation and loss always enter the exploratory combination. Other axes enter only if > noise.
    deltas={key:value["val_macro_f1"]-base0 for key,value in summaries.items()}
    best_aug=max(("T03","T04","T05","T06"),key=lambda key:deltas[key])
    best_loss=max(("T07","T08","T09"),key=lambda key:deltas[key])
    selected=[best_aug,best_loss]
    for key in (max(("T01","T02"),key=lambda item:deltas[item]),"T10"):
        if deltas[key] > noise_std: selected.append(key)
    combo_overrides={}
    lookup={row[0]:row[3] for row in VARIANTS}
    for key in selected: combo_overrides.update(lookup[key])
    combo_overrides["curve_label"]="best_combination"
    combo=run(_cfg(root,backbone,"T11",0,epochs,batch_size,combo_overrides))
    combo_record=_record(combo,"Combined"," + ".join(selected),base0,noise_std)
    records.append(combo_record); frame=pd.DataFrame(records)
    frame.to_csv(root/"training.csv",index=False); _write_sheet(frame,root/"results.xlsx")

    seed0=frame[frame.seed==0].copy(); best=seed0.sort_values("val_macro_f1",ascending=False).iloc[0]
    selected_delta=sum(max(0,deltas[key]) for key in selected); combo_delta=combo["val_macro_f1"]-base0
    interaction=("cộng dồn gần đầy đủ" if selected_delta>0 and combo_delta>=0.75*selected_delta else
                 "cộng dồn một phần / có tương tác" if combo_delta>noise_std else "không phân biệt được hoặc triệt tiêu")
    best_row_overrides={} if best.exp_id=="T00" else (combo_overrides if best.exp_id=="T11" else lookup[best.exp_id])
    recipe={"backbone":backbone,"exp_id":best.exp_id,"overrides":best_row_overrides,
            "val_macro_f1":float(best.val_macro_f1),"baseline_seed0":base0,"baseline_std":noise_std,
            "combo_components":selected,"combo_interaction":interaction}
    (root/"training_selection.json").write_text(json.dumps(recipe,indent=2),encoding="utf-8")
    lines=["# Phân tích công thức huấn luyện","",f"Backbone: `{backbone}`. T00 qua ba seed có macro-F1 std = {noise_std:.4f}.",""]
    for axis in ("A","B","C","F"):
        group=frame[(frame.axis==axis)&(frame.seed==0)].sort_values("val_macro_f1",ascending=False)
        if len(group):
            row=group.iloc[0]; lines.append(f"- Trục {axis}: tốt nhất `{row.exp_id}` ({row.change_vs_T00}), Δ={row.delta_vs_T00_seed0:+.4f}; {row.verdict_vs_noise}.")
    lines += [f"- Kết hợp T11: Δ={combo_delta:+.4f}; hiệu ứng {interaction}.",
              f"- Công thức chuyển sang Bước 3: `{best.exp_id}`, macro-F1 val={best.val_macro_f1:.4f}.","",
              "Các ablation dùng seed 0; std của ba lượt T00 là ngưỡng nhiễu tham chiếu, không phải std riêng của từng kỹ thuật."]
    (root/"training_analysis.md").write_text("\n".join(lines),encoding="utf-8")
    return {"table":frame,"selection":recipe,"results_xlsx":str(root/"results.xlsx")}
