"""Locked final training, one-pass test inference, official evaluation and error analysis."""
from __future__ import annotations
import json
import subprocess
import sys
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import inference
from step2_training import baseline_config
from train import Config, run

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from eval import CLASS_NAMES, compute_metrics, save_predictions


def _paths(root,exp_id,seed):
    run_root=root/"runs"/exp_id/f"seed{seed}"
    return run_root,root/"predictions"/f"{exp_id}_seed{seed}_test.csv"


def _config(root,exp_id,seed,backbone,epochs,batch_size,overrides):
    values=dict(overrides)
    values.pop("curve_label",None)
    return Config(exp_id=exp_id,seed=seed,backbone=backbone,epochs=epochs,batch_size=batch_size,
                  save_test_predictions=True,amp=True,curve_label=f"seed{seed}",
                  out_dir=str(root/"runs"),pred_dir=str(root/"predictions"),curves_dir=str(root/"curves"),
                  **values)


def _calibrate_saved_predictions(root,seed,labels_dir):
    """Fit T on val only and calibrate already-saved val/test logits (no second test pass)."""
    run_root,test_path=_paths(root,"F01",seed)
    val_logits=np.load(run_root/"val_logits.npy"); test_logits=np.load(run_root/"test_logits.npy")
    val=pd.read_csv(Path(labels_dir)/"val_subset0.csv")
    test=pd.read_csv(Path(labels_dir)/"test_subset0.csv")
    temperature=inference.fit_temperature(val_logits,val.Label.to_numpy(dtype=np.int64))
    uncal_test=inference.apply_temperature(test_logits,1.0)
    uncal_val=inference.apply_temperature(val_logits,1.0)
    calibrated_test=inference.apply_temperature(test_logits,temperature)
    calibrated_val=inference.apply_temperature(val_logits,temperature)
    save_predictions(root/"predictions"/f"F01uncal_seed{seed}_test.csv",test.Filename,test.Label,uncal_test)
    save_predictions(root/"predictions"/f"F01uncal_seed{seed}_val.csv",val.Filename,val.Label,uncal_val)
    save_predictions(test_path,test.Filename,test.Label,calibrated_test)
    save_predictions(root/"predictions"/f"F01_seed{seed}_val.csv",val.Filename,val.Label,calibrated_val)
    return {"seed":seed,"temperature":temperature,
            "val_ece_before":compute_metrics(val.Label.to_numpy(),uncal_val.argmax(1),uncal_val)["ece"],
            "val_ece_after":compute_metrics(val.Label.to_numpy(),calibrated_val.argmax(1),calibrated_val)["ece"]}


def _run_eval(root,labels_dir,latency_p95):
    eval_py=ROOT/"eval.py"; pred=root/"predictions"; out=root/"eval_out"; out.mkdir(parents=True,exist_ok=True)
    common=["--test-csv",str(Path(labels_dir)/"test_subset0.csv"),"--labels",str(Path(labels_dir)/"labels.csv")]
    commands=[
        [sys.executable,str(eval_py),"score","--pred",str(pred/"F01_seed*_test.csv"),*common,"--tag","F01","--out",str(out)],
        [sys.executable,str(eval_py),"score","--pred",str(pred/"T00_seed*_test.csv"),*common,"--tag","T00","--out",str(out)],
        [sys.executable,str(eval_py),"grade","--final",str(pred/"F01_seed*_test.csv"),
         "--baseline",str(pred/"T00_seed*_test.csv"),"--uncal",str(pred/"F01uncal_seed*_test.csv"),
         "--final-val",str(pred/"F01_seed*_val.csv"),"--latency-p95-ms",str(latency_p95),
         "--latency-method","proper",*common,"--val-csv",str(Path(labels_dir)/"val_subset0.csv"),"--out",str(out)],
    ]
    logs=[]
    for command in commands:
        completed=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"eval.py thất bại ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}")
        print(completed.stdout); logs.append(completed.stdout)
    (out/"eval_console.md").write_text("\n\n---\n\n".join(logs),encoding="utf-8")


def _load_ensemble_predictions(root,prefix,split="test"):
    frames=[]
    for path in sorted((root/"predictions").glob(f"{prefix}_seed*_{split}.csv")):
        frame=pd.read_csv(path).set_index("Filename"); frames.append(frame)
    if len(frames)<3: raise ValueError(f"{prefix}: cần ít nhất 3 seed, hiện có {len(frames)}")
    names=frames[0].index
    if any(not names.equals(frame.index) for frame in frames[1:]): raise ValueError("Thứ tự Filename giữa các seed không khớp")
    probs=np.mean([frame[[f"p{i}" for i in range(9)]].to_numpy() for frame in frames],axis=0)
    return names.to_numpy(),frames[0].y_true.to_numpy(dtype=np.int64),probs


def _error_analysis(root,images_dir):
    names,y,probs=_load_ensemble_predictions(root,"F01"); pred=probs.argmax(1)
    metrics=compute_metrics(y,pred,probs); cm=metrics["confusion"]
    off=cm.copy(); np.fill_diagonal(off,0); true_idx,pred_idx=np.unravel_index(np.argmax(off),off.shape)
    curve_dir=root/"curves"; curve_dir.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,7)); image=ax.imshow(cm,cmap="Blues"); fig.colorbar(image,ax=ax)
    ax.set(xticks=range(9),yticks=range(9),xticklabels=CLASS_NAMES,yticklabels=CLASS_NAMES,
           xlabel="Predicted",ylabel="True",title="F01 test confusion matrix — ensemble of 3 seeds")
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right")
    for i in range(9):
        for j in range(9): ax.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=7)
    fig.tight_layout(); fig.savefig(curve_dir/"F01_confusion_matrix.png",dpi=180); plt.close(fig)

    selected=np.where((y==true_idx)&(pred==pred_idx))[0][:8]
    if len(selected):
        fig,axes=plt.subplots(2,4,figsize=(12,6)); axes=np.asarray(axes).ravel()
        for ax in axes: ax.axis("off")
        for ax,index in zip(axes,selected):
            with Image.open(Path(images_dir)/names[index]) as source: ax.imshow(source.convert("RGB"))
            ax.set_title(f"true: {CLASS_NAMES[true_idx]}\npred: {CLASS_NAMES[pred_idx]}",fontsize=8); ax.axis("off")
        fig.suptitle("Representative errors from the most-confused class pair"); fig.tight_layout()
        fig.savefig(curve_dir/"F01_misclassified_examples.png",dpi=180); plt.close(fig)
    return metrics,true_idx,pred_idx,int(off[true_idx,pred_idx])


def _write_results(root,temperatures,metrics,true_idx,pred_idx,count):
    final=pd.read_csv(root/"eval_out"/"F01_per_seed.csv")
    baseline=pd.read_csv(root/"eval_out"/"T00_per_seed.csv")
    final.insert(0,"exp_id","F01"); baseline.insert(0,"exp_id","T00")
    final_table=pd.concat([final,baseline],ignore_index=True)
    per_class=pd.read_csv(root/"eval_out"/"F01_per_class.csv")
    hard=per_class[per_class["class"].isin(["Chinee Apple","Snake Weed"])].copy()
    summary=pd.DataFrame([
        {"item":"final configuration","value":"F01 / selected Step-2 recipe / temperature scaling"},
        {"item":"test policy","value":"one forward pass per seed; calibration reuses saved logits"},
        {"item":"most confused pair","value":f"{CLASS_NAMES[true_idx]} -> {CLASS_NAMES[pred_idx]} ({count})"},
        {"item":"macro-F1 ensemble-of-seeds (diagnostic)","value":metrics["macro_f1"]},
        {"item":"top-1 ensemble-of-seeds (diagnostic)","value":metrics["top1"]},
        {"item":"temperatures","value":", ".join(f"seed {x['seed']}: {x['temperature']:.4f}" for x in temperatures)},
    ])
    path=root/"results.xlsx"; mode="a" if path.exists() else "w"; kwargs={"engine":"openpyxl","mode":mode}
    if mode=="a": kwargs["if_sheet_exists"]="replace"
    with pd.ExcelWriter(path,**kwargs) as writer:
        for sheet,frame,title in (("Final",final_table,"Final and baseline test metrics by seed"),
                                  ("PerClass",per_class,"F01 test per-class mean ± std"),
                                  ("Summary",summary,"Final locked configuration summary")):
            frame.to_excel(writer,sheet_name=sheet,index=False,startrow=2); ws=writer.book[sheet]
            ws["A1"]=title; ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(frame.columns)); ws.freeze_panes="A4"
            for cell in ws[3]:
                font=copy(cell.font); font.bold=True; font.color="FFFFFF"; cell.font=font
                cell.fill=__import__("openpyxl").styles.PatternFill("solid",fgColor="1F4E78")
            for column,name in enumerate(frame.columns,1):
                width=max(12,min(48,max(len(str(name))+2,*(len(str(v))+2 for v in frame[name].fillna("")))))
                ws.column_dimensions[__import__("openpyxl").utils.get_column_letter(column)].width=width
    return final_table,per_class,hard


def run_final(artifact_root: str | Path,labels_dir="data/labels",images_dir="data/images",epochs=12,batch_size=64):
    root=Path(artifact_root); root.mkdir(parents=True,exist_ok=True)
    training=json.loads((root/"training_selection.json").read_text(encoding="utf-8"))
    inference_selection=json.loads((root/"inference_selection.json").read_text(encoding="utf-8"))
    latency=pd.read_csv(root/"latency.csv")
    i07=latency[(latency.exp_id=="I07")&(latency.batch==1)]
    if i07.empty: raise ValueError("Thiếu latency batch-1 của I07 từ Bước 3")
    lock={"locked_from_validation":True,"backbone":training["backbone"],"training_source":training["exp_id"],
          "training_overrides":training.get("overrides",{}),"inference":"I07_temperature_scaling",
          "temperature_source":"fit separately on validation logits for each seed","seeds":[0,1,2],
          "epochs":epochs,"batch_size":batch_size,
          "step3_methods":{"offline":inference_selection.get("offline_method"),"realtime":inference_selection.get("realtime_method")}}
    # Chỉ so các trường định nghĩa cấu hình; số đo (độ trễ, T) có thể dao động khi chạy lại notebook.
    core=("backbone","training_source","training_overrides","inference","seeds","epochs","batch_size")
    lock_path=root/"final_lock.json"
    if lock_path.exists():
        saved=json.loads(lock_path.read_text(encoding="utf-8"))
        changed=[key for key in core if saved.get(key)!=lock[key]]
        if changed:
            raise RuntimeError(f"final_lock.json đã khóa cấu hình khác ở {changed}; không được đổi cấu hình sau khi mở test")
        lock=saved
    else:
        lock_path.write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding="utf-8")

    temperatures=[]
    for seed in lock["seeds"]:
        run(_config(root,"F01",seed,training["backbone"],epochs,batch_size,training.get("overrides",{})))
        temperatures.append(_calibrate_saved_predictions(root,seed,labels_dir))
        run(baseline_config(root,training["backbone"],seed,epochs,batch_size))  # tái sử dụng T00 của Bước 2

    latency_p95=float(i07.p95_ms.iloc[0]); _run_eval(root,labels_dir,latency_p95)
    metrics,true_idx,pred_idx,count=_error_analysis(root,images_dir)
    final_table,per_class,hard=_write_results(root,temperatures,metrics,true_idx,pred_idx,count)
    hard_text="; ".join(f"{row['class']}: precision={row.precision_mean:.4f}, recall={row.recall_mean:.4f}, F1={row.f1_mean:.4f}" for _,row in hard.iterrows())
    report=("# Phân tích chung kết\n\n"
            "Cấu hình đã được khóa bằng `final_lock.json` trước khi test được mở. Mỗi seed chỉ forward test một lần; temperature scaling tái sử dụng logit đã lưu.\n\n"
            f"- Cặp nhầm nhiều nhất: **{CLASS_NAMES[true_idx]} → {CLASS_NAMES[pred_idx]}**, {count} lần trong ma trận cộng gộp. "
            "Giả thuyết: hình thái lá, nền thực địa và mức che khuất tương tự làm giảm tín hiệu phân biệt.\n"
            f"- Hai lớp khó: {hard_text}.\n"
            "- Xem `eval_out/eval_console.md` để lấy mean ± std chính thức từ `eval.py`; không dùng điểm test để đổi cấu hình.\n")
    (root/"final_analysis.md").write_text(report,encoding="utf-8")
    return {"final":final_table,"per_class":per_class,"hard_classes":hard,"temperatures":pd.DataFrame(temperatures),
            "lock":lock,"eval_log":str(root/"eval_out"/"eval_console.md"),"results_xlsx":str(root/"results.xlsx")}
