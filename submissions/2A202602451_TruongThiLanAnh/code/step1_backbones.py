"""Run the fair five-backbone screen and build the Backbones result sheet."""
from __future__ import annotations
import json
from copy import copy
from pathlib import Path
import pandas as pd
from train import Config, run

BACKBONES = [
    ("B01", "resnet50.a1_in1k", "ResNet baseline"),
    ("B02", "convnext_tiny.fb_in1k", "Modern CNN"),
    ("B03", "deit_small_patch16_224.fb_in1k", "Transformer"),
    ("B04", "efficientnet_b0.ra_in1k", "Lightweight CNN"),
    ("B05", "mobilenetv3_large_100.ra_in1k", "Lightweight CNN"),
]


def _flatten(summary: dict, note: str) -> dict:
    latency=summary.get("latency",{})
    history=pd.read_csv(summary["history"])
    overfit=bool(len(history)>=3 and history.val_loss.iloc[-1] > history.val_loss.min()*1.05
                 and history.train_loss.iloc[-1] < history.train_loss.iloc[0])
    return {"exp_id":summary["exp_id"],"backbone":summary["backbone"],
            "pretrained_tag":summary["pretrained_tag"],"params_m":summary["params_m"],
            "gmac":summary["gmac"],"img_size":summary["img_size"],"epochs":summary["epochs"],
            "seed":summary["seed"],"physical_batch":summary["physical_batch"],
            "effective_batch":summary["effective_batch"],"val_macro_f1":summary["val_macro_f1"],
            "val_top1":summary["val_top1"],"best_epoch":summary["best_epoch"],
            "train_seconds_epoch":summary["mean_train_seconds"],"latency_p50_ms":latency.get("p50"),
            "latency_p95_ms":latency.get("p95"),"latency_dtype":latency.get("dtype"),
            "gpu":latency.get("gpu"),"curve":summary["curve"],"note":note,
            "overfit_flag":overfit}


def _write_workbook(frame: pd.DataFrame, path: Path) -> None:
    mode="a" if path.exists() else "w"
    kwargs={"engine":"openpyxl","mode":mode}
    if mode=="a": kwargs["if_sheet_exists"]="replace"
    with pd.ExcelWriter(path,**kwargs) as writer:
        frame.to_excel(writer,sheet_name="Backbones",index=False,startrow=2)
        sheet=writer.book["Backbones"]
        sheet["A1"]="DeepWeeds backbone screening — fold 0, T00, seed 0"
        sheet.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(frame.columns))
        sheet.freeze_panes="A4"; sheet.auto_filter.ref=sheet.dimensions
        for cell in sheet[3]:
            font=copy(cell.font); font.bold=True; font.color="FFFFFF"; cell.font=font
            cell.fill=__import__("openpyxl").styles.PatternFill("solid",fgColor="1F4E78")
        widths={name:max(12,min(38,max(len(name)+2,*(len(str(v))+2 for v in frame[name].fillna(""))))) for name in frame.columns}
        for idx,name in enumerate(frame.columns,1): sheet.column_dimensions[__import__("openpyxl").utils.get_column_letter(idx)].width=widths[name]
        for row in range(4,4+len(frame)):
            for name in ("val_macro_f1","val_top1"):
                sheet.cell(row,frame.columns.get_loc(name)+1).number_format="0.0000"
            for name in ("params_m","gmac","train_seconds_epoch","latency_p50_ms","latency_p95_ms"):
                sheet.cell(row,frame.columns.get_loc(name)+1).number_format="0.00"


def run_screening(artifact_root: str | Path, epochs: int = 12, batch_size: int = 64,
                  grad_accum_steps: int = 1, seed: int = 0) -> dict:
    """Run/continue B01..B05 and persist a table after every completed backbone."""
    root=Path(artifact_root); root.mkdir(parents=True,exist_ok=True)
    records=[]
    for exp_id,backbone,note in BACKBONES:
        cfg=Config(exp_id=exp_id,backbone=backbone,seed=seed,epochs=epochs,batch_size=batch_size,
                   grad_accum_steps=grad_accum_steps,amp=True,measure_latency=True,
                   latency_dtype="amp",latency_warmup=10,latency_iters=30,
                   out_dir=str(root/"runs"),pred_dir=str(root/"predictions"),curves_dir=str(root/"curves"))
        summary=run(cfg); records.append(_flatten(summary,note))
        frame=pd.DataFrame(records); frame.to_csv(root/"backbones.csv",index=False)
        _write_workbook(frame,root/"results.xlsx")
    frame=pd.DataFrame(records).sort_values("val_macro_f1",ascending=False).reset_index(drop=True)
    best=frame.iloc[0]
    eligible=frame[frame.val_macro_f1 >= best.val_macro_f1-0.02].sort_values("latency_p50_ms")
    balanced=eligible.iloc[0]
    selection={"best_quality":best.exp_id,"best_quality_backbone":best.pretrained_tag,
               "balanced":balanced.exp_id,"balanced_backbone":balanced.pretrained_tag,
               "rule":"balanced = fastest model within 0.02 macro-F1 of the best validation result",
               "caution":"Backbone screening uses one seed; small differences are not conclusive."}
    (root/"backbone_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    fastest_epoch=frame.sort_values("train_seconds_epoch").iloc[0]
    earliest=frame.sort_values("best_epoch").iloc[0]
    corr_latency=float(frame[["gmac","latency_p50_ms"]].corr().iloc[0,1])
    corr_train=float(frame[["gmac","train_seconds_epoch"]].corr().iloc[0,1])
    analysis=(f"# Phân tích backbone (seed {seed})\n\n"
              f"- Macro-F1 validation cao nhất: **{best.exp_id} — {best.pretrained_tag}** "
              f"({best.val_macro_f1:.4f}).\n"
              f"- Cấu hình cân bằng theo quy tắc đã khai báo: **{balanced.exp_id} — {balanced.pretrained_tag}**; "
              f"đây là model nhanh nhất trong phạm vi 0,02 macro-F1 so với model tốt nhất.\n"
              f"- Thời gian train/epoch thấp nhất: **{fastest_epoch.exp_id}** "
              f"({fastest_epoch.train_seconds_epoch:.1f} giây).\n"
              f"- Checkpoint tốt nhất xuất hiện sớm nhất: **{earliest.exp_id}**, epoch {int(earliest.best_epoch)}.\n"
              f"- Tương quan Pearson GMAC–latency p50: {corr_latency:.3f}; GMAC–thời gian train/epoch: {corr_train:.3f}. "
              "FLOPs chỉ là chỉ báo; kernel, memory access và kiến trúc phần cứng cũng ảnh hưởng tốc độ.\n"
              f"- Model có dấu hiệu overfit theo quy tắc val loss cuối > min val loss 5%: "
              f"{', '.join(frame.loc[frame.overfit_flag,'exp_id']) or 'không có'}.\n\n"
              "Kết quả chỉ dùng một seed, nên chênh lệch nhỏ chưa đủ để kết luận chắc chắn. "
              "Thứ hạng ImageNet và DeepWeeds không được coi là tương đương; bảng này chỉ xếp hạng theo validation DeepWeeds.\n")
    (root/"backbone_analysis.md").write_text(analysis,encoding="utf-8")
    return {"table":frame,"selection":selection,"results_xlsx":str(root/"results.xlsx")}
