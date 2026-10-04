"""Bước 5: gom kết quả Bước 1–4 thành results.xlsx (7 sheet), curves/, report.md và các file nộp.

Mọi con số được đọc lại từ log/predictions đã lưu (không nhập tay). Chỉ số test được tính lại từ
predictions/ bằng chính eval.read_pred + eval.compute_metrics, rồi đối chiếu với eval_out/ của eval.py.
"""
from __future__ import annotations
import json, platform, re, shutil, sys
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import eval as ev

HARD=("Chinee Apple","Snake Weed")
PAPER={"resnet50_top1":0.957,"inception_top1":0.951,"Chinee Apple":0.885,"Snake Weed":0.888}  # trích dẫn README 2.3
COLAB_URL=("https://colab.research.google.com/github/SxAinsworth/K4-Track4-Day2-Deeplearning-Advance/blob/main/"
           "submissions/2A202602451_TruongThiLanAnh/code/lab_day2.ipynb")
TRAIN_EXP=re.compile(r"^[BTF]\d{2}$")
SCORE_WORDS=("macro-F1","top-1","ECE","F1","precision","recall","Δ","std","NLL","balanced")
COST_WORDS=("(ms)","(s)","(M)","GMAC","img/s","cost")


# --------------------------------------------------------------------------- helpers
def _json(path,default=None):
    path=Path(path); return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _copy(src,dst):
    src,dst=Path(src),Path(dst)
    if not src.is_file() or (dst.exists() and src.resolve()==dst.resolve()): return
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)


def _fmt(value,digits=4):
    if value is None or (isinstance(value,float) and np.isnan(value)): return "—"
    if isinstance(value,(bool,np.bool_)): return "có" if value else "không"
    if isinstance(value,(int,np.integer)): return str(int(value))
    if isinstance(value,(float,np.floating)): return f"{value:.{digits}f}"
    return str(value).replace("|","\\|")


def _md(frame):
    """Bảng markdown không cần thư viện tabulate; 4 chữ số cho điểm số, 2 cho chi phí như trong results.xlsx."""
    def cell(value,column):
        score=any(w in str(column) for w in SCORE_WORDS); cost=any(w in str(column) for w in COST_WORDS)
        if isinstance(value,(float,np.floating)) and not np.isnan(value) and float(value).is_integer() and not (score or cost):
            return str(int(value))
        return _fmt(value,4 if score else 2)
    head="| "+" | ".join(map(str,frame.columns))+" |"; sep="|"+"---|"*len(frame.columns)
    rows=["| "+" | ".join(cell(v,c) for v,c in zip(row,frame.columns))+" |" for row in frame.itertuples(index=False)]
    return "\n".join([head,sep,*rows])


def _pm(mean,std,digits=4): return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _runs(root):
    """Mọi lần huấn luyện B/T/F còn log: exp_id, seed, summary, config, history."""
    out=[]
    for summary_path in sorted((Path(root)/"runs").glob("*/seed*/summary.json")):
        run_dir=summary_path.parent; exp_id=run_dir.parent.name
        if not TRAIN_EXP.match(exp_id) or not (run_dir/"history.csv").is_file(): continue
        out.append({"exp_id":exp_id,"seed":int(run_dir.name[4:]),"dir":run_dir,"summary":_json(summary_path),
                    "config":_json(run_dir/"config.json",{}),"history":pd.read_csv(run_dir/"history.csv")})
    return out


def _curve_name(run):
    name=Path(run["summary"].get("curve") or "").name
    return name if name.startswith(run["exp_id"]+"_") else f"{run['exp_id']}_seed{run['seed']}.png"


# --------------------------------------------------------------------------- curves
def plot_curves(history,path,title):
    """Loss train/val, macro-F1/top-1 val (đánh dấu epoch được chọn) và LR theo epoch."""
    df=history; best=int(df.val_macro_f1.idxmax())
    fig,axes=plt.subplots(1,3,figsize=(16,4.2))
    axes[0].plot(df.epoch,df.train_loss,"o-",label="train loss"); axes[0].plot(df.epoch,df.val_loss,"o-",label="val loss")
    axes[0].set(xlabel="Epoch",ylabel="Loss",title="Loss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(df.epoch,df.val_macro_f1,"o-",label="val macro-F1"); axes[1].plot(df.epoch,df.val_top1,"o-",label="val top-1")
    axes[1].axvline(df.epoch[best],color="grey",ls="--",lw=1,label=f"checkpoint (epoch {int(df.epoch[best])})")
    axes[1].set(xlabel="Epoch",ylabel="Score",title="Validation metrics",ylim=(min(.5,float(df[["val_macro_f1","val_top1"]].min().min())-.02),1))
    axes[1].legend(); axes[1].grid(alpha=.25)
    if "lr" in df:
        axes[2].plot(df.epoch,df.lr,"o-",color="tab:green",label="LR head (cuối epoch)")
        axes[2].set(xlabel="Epoch",ylabel="Learning rate",title="Warmup + cosine"); axes[2].legend(); axes[2].grid(alpha=.25)
    else: axes[2].axis("off")
    from matplotlib.ticker import MaxNLocator
    for ax in axes: ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title); fig.tight_layout(); Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)


def build_curves(runs,curves_dir):
    rows=[]
    for run in runs:
        name=_curve_name(run); h=run["history"]; s=run["summary"]
        plot_curves(h,Path(curves_dir)/name,f"{run['exp_id']} — {s.get('pretrained_tag',s.get('backbone'))} — seed {run['seed']}")
        best=float(h.val_macro_f1.max())
        rows.append({"exp_id":run["exp_id"],"seed":run["seed"],"curve file":f"curves/{name}","epochs":len(h),
                     "best epoch":int(h.epoch[h.val_macro_f1.idxmax()]),
                     "first epoch ≥ 98% best macro-F1":int(h.epoch[h.val_macro_f1>=.98*best].iloc[0]),
                     "train loss first→last":f"{h.train_loss.iloc[0]:.3f} → {h.train_loss.iloc[-1]:.3f}",
                     "val loss min (epoch)":f"{h.val_loss.min():.3f} ({int(h.epoch[h.val_loss.idxmin()])})",
                     "val loss last":float(h.val_loss.iloc[-1]),
                     "overfit flag":bool(h.val_loss.iloc[-1]>1.05*h.val_loss.min() and h.train_loss.iloc[-1]<h.train_loss.min()*1.05)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- metrics from predictions
def _pred_metrics(path):
    pred=ev.read_pred(str(path)); return pred,ev.compute_metrics(pred.y_true,pred.y_pred,pred.probs)


def _group(pred_dir,tag,split):
    files=sorted(Path(pred_dir).glob(f"{tag}_seed*_{split}.csv"),key=lambda p:int(re.search(r"seed(\d+)",p.name).group(1)))
    return [(int(re.search(r"seed(\d+)",p.name).group(1)),*_pred_metrics(p)) for p in files]


def final_tables(root,lock):
    pred_dir=Path(root)/"predictions"; names=list(ev.CLASS_NAMES)
    overrides=lock.get("training_overrides") or {}
    recipe=lock.get("training_source","T00")+(f" {json.dumps(overrides)}" if overrides else "")
    configs=[("F01",f"{lock['backbone']} + {recipe} + I07 temperature scaling (T khớp trên val)"),
             ("F01uncal",f"F01 trước temperature scaling (cùng logit, T = 1)"),
             ("T00",f"{lock['backbone']} + T00 + I00 (1-view) — mốc")]
    rows=[]; per_class=[]; groups={}
    for tag,desc in configs:
        test=_group(pred_dir,tag,"test"); val={seed:m for seed,_,m in _group(pred_dir,tag,"val")}
        if not test: continue
        groups[tag]=test
        for seed,pred,m in test:
            rows.append({"exp_id":tag,"configuration":desc,"seed":str(seed),
                         "val macro-F1":val[seed]["macro_f1"] if seed in val else np.nan,
                         "val top-1":val[seed]["top1"] if seed in val else np.nan,
                         "test macro-F1":m["macro_f1"],"test top-1":m["top1"],"test balanced acc":m["balanced_acc"],
                         "test ECE (15 bins)":m["ece"],"test NLL":m["nll"],
                         "test recall Chinee Apple":m["recall"][0],"test recall Snake Weed":m["recall"][7],
                         "test images":m["n"]})
        block=pd.DataFrame([r for r in rows if r["exp_id"]==tag]); numeric=[c for c in block.columns if c.startswith(("val ","test "))]
        for label,fn in ((f"mean ({len(test)} seed)",lambda c:c.mean()),("std (ddof=1)",lambda c:c.std(ddof=1))):
            rows.append({"exp_id":tag,"configuration":desc,"seed":label,**{c:float(fn(block[c])) for c in numeric if c!="test images"},
                         "test images":int(block["test images"].iloc[0])})
        if tag!="F01uncal":
            stats={key:ev.mean_std([m[key] for _,_,m in test]) for key in ("precision","recall","f1")}
            for i,cls in enumerate(names):
                per_class.append({"configuration":"F01 (chung kết, tốt nhất)" if tag=="F01" else "T00 + I00 (mốc)","class":cls,
                                  "test images":int(test[0][2]["support"][i]),
                                  "precision mean":stats["precision"][0][i],"precision std":stats["precision"][1][i],
                                  "recall mean":stats["recall"][0][i],"recall std":stats["recall"][1][i],
                                  "F1 mean":stats["f1"][0][i],"F1 std":stats["f1"][1][i]})
    return pd.DataFrame(rows),pd.DataFrame(per_class),groups


def _stat(final,tag,col,row="mean"):
    sel=final[(final.exp_id==tag)&final.seed.str.startswith(row)]
    return float(sel[col].iloc[0]) if len(sel) else np.nan


# --------------------------------------------------------------------------- sheets
def _rename(frame,mapping): return frame.rename(columns=mapping)[[mapping.get(c,c) for c in frame.columns]]


def sheet_backbones(root,runs_by_key,selection):
    df=pd.read_csv(Path(root)/"backbones.csv")
    def chosen(exp_id):
        reasons=[text for key,text in (("best_quality","macro-F1 val cao nhất → Bước 2–4"),("balanced","cân bằng F1–độ trễ"))
                 if selection.get(key)==exp_id]
        return "; ".join(reasons)
    df["selected"]=df.exp_id.map(chosen)
    df["curve"]=[f"curves/{_curve_name(runs_by_key[(e,int(s))])}" if (e,int(s)) in runs_by_key else "THIẾU" for e,s in zip(df.exp_id,df.seed)]
    return _rename(df,{"backbone":"architecture","pretrained_tag":"weight tag (timm)","params_m":"params (M)","gmac":"GMAC",
                       "img_size":"resolution (px)","epochs":"epochs","physical_batch":"physical batch","effective_batch":"effective batch",
                       "val_macro_f1":"val macro-F1","val_top1":"val top-1","best_epoch":"best epoch",
                       "train_seconds_epoch":"train time / epoch (s)","latency_p50_ms":"latency batch-1 p50 (ms)",
                       "latency_p95_ms":"latency batch-1 p95 (ms)","latency_dtype":"latency dtype","note":"note","overfit_flag":"overfit flag"})


def sheet_training(root,runs_by_key,selection):
    df=pd.read_csv(Path(root)/"training.csv")
    df["curve"]=[f"curves/{_curve_name(runs_by_key[(e,int(s))])}" if (e,int(s)) in runs_by_key else "THIẾU" for e,s in zip(df.exp_id,df.seed)]
    df["note"]=["công thức chọn cho Bước 3–4" if (e==selection.get("exp_id") and int(s)==0) else
                ("kết hợp: "+" + ".join(selection.get("combo_components",[])) if e=="T11" else "") for e,s in zip(df.exp_id,df.seed)]
    return _rename(df,{"axis":"axis (A–G)","change_vs_T00":"change vs T00","val_macro_f1":"val macro-F1","val_top1":"val top-1",
                       "delta_vs_T00_seed0":"Δ macro-F1 vs T00 seed 0","baseline_macro_f1_std":"T00 macro-F1 std (3 seed)",
                       "verdict_vs_noise":"Δ vs noise","chinee_apple_f1":"val F1 Chinee Apple","snake_weed_f1":"val F1 Snake Weed",
                       "best_epoch":"best epoch","train_seconds_epoch":"train time / epoch (s)"})


def sheet_inference(root):
    df=pd.read_csv(Path(root)/"inference.csv"); lat=pd.read_csv(Path(root)/"latency.csv")
    thr=lat[lat.batch==32].set_index("exp_id").images_per_s; thr1=lat[lat.batch==1].set_index("exp_id").images_per_s
    base=float(df.loc[df.exp_id=="I00","val_macro_f1"].iloc[0])
    df["Δ macro-F1 vs I00"]=df.val_macro_f1-base
    df["throughput batch-1 (img/s)"]=df.exp_id.map(thr1); df["throughput batch-32 (img/s)"]=df.exp_id.map(thr)
    order=["exp_id","method","checkpoint","K","val_macro_f1","val_top1","Δ macro-F1 vs I00","ece","latency_p50_ms","latency_p95_ms",
           "latency_p99_ms","throughput batch-1 (img/s)","throughput batch-32 (img/s)","relative_cost","dtype","note"]
    df=df[[c for c in order if c in df.columns]]
    return _rename(df,{"checkpoint":"model / checkpoint","K":"K (views or models)","val_macro_f1":"val macro-F1","val_top1":"val top-1",
                       "ece":"val ECE (15 bins)","latency_p50_ms":"latency batch-1 p50 (ms)","latency_p95_ms":"latency batch-1 p95 (ms)",
                       "latency_p99_ms":"latency batch-1 p99 (ms)","relative_cost":"relative cost vs I00 (p50)"})


def sheet_latency(root):
    df=pd.read_csv(Path(root)/"latency.csv")
    df.insert(0,"configuration",df.exp_id+" — "+df.method); df=df.drop(columns=["exp_id","method"])
    order=["configuration","gpu","dtype","batch","fused_bn","img_size","p50_ms","p95_ms","p99_ms","mean_ms","images_per_s",
           "warmup","iterations","includes_preprocessing","measurement_scope","torch"]
    df=df[[c for c in order if c in df.columns]]
    return _rename(df,{"gpu":"GPU","fused_bn":"BN fused","img_size":"resolution (px)","p50_ms":"p50 (ms)","p95_ms":"p95 (ms)",
                       "p99_ms":"p99 (ms)","mean_ms":"mean (ms)","images_per_s":"throughput (img/s)","warmup":"warmup iters",
                       "iterations":"measured iters","includes_preprocessing":"includes TTA preprocessing"})


def sheet_summary(backbones,training,inference,final,lock,latency,grade):
    sel_backbone=lock["backbone"]; b_row=backbones[backbones["weight tag (timm)"]==sel_backbone]
    b_lat=float(b_row["latency batch-1 p95 (ms)"].iloc[0]) if len(b_row) else np.nan
    b_par=float(b_row["params (M)"].iloc[0]) if len(b_row) else np.nan
    cands=[]
    for _,r in backbones.iterrows():
        cands.append({"exp_id":r.exp_id,"stage":"Bước 1 backbone","configuration":f"{r['weight tag (timm)']} + T00 + I00",
                      "val macro-F1":r["val macro-F1"],"val top-1":r["val top-1"],"latency batch-1 p95 (ms)":r["latency batch-1 p95 (ms)"],
                      "train time / epoch (s)":r["train time / epoch (s)"],"params (M)":r["params (M)"],"seeds":1})
    for _,r in training[training.seed==0].iterrows():
        cands.append({"exp_id":r.exp_id,"stage":"Bước 2 công thức","configuration":f"{sel_backbone} + {r['change vs T00']} + I00",
                      "val macro-F1":r["val macro-F1"],"val top-1":r["val top-1"],"latency batch-1 p95 (ms)":b_lat,
                      "train time / epoch (s)":r["train time / epoch (s)"],"params (M)":b_par,"seeds":1})
    for _,r in inference.iterrows():
        cands.append({"exp_id":r.exp_id,"stage":"Bước 3 suy luận","configuration":f"{r['model / checkpoint']} + {r['method']}",
                      "val macro-F1":r["val macro-F1"],"val top-1":r["val top-1"],"latency batch-1 p95 (ms)":r["latency batch-1 p95 (ms)"],
                      "relative cost vs I00 (p50)":r["relative cost vs I00 (p50)"],"params (M)":b_par,"seeds":1})
    i07=latency[latency.configuration.str.startswith("I07")&(latency.batch==1)]
    if "F01" in set(final.exp_id):
        cands.append({"exp_id":"F01","stage":"Bước 4 chung kết","configuration":final.loc[final.exp_id=="F01","configuration"].iloc[0],
                      "val macro-F1":_stat(final,"F01","val macro-F1"),"val top-1":_stat(final,"F01","val top-1"),
                      "latency batch-1 p95 (ms)":float(i07["p95 (ms)"].iloc[0]) if len(i07) else np.nan,"params (M)":b_par,
                      "seeds":int((final.exp_id=="F01").sum()-2)})
    top=pd.DataFrame(cands).sort_values("val macro-F1",ascending=False).head(10).reset_index(drop=True)
    top.insert(0,"rank",range(1,len(top)+1))
    top=top[["rank","exp_id","stage","configuration","seeds","val macro-F1","val top-1","latency batch-1 p95 (ms)",
             "relative cost vs I00 (p50)","train time / epoch (s)","params (M)"]]

    compare=[]
    for col in ("test macro-F1","test top-1","test balanced acc","test ECE (15 bins)","test recall Chinee Apple","test recall Snake Weed","val macro-F1"):
        fm,fs,bm,bs=(_stat(final,"F01",col),_stat(final,"F01",col,"std"),_stat(final,"T00",col),_stat(final,"T00",col,"std"))
        s=max(fs,bs); delta=fm-bm
        compare.append({"metric":col,"F01 mean":fm,"F01 std":fs,"T00 mean":bm,"T00 std":bs,"Δ (F01 − T00)":delta,
                        "s = max std":s,"Δ > s":bool(delta>s) if "ECE" not in col else bool(-delta>s)})
    facts=pd.DataFrame([
        {"item":"Cấu hình chung kết","value":final.loc[final.exp_id=="F01","configuration"].iloc[0] if "F01" in set(final.exp_id) else "—"},
        {"item":"Độ trễ cấu hình chung kết (I07, batch 1, p95)","value":f"{float(i07['p95 (ms)'].iloc[0]):.2f} ms" if len(i07) else "—"},
        {"item":"GPU đo độ trễ","value":str(latency.GPU.iloc[0]) if len(latency) else "—"},
        {"item":"Điểm tự chấm phần I (eval.py grade)","value":f"{grade['total']} / {grade['max_scored']}" if grade else "chưa chạy"},
    ])
    return top,pd.DataFrame(compare),facts


# --------------------------------------------------------------------------- workbook
def _style(ws,frame,start,title,highlight=()):
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    ws.cell(start,1,title).font=Font(bold=True,size=12)
    header=start+2
    for cell in ws[header][:len(frame.columns)]:
        font=copy(cell.font); font.bold=True; font.color="FFFFFF"; cell.font=font
        cell.fill=PatternFill("solid",fgColor="1F4E78")
    for idx,name in enumerate(frame.columns,1):
        letter=get_column_letter(idx)
        width=max(10,min(48,max(len(str(name))+2,*(len(_fmt(v))+2 for v in frame[name]))))
        ws.column_dimensions[letter].width=max(ws.column_dimensions[letter].width or 0,width)
        fmt="0.0000" if any(w in str(name) for w in SCORE_WORDS) else ("0.00" if any(w in str(name) for w in COST_WORDS) else None)
        for row in range(header+1,header+1+len(frame)):
            cell=ws.cell(row,idx)
            if fmt and isinstance(cell.value,(int,float)) and not isinstance(cell.value,bool): cell.number_format=fmt
    for i in highlight:
        for cell in ws[header+1+i][:len(frame.columns)]: cell.fill=PatternFill("solid",fgColor="C6EFCE"); cell.font=Font(bold=True)


def _best(frame,col,largest=True):
    s=pd.to_numeric(frame[col],errors="coerce")
    return [] if s.isna().all() else [int((s.idxmax() if largest else s.idxmin()))]


def write_workbook(path,sheets):
    """sheets: {name: [(title, frame, highlight_rows), ...]} — nhiều bảng xếp dọc trong một sheet."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with pd.ExcelWriter(path,engine="openpyxl") as writer:
        for name,blocks in sheets.items():
            start=1
            for title,frame,highlight in blocks:
                frame=frame.reset_index(drop=True)
                frame.to_excel(writer,sheet_name=name,index=False,startrow=start+1)
                _style(writer.book[name],frame,start,title,highlight); start+=len(frame)+5
            writer.book[name].freeze_panes="A4"


# --------------------------------------------------------------------------- checks
def run_checks(root,runs,backbones,training,inference,final,curves,lock):
    rows=[]; add=lambda check,ok,detail:rows.append({"check":check,"ok":bool(ok),"detail":detail})
    pred_dir=Path(root)/"predictions"; tol=1e-6
    expected={(r["exp_id"],r["seed"]) for r in runs}
    have={(e,int(s)) for e,s in zip(curves.exp_id,curves.seed)}
    add("Mọi lần huấn luyện B/T/F có ảnh curves/",expected==have,f"{len(have)}/{len(expected)} lần chạy")
    for sheet,frame in (("Backbones",backbones),("Training",training)):
        missing=frame[frame.curve=="THIẾU"].exp_id.tolist()
        add(f"{sheet}: mỗi dòng trỏ tới ảnh đường cong",not missing,", ".join(missing) or "đủ")
    by_key={(r["exp_id"],r["seed"]):r for r in runs}
    for sheet,frame in (("Backbones",backbones),("Training",training)):
        for _,r in frame.iterrows():
            key=(r.exp_id,int(r.seed)); path=pred_dir/f"{r.exp_id}_seed{int(r.seed)}_val.csv"
            if path.is_file():
                m=_pred_metrics(path)[1]["macro_f1"]
                add(f"{sheet} {r.exp_id} seed {int(r.seed)}: macro-F1 val = predictions/*_val.csv",abs(m-r["val macro-F1"])<tol,
                    f"sheet {r['val macro-F1']:.6f}, predictions {m:.6f}"+
                    (" — predictions đã bị ghi đè bởi lượt chạy sau (xem ghi chú T00)" if abs(m-r["val macro-F1"])>=tol and r.exp_id=="T00" else ""))
            elif key in by_key:
                m=by_key[key]["summary"]["val_macro_f1"]
                add(f"{sheet} {r.exp_id} seed {int(r.seed)}: macro-F1 val = summary.json",abs(m-r["val macro-F1"])<tol,f"sheet {r['val macro-F1']:.6f}, log {m:.6f}")
    for _,r in inference.iterrows():
        path=pred_dir/f"{r.exp_id}_seed0_val.csv"
        if path.is_file():
            m=_pred_metrics(path)[1]
            add(f"Inference {r.exp_id}: macro-F1/ECE val = predictions",abs(m["macro_f1"]-r["val macro-F1"])<tol and abs(m["ece"]-r["val ECE (15 bins)"])<tol,
                f"sheet {r['val macro-F1']:.6f}/{r['val ECE (15 bins)']:.6f}, predictions {m['macro_f1']:.6f}/{m['ece']:.6f}")
    src=lock.get("training_source")
    t_row=training[(training.exp_id==src)&(training.seed==0)]; i00=inference[inference.exp_id=="I00"]
    if len(t_row) and len(i00):
        a,b=float(t_row["val macro-F1"].iloc[0]),float(i00["val macro-F1"].iloc[0])
        add(f"Inference I00 = Training {src} seed 0",abs(a-b)<tol,f"{b:.6f} vs {a:.6f}")
    for tag in ("F01","T00","F01uncal"):
        seeds=final[(final.exp_id==tag)&final.seed.str.isdigit()]
        add(f"{tag}: ≥ 3 seed có predictions test",len(seeds)>=3,f"{len(seeds)} seed")
        if tag!="F01uncal" and len(seeds):
            add(f"{tag}: đủ 3507 ảnh test mỗi seed",(seeds["test images"]==3507).all(),str(seeds["test images"].tolist()))
        per_seed=Path(root)/"eval_out"/f"{tag}_per_seed.csv"
        if per_seed.is_file():
            ref=pd.read_csv(per_seed).set_index("seed")
            diffs=[max(abs(ref.loc[int(r.seed),"macro_f1"]-r["test macro-F1"]),abs(ref.loc[int(r.seed),"top1"]-r["test top-1"]),
                       abs(ref.loc[int(r.seed),"ece"]-r["test ECE (15 bins)"])) for _,r in seeds.iterrows()]
            add(f"Final {tag} khớp eval_out/{tag}_per_seed.csv (eval.py score)",max(diffs,default=0)<tol,f"sai lệch lớn nhất {max(diffs,default=0):.2e}")
        summary=_json(Path(root)/"eval_out"/f"{tag}_summary.json")
        if summary:
            diff=max(abs(summary[k]["mean"]-_stat(final,tag,c)) for k,c in (("macro_f1","test macro-F1"),("top1","test top-1")))
            add(f"Final {tag} mean khớp eval_out/{tag}_summary.json",diff<tol,f"sai lệch {diff:.2e}")
    for r in runs:
        if r["exp_id"]=="T00":
            row=training[(training.exp_id=="T00")&(training.seed==r["seed"])]
            if len(row):
                same=abs(float(row["val macro-F1"].iloc[0])-r["summary"]["val_macro_f1"])<tol
                add(f"T00 seed {r['seed']}: Training sheet = log hiện tại",same,
                    "khớp" if same else f"Training {float(row['val macro-F1'].iloc[0]):.6f} ≠ log {r['summary']['val_macro_f1']:.6f}: "
                    "Bước 4 đã huấn luyện lại T00 (config khác ở save_test_predictions/curve_label)")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def backbone_figures(backbones,out_dir):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); paths={}
    for col,name,label in (("latency batch-1 p50 (ms)","backbones_f1_latency.png","Latency batch-1 p50 (ms)"),
                           ("params (M)","backbones_f1_params.png","Parameters (M)")):
        fig,ax=plt.subplots(figsize=(7,4.5)); ax.scatter(backbones[col],backbones["val macro-F1"])
        for _,r in backbones.iterrows():
            ax.annotate(f"{r.exp_id} {r.architecture}",(r[col],r["val macro-F1"]),xytext=(4,4),textcoords="offset points",fontsize=8)
        ax.set(xlabel=label,ylabel="Validation macro-F1",title=f"Backbones (T00, seed 0): macro-F1 vs {label.split(' (')[0].lower()}")
        ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out_dir/name,dpi=160); plt.close(fig); paths[col]=f"figures/{name}"
    return paths


def _confusion_pairs(root,tag="F01",top=3):
    path=Path(root)/"eval_out"/f"{tag}_confusion_sum.csv"
    if not path.is_file(): return pd.DataFrame()
    cm=pd.read_csv(path,index_col=0); names=[c.replace("pred_","") for c in cm.columns]; values=cm.to_numpy()
    rows=[{"true":names[i],"predicted":names[j],"count (sum over seeds)":int(values[i,j]),"share of true class":values[i,j]/values[i].sum()}
          for i in range(len(names)) for j in range(len(names)) if i!=j and values[i,j]>0]
    return pd.DataFrame(rows).sort_values("count (sum over seeds)",ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- report
def _versions():
    out={"python":platform.python_version()}
    for mod in ("torch","timm","numpy","pandas","sklearn"):
        try: out[mod]=__import__(mod).__version__
        except Exception: out[mod]="?"
    return out


def write_report(path,ctx):
    c=ctx; bb=c["backbones"]; tr=c["training"]; inf=c["inference"]; fin=c["final"]; cmp_=c["compare"].set_index("metric")
    lock=c["lock"]; sel_b=c["backbone_selection"]; sel_t=c["training_selection"]; sel_i=c["inference_selection"]
    noise=float(tr["T00 macro-F1 std (3 seed)"].dropna().iloc[0]) if tr["T00 macro-F1 std (3 seed)"].notna().any() else np.nan
    t0=tr[tr.seed==0]; tbase=float(t0.loc[t0.exp_id=="T00","val macro-F1"].iloc[0])
    best_b=bb.loc[bb["val macro-F1"].idxmax()]; resnet=bb[bb.architecture.str.startswith("resnet")]
    best_t=t0[t0.exp_id!="T00"].sort_values("val macro-F1",ascending=False).iloc[0]
    best_i=inf.loc[inf["val macro-F1"].idxmax()]; i00=inf[inf.exp_id=="I00"].iloc[0]
    f=lambda col,stat="mean":_stat(fin,"F01",col,stat); b=lambda col,stat="mean":_stat(fin,"T00",col,stat)
    d=cmp_.loc["test macro-F1"]; verdict=("vượt nhiễu (Δ > s)" if d["Δ > s"] else "không phân biệt được với nhiễu (Δ ≤ s)")
    lat=c["latency"]; i07=lat[lat.configuration.str.startswith("I07")&(lat.batch==1)]
    p95_final=float(i07["p95 (ms)"].iloc[0]) if len(i07) else np.nan
    gpu=str(lat.GPU.iloc[0]) if len(lat) else "?"
    curves=c["curves"]; bcur=curves[curves.exp_id.str.startswith("B")]
    fastest=bcur.sort_values("first epoch ≥ 98% best macro-F1").iloc[0]
    overfit=bcur[bcur["overfit flag"]].exp_id.tolist()
    corr=lambda a,b_:float(pd.to_numeric(bb[a]).corr(pd.to_numeric(bb[b_]))) if len(bb)>2 else np.nan
    rank_corr=float(pd.to_numeric(bb["GMAC"]).rank().corr(pd.to_numeric(bb["val macro-F1"]).rank())) if len(bb)>2 else np.nan
    contrib={"backbone":float(best_b["val macro-F1"]-resnet["val macro-F1"].iloc[0]) if len(resnet) else np.nan,
             "công thức huấn luyện":float(best_t["val macro-F1"]-tbase),"suy luận":float(best_i["val macro-F1"]-i00["val macro-F1"])}
    top_factor=max(contrib,key=lambda k:-np.inf if np.isnan(contrib[k]) else contrib[k])
    temps=", ".join(f"{t:.3f}" for t in c["temperatures"]) or "—"
    gap=abs(f("val macro-F1")-f("test macro-F1"))
    t00_cfg=c["t00_config"]; v=c["versions"]; pairs=c["pairs"]
    split=c["split"] or {}; n=split.get("n",{})
    rt=sel_i.get("realtime_method"); rt_row=inf[inf.exp_id==rt]
    lines=[]; L=lines.append

    L("# Báo cáo Lab Day 2 — DeepWeeds: backbone, công thức huấn luyện và suy luận\n")
    L("Mọi con số dưới đây được sinh tự động từ `results.xlsx` (tên sheet ghi trong ngoặc), các file log trong `logs/` "
      "và `predictions/`; chỉ số test được tính lại bằng `eval.py`. Kết quả sàng lọc ở Bước 1–3 là **1 seed**.\n")

    L("## 1. Tóm tắt\n")
    L(f"- Bài toán: phân loại 9 lớp DeepWeeds (fold 0), chỉ số chính macro-F1.")
    L(f"- Đã chạy {len(bb)} backbone (Bước 1), {t0.exp_id.nunique()-1} biến thể công thức + T00 ×3 seed (Bước 2), "
      f"{len(inf)} cấu hình suy luận (Bước 3), chung kết F01 và mốc T00 + I00 mỗi cấu hình 3 seed (Bước 4).")
    L(f"- Cấu hình tốt nhất: **{fin.loc[fin.exp_id=='F01','configuration'].iloc[0]}** (sheet Final).")
    L(f"- Test (3 seed): macro-F1 **{_pm(f('test macro-F1'),f('test macro-F1','std'))}**, top-1 **{_pm(f('test top-1'),f('test top-1','std'))}**; "
      f"mốc T00 + I00: macro-F1 {_pm(b('test macro-F1'),b('test macro-F1','std'))}.")
    L(f"- Cải thiện macro-F1 test so với mốc: Δ = {d['Δ (F01 − T00)']:+.4f}, s = {d['s = max std']:.4f} → **{verdict}** (sheet Summary).")
    L(f"- Độ trễ cấu hình chung kết: p95 = {p95_final:.2f} ms ở batch 1 trên {gpu} (sheet Latency).\n")

    L("## 2. Dữ liệu và thiết lập\n")
    L(f"- DeepWeeds, 17.509 ảnh RGB 256×256, 9 lớp; fold 0 chính chủ (`train/val/test_subset0.csv`), không sửa CSV. "
      f"Số ảnh: train {n.get('train','?')}, val {n.get('val','?')}, test {n.get('test','?')}; ba giao rỗng, hợp đủ 17.509 (xem `step0_report.md`).")
    L("- Mất cân bằng: `Negative` ≈ 52%, tỉ lệ lớn nhất/nhỏ nhất ≈ 9,0 → dùng macro-F1 để chọn mô hình.\n")
    L("![Phân bố lớp](eda/class_distribution.png)\n")
    L("- Chỉ số: macro-F1 (chính), top-1, balanced accuracy, F1/recall theo lớp, ECE 15 bin — đúng định nghĩa `eval.py`.")
    L("- Val dùng cho mọi lựa chọn (backbone, công thức, suy luận, checkpoint, nhiệt độ T); test mở **một lần mỗi seed** ở Bước 4, "
      "cấu hình đã khóa trước trong `logs/final_lock.json`.")
    keys=["epochs","batch_size","grad_accum_steps","lr_backbone","lr_head","weight_decay","warmup_epochs","img_size","aug","loss","amp"]
    L("\nCông thức nền T00 (từ `logs/T00/seed0/config.json`):\n")
    L(_md(pd.DataFrame([{"tham số":k,"giá trị":str(t00_cfg.get(k,"—"))} for k in keys])))
    L("\nAdamW, weight decay không áp dụng cho norm/bias, head mới LR ×10, warmup 1 epoch rồi cosine, chọn checkpoint theo macro-F1 val "
      "(hòa lấy epoch sớm hơn). Train: RandomResizedCrop(224, scale 0,7–1) + lật ngang; val/test: resize 256 → CenterCrop 224; chuẩn hoá ImageNet.")
    L(f"\nPhần cứng đo độ trễ: {gpu}. Phiên bản (môi trường chạy Bước 5): " + ", ".join(f"{k} {val}" for k,val in v.items()) + ".")
    L("Seed: 0 cho sàng lọc/ablation; 0, 1, 2 cho T00 (đo nhiễu) và chung kết.\n")

    L("## 3. So sánh backbone (sheet Backbones)\n")
    cols=["exp_id","weight tag (timm)","params (M)","GMAC","val macro-F1","val top-1","train time / epoch (s)","latency batch-1 p50 (ms)","best epoch"]
    L(_md(bb[cols]))
    L(f"\n![F1 theo độ trễ]({c['fig_paths']['latency batch-1 p50 (ms)']})  ![F1 theo tham số]({c['fig_paths']['params (M)']})\n")
    L(f"- Macro-F1 val cao nhất: **{best_b.exp_id} ({best_b['weight tag (timm)']})** = {best_b['val macro-F1']:.4f}.")
    L(f"- Chọn đi tiếp: `{sel_b.get('best_quality_backbone')}` (macro-F1 cao nhất); cấu hình cân bằng theo quy tắc "
      f"\"nhanh nhất trong phạm vi 0,02 macro-F1\": `{sel_b.get('balanced_backbone')}`.")
    L(f"- Hội tụ nhanh nhất (epoch đầu tiên đạt ≥ 98% macro-F1 tốt nhất của chính nó): **{fastest.exp_id}**, epoch {fastest['first epoch ≥ 98% best macro-F1']}.")
    L(f"- Dấu hiệu quá khớp (val loss cuối > 1,05 × val loss nhỏ nhất trong khi train loss vẫn giảm): {', '.join(overfit) or 'không có'}.")
    L(f"- Tương quan Pearson GMAC–độ trễ p50: {corr('GMAC','latency batch-1 p50 (ms)'):.3f}; GMAC–thời gian train/epoch: "
      f"{corr('GMAC','train time / epoch (s)'):.3f}. FLOPs chỉ là chỉ báo thô của tốc độ (slide trang 43): độ trễ còn phụ thuộc kernel, "
      "truy cập bộ nhớ và depthwise/attention.")
    L(f"- Tương quan hạng Spearman GMAC–macro-F1 val: {rank_corr:.3f}. Bảng không ghi top-1 ImageNet của từng tag, nên không khẳng định "
      "thứ hạng trên DeepWeeds trùng thứ hạng ImageNet; các tag còn khác nhau về công thức tiền huấn luyện (slide trang 45).")
    L("- Một seed: chênh lệch nhỏ hơn std của T00 (mục 4) không được coi là khác biệt thật.\n")

    L("## 4. Công thức huấn luyện (sheet Training)\n")
    L(f"Backbone `{sel_t.get('backbone')}`. Mỗi biến thể khác T00 đúng một yếu tố (seed 0); T00 chạy 3 seed để lấy ngưỡng nhiễu "
      f"std = **{noise:.4f}**. T11 kết hợp các yếu tố tốt nhất theo trục (cách tham lam theo trục; thứ tự trục có thể ảnh hưởng).\n")
    cols=["exp_id","axis (A–G)","change vs T00","seed","val macro-F1","Δ macro-F1 vs T00 seed 0","Δ vs noise","val F1 Chinee Apple","val F1 Snake Weed"]
    L(_md(tr[cols]))
    L("")
    for axis in ("A","B","C","F"):
        g=t0[t0["axis (A–G)"]==axis].sort_values("val macro-F1",ascending=False)
        if len(g): r=g.iloc[0]; L(f"- Trục {axis}: tốt nhất `{r.exp_id}` ({r['change vs T00']}), Δ = {r['Δ macro-F1 vs T00 seed 0']:+.4f} → {r['Δ vs noise']}.")
    t11=t0[t0.exp_id=="T11"]
    if len(t11):
        L(f"- Kết hợp T11 ({' + '.join(sel_t.get('combo_components',[]))}): Δ = {float(t11['Δ macro-F1 vs T00 seed 0'].iloc[0]):+.4f}; "
          f"hiệu ứng: {sel_t.get('combo_interaction','—')}.")
    L(f"- Công thức chọn cho Bước 3–4: `{sel_t.get('exp_id')}` (macro-F1 val {sel_t.get('val_macro_f1',np.nan):.4f}).")
    L("- Δ của một seed được so với std của T00 qua 3 seed; |Δ| ≤ std được ghi là \"không phân biệt được\".\n")

    L("## 5. Suy luận (sheets Inference, Latency)\n")
    cols=["exp_id","method","K (views or models)","val macro-F1","Δ macro-F1 vs I00","val ECE (15 bins)","latency batch-1 p50 (ms)",
          "latency batch-1 p95 (ms)","throughput batch-32 (img/s)","relative cost vs I00 (p50)"]
    L(_md(inf[[x for x in cols if x in inf.columns]]))
    L("\n![Đánh đổi độ chính xác – độ trễ](figures/I_accuracy_latency.png)\n")
    i07r=inf[inf.exp_id=="I07"]
    if len(i07r):
        L(f"- Hiệu chuẩn: ECE val {float(i00['val ECE (15 bins)']):.4f} → {float(i07r['val ECE (15 bins)'].iloc[0]):.4f} sau temperature scaling "
          f"(T khớp trên val); accuracy không đổi vì argmax không đổi. Trên test (3 seed): ECE {_pm(_stat(fin,'F01uncal','test ECE (15 bins)'),_stat(fin,'F01uncal','test ECE (15 bins)','std'))} "
          f"→ {_pm(f('test ECE (15 bins)'),f('test ECE (15 bins)','std'))}.")
    L(f"- Macro-F1 val cao nhất: `{best_i.exp_id}` ({best_i['method']}), Δ = {best_i['Δ macro-F1 vs I00']:+.4f} so với I00, "
      f"chi phí ×{best_i['relative cost vs I00 (p50)']:.2f}. Δ này là 1 seed trên val; so với std T00 = {noise:.4f} → "
      f"{'vượt nhiễu' if best_i['Δ macro-F1 vs I00']>noise else 'không phân biệt được'}.")
    L(f"- Thời gian thực (p95 ≤ 100 ms): `{rt or 'không có'}`"+(f", p95 = {float(rt_row['latency batch-1 p95 (ms)'].iloc[0]):.2f} ms." if len(rt_row) else "."))
    L("- TTA/ensemble tốn gần K lần chi phí nên phù hợp suy luận ngoại tuyến; temperature scaling, AMP và gộp BN không tăng (hoặc giảm) "
      "chi phí nên phù hợp robot (slide trang 63, 67, 71).")
    L(f"- Gộp Conv-BN: {json.dumps(c['bn_fusion'],ensure_ascii=False)}.\n")
    L("Điều kiện đo: warmup 10 lần, 50 lần đo, `torch.cuda.synchronize()` trước/sau, đầu vào tensor trên GPU (không tính giải mã ảnh), "
      "batch 1 và 32; chi tiết từng dòng ở sheet Latency.\n")

    L("## 6. Cấu hình tốt nhất (sheets Final, PerClass)\n")
    L(f"- Tái lập: backbone `{lock['backbone']}`, công thức `{lock.get('training_source')}` "
      f"{json.dumps(lock.get('training_overrides') or {})} trên nền T00, {lock.get('epochs')} epoch, batch {lock.get('batch_size')}, "
      f"seed {lock.get('seeds')}; suy luận 1-view + temperature scaling, T khớp riêng trên val mỗi seed (T = {temps}).")
    show=fin[["exp_id","seed","val macro-F1","test macro-F1","test top-1","test ECE (15 bins)","test recall Chinee Apple","test recall Snake Weed"]]
    L(""); L(_md(show)); L("")
    pc=c["per_class"]; hard=pc[pc["class"].isin(HARD)]
    L(_md(hard[["configuration","class","test images","precision mean","recall mean","recall std","F1 mean","F1 std"]])); L("")
    L(f"- So với bài báo (trích dẫn, 100 epoch, augmentation mạnh; định nghĩa accuracy khác): top-1 test {f('test top-1'):.4f} "
      f"so với ResNet-50 {PAPER['resnet50_top1']:.3f} / Inception-v3 {PAPER['inception_top1']:.3f}; recall Chinee Apple "
      f"{f('test recall Chinee Apple'):.4f} (bài báo {PAPER['Chinee Apple']:.3f}), Snake Weed {f('test recall Snake Weed'):.4f} (bài báo {PAPER['Snake Weed']:.3f}).")
    L(f"- Chênh lệch macro-F1 val–test của F01: {gap:.4f} ({'≤' if gap<=0.02 else '>'} 0,02).\n")
    L("![Ma trận nhầm lẫn](figures/F01_confusion_matrix.png)\n")
    if len(pairs):
        L("Các cặp nhầm nhiều nhất (cộng 3 seed, `eval/F01_confusion_sum.csv`):\n"); L(_md(pairs)); L("")
    L("![Ảnh bị đoán sai](figures/F01_misclassified_examples.png)\n")
    L("Giả thuyết: các cặp nhầm có hình thái lá và nền thực địa tương tự (bụi lá bầu dục, đất và cỏ khô), đối tượng nhỏ hoặc bị che, "
      "ánh sáng thay đổi mạnh; `Negative` rất không đồng nhất nên các loài xuất hiện nhỏ trong khung dễ bị đoán thành `Negative`. "
      "Đối chiếu trực tiếp với ảnh ở hình trên.\n")

    L("## 7. Kết luận và khuyến nghị\n")
    L(f"- Cấu hình tốt nhất: F01 (mục 6). So với mốc T00 + I00 trên test: macro-F1 Δ = {d['Δ (F01 − T00)']:+.4f} với s = {d['s = max std']:.4f} → {verdict}; "
      f"top-1 Δ = {cmp_.loc['test top-1','Δ (F01 − T00)']:+.4f}.")
    L(f"- Đóng góp ước lượng trên val (1 seed, so với mốc của từng bước): backbone {contrib['backbone']:+.4f} (tốt nhất so với ResNet), "
      f"công thức huấn luyện {contrib['công thức huấn luyện']:+.4f} (tốt nhất so với T00), suy luận {contrib['suy luận']:+.4f} (tốt nhất so với I00). "
      f"Lớn nhất: **{top_factor}**; mọi mức nhỏ hơn std T00 = {noise:.4f} là không phân biệt được.")
    L(f"- Robot, ngân sách 30–100 ms/khung: dùng cấu hình chung kết 1-view + temperature scaling (p95 = {p95_final:.2f} ms trên {gpu}"
      f"{', vừa cả ngân sách 30 ms' if p95_final<=30 else (', trong ngân sách 100 ms' if p95_final<=100 else ', vượt 100 ms')}); "
      "không dùng TTA/ensemble trên robot vì tốn K lần độ trễ. Phần cứng robot (Jetson) chậm hơn GPU đo nên cần đo lại trên thiết bị đích.\n")

    L("## 8. Hạn chế và việc tiếp theo\n")
    L("- Sàng lọc backbone, ablation công thức và so sánh suy luận chỉ 1 seed; chỉ chung kết và T00 có 3 seed. Một fold (fold 0).")
    L("- Dữ liệu chia ngẫu nhiên, không theo địa điểm chụp, nên điểm test có thể **lạc quan** khi robot gặp địa điểm mới.")
    L(f"- Ngân sách GPU: {t00_cfg.get('epochs','?')} epoch thay vì ~100 epoch như bài báo; augmentation nhẹ hơn.")
    if not c["checks"].ok.all():
        L("- Kiểm tra nhất quán còn mục chưa đạt (Phụ lục B); các con số liên quan cần đọc kèm ghi chú đó.")
    L("- Rủi ro lệch phân phối: mùa, ánh sáng, góc chụp, địa hình khác; T khớp trên val có thể không còn đúng khi miền thay đổi.")
    L("- Việc tiếp theo: chạy thêm fold, thêm seed cho các ablation sát ngưỡng nhiễu, chưng cất từ mô hình lớn sang mạng nhẹ, đo trên Jetson.\n")

    L("## 9. Phụ lục\n")
    L(f"- Notebook: [{COLAB_URL}]({COLAB_URL}) (`code/lab_day2.ipynb`). Log từng lần chạy: `logs/<exp_id>/seed<k>/` (config, history, summary).\n")
    L("### A. Danh sách lần huấn luyện\n"); L(_md(c["appendix"])); L("")
    L("### B. Kiểm tra nhất quán (sheet Checks)\n"); L(_md(c["checks"])); L("")
    if c["grade"]:
        L("### C. Tự chấm phần I (`eval.py grade`)\n")
        L(_md(pd.DataFrame(c["grade"]["items"])[["code","criterion","points","max","note"]]))
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")


# --------------------------------------------------------------------------- entry points (notebook gọi từng bước)
def load_artifacts(artifact_root):
    """Đọc mọi thứ Bước 1–4 đã lưu trong ARTIFACT_ROOT."""
    root=Path(artifact_root); runs=_runs(root)
    lock=_json(root/"final_lock.json")
    if lock is None: raise FileNotFoundError("Thiếu final_lock.json: chạy Bước 4 trước")
    return {"root":root,"runs":runs,"runs_by_key":{(r["exp_id"],r["seed"]):r for r in runs},"lock":lock,
            "backbone_selection":_json(root/"backbone_selection.json",{}),"training_selection":_json(root/"training_selection.json",{}),
            "inference_selection":_json(root/"inference_selection.json",{}),"grade":_json(root/"eval_out"/"grade_I.json")}


def build_tables(art):
    """Mọi bảng của results.xlsx dưới dạng DataFrame (chưa ghi file)."""
    root,rk=art["root"],art["runs_by_key"]
    t={"backbones":sheet_backbones(root,rk,art["backbone_selection"]),"training":sheet_training(root,rk,art["training_selection"]),
       "inference":sheet_inference(root),"latency":sheet_latency(root)}
    t["final"],t["per_class"],_=final_tables(root,art["lock"])
    t["top"],t["compare"],t["facts"]=sheet_summary(t["backbones"],t["training"],t["inference"],t["final"],art["lock"],t["latency"],art["grade"])
    return t


def workbook_sheets(t,curves,checks):
    """Bố cục sheet: tiêu đề, bảng và dòng tô nổi bật (tốt nhất) của từng sheet."""
    per_class,final,latency,top=t["per_class"],t["final"],t["latency"],t["top"]
    hard_rows=[i for i,(cfg,cls) in enumerate(zip(per_class.configuration,per_class["class"])) if cfg.startswith("F01") and cls in HARD]
    batch1=latency[latency.batch==1]
    return {
        "Backbones":[("Bước 1 — backbone, cùng công thức T00, seed 0 (1 seed)",t["backbones"],_best(t["backbones"],"val macro-F1"))],
        "Training":[("Bước 2 — ablation công thức (seed 0; T00 có 3 seed để đo nhiễu)",t["training"],
                     _best(t["training"][t["training"].seed==0].reindex(t["training"].index),"val macro-F1"))],
        "Inference":[("Bước 3 — phương pháp suy luận trên val (checkpoint seed 0)",t["inference"],_best(t["inference"],"val macro-F1"))],
        "Final":[("Bước 4 — test một lần mỗi seed; dòng mean/std tính với ddof=1",final,
                  [i for i,(e,s) in enumerate(zip(final.exp_id,final.seed)) if e=="F01" and s.startswith("mean")])],
        "PerClass":[("Theo lớp trên test, mean và std qua seed",per_class,hard_rows)],
        "Latency":[("Độ trễ: warmup 10, 50 lần đo, cuda.synchronize, đầu vào trên GPU",latency,
                    [int(batch1["p95 (ms)"].idxmin())] if len(batch1) else [])],
        "Summary":[("Top 10 cấu hình theo macro-F1 val",top,[i for i,e in enumerate(top.exp_id) if e=="F01"] or [0]),
                   ("Chung kết F01 so với mốc T00 + I00 trên test (mean và std qua seed)",t["compare"],[0]),
                   ("Thông tin chính",t["facts"],[])],
        "Curves":[("Ảnh đường cong của từng lần huấn luyện",curves,[])],
        "Checks":[("Kiểm tra nhất quán giữa sheet, log, predictions và eval.py",checks,[])],
    }


def copy_deliverables(art,submission_dir):
    """Chép file nộp nhỏ: predictions chung kết/mốc, log từng lần chạy, kết quả eval.py, hình cho báo cáo."""
    root,sub=art["root"],Path(submission_dir)
    for pattern in ("F01_seed*_test.csv","F01_seed*_val.csv","F01uncal_seed*_*.csv","T00_seed*_test.csv","T00_seed*_val.csv"):
        for src in (root/"predictions").glob(pattern): _copy(src,sub/"predictions"/src.name)
    for r in art["runs"]:
        for name in ("config.json","history.csv","summary.json"): _copy(r["dir"]/name,sub/"logs"/r["exp_id"]/f"seed{r['seed']}"/name)
    for name in ("backbones.csv","training.csv","inference.csv","latency.csv","backbone_selection.json","training_selection.json",
                 "inference_selection.json","final_lock.json","bn_fusion_check.json","backbone_analysis.md","training_analysis.md",
                 "inference_analysis.md","final_analysis.md"):
        _copy(root/name,sub/"logs"/name)
    for src in (root/"eval_out").glob("*"): _copy(src,sub/"eval"/src.name)
    for name in ("I_accuracy_latency.png","F01_confusion_matrix.png","F01_misclassified_examples.png"):
        _copy(root/"curves"/name,sub/"figures"/name)


def report_context(art,t,curves,checks,fig_paths):
    t00=art["runs_by_key"].get(("T00",0)); split=_json(t00["dir"]/"split_report.json") if t00 else None
    appendix=pd.DataFrame([{"exp_id":r["exp_id"],"seed":r["seed"],"backbone":r["config"].get("backbone"),"init":r["config"].get("init"),
                            "aug":r["config"].get("aug"),"mix":r["config"].get("mix"),"loss":r["config"].get("loss"),
                            "ema":r["config"].get("ema_decay"),"epochs":r["config"].get("epochs"),
                            "batch":f"{r['config'].get('batch_size')}×{r['config'].get('grad_accum_steps',1)}",
                            "val macro-F1":r["summary"]["val_macro_f1"],"curve":f"curves/{_curve_name(r)}"} for r in art["runs"]])
    return {**{k:t[k] for k in ("backbones","training","inference","latency","final","per_class","compare")},
            "lock":art["lock"],"backbone_selection":art["backbone_selection"],"training_selection":art["training_selection"],
            "inference_selection":art["inference_selection"],"grade":art["grade"],"curves":curves,"checks":checks,"fig_paths":fig_paths,
            "pairs":_confusion_pairs(art["root"]),"bn_fusion":_json(art["root"]/"bn_fusion_check.json",{}),
            "temperatures":_temperatures(art["root"]),"t00_config":(t00 or {}).get("config",{}),"versions":_versions(),
            "split":split,"appendix":appendix}


def build_deliverables(artifact_root,submission_dir,labels_dir="data/labels"):
    """Chạy cả Bước 5 trong một lần gọi (notebook làm cùng các bước này, từng ô một)."""
    sub=Path(submission_dir); art=load_artifacts(artifact_root)
    curves=build_curves(art["runs"],sub/"curves"); t=build_tables(art)
    checks=run_checks(art["root"],art["runs"],t["backbones"],t["training"],t["inference"],t["final"],curves,art["lock"])
    write_workbook(sub/"results.xlsx",workbook_sheets(t,curves,checks))
    copy_deliverables(art,sub); fig_paths=backbone_figures(t["backbones"],sub/"figures")
    write_report(sub/"report.md",report_context(art,t,curves,checks,fig_paths))
    failed=checks[~checks.ok]
    if len(failed): print(f"CẢNH BÁO: {len(failed)} kiểm tra nhất quán chưa đạt (sheet Checks):\n"+failed.to_string(index=False))
    else: print(f"Đạt cả {len(checks)} kiểm tra nhất quán.")
    return {"results_xlsx":str(sub/"results.xlsx"),"report":str(sub/"report.md"),"checks":checks,"summary":t["top"],
            "compare":t["compare"],"curves":curves}


def _temperatures(root):
    """T của từng seed chung kết, khớp lại trên val logits đã lưu (chỉ dùng val, giống hệt Bước 4)."""
    import inference as inf_utils
    out=[]
    for logits_path in sorted((Path(root)/"runs"/"F01").glob("seed*/val_logits.npy")):
        seed=logits_path.parent.name; val_csv=Path(root)/"predictions"/f"F01_{seed}_val.csv"
        if val_csv.is_file():
            out.append(float(inf_utils.fit_temperature(np.load(logits_path),ev.read_pred(str(val_csv)).y_true)))
    return out
