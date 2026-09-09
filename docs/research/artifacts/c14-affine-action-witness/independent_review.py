#!/usr/bin/env python3
"""C14 independent augmented-least-squares and scoring review; no model imports."""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import time
import numpy as np

FIT = (0,2,4,5,7,8,9,10,13,14,15,16,18,19,21,23)
MAPS = {"seen": FIT, "familiar": FIT, "heldout": (1,3,6,11,12,17,20,22)}
PERMS = tuple(itertools.permutations(range(4)))
ARMS = ("final_true", "initial_true", "final_permuted")
EXTRAS = {"logits", "attention", "pooled", "checkpointstage"}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def decode(raw):
    def pairs(items):
        d = {}
        for k,v in items:
            require(k not in d, "duplicate JSON key")
            d[k] = v
        return d
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: require(False,"nonfinite JSON"))

def same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False) == json.dumps(b,sort_keys=True,allow_nan=False)

def pinned(record):
    p = Path(record["path"])
    require(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size <= 128*1024**2,"bound file")
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == record["sha256"],"file hash mismatch: "+str(p))
    return raw

def verify_frozen(path,sha):
    p=Path(path)
    require(p.is_absolute() and p.is_file() and not p.is_symlink(),"frozen file type")
    with p.open("rb") as handle:
        require(hashlib.file_digest(handle,"sha256").hexdigest()==sha,"frozen file hash mismatch: "+path)

def numbers(value,shape):
    def typed(x):
        return all(typed(y) for y in x) if type(x) is list else type(x) in (int,float)
    require(typed(value),"numeric scalar type")
    x = np.asarray(value,dtype=float)
    require(x.shape == shape and np.isfinite(x).all(),"numeric shape/finiteness")
    return x

def geometry(row,cohort,i):
    m = len(MAPS[cohort]); group,slot = divmod(i,m)
    require(type(row["row_index"]) is int and row["row_index"] == i and type(row["group_index"]) is int and row["group_index"] == group,"row/group order")
    require(type(row["permutation_id"]) is int and row["permutation_id"] == MAPS[cohort][slot],"map order")
    require(row["schema"] == "looped-grounded-policy-data-v1" and row["cohort"] == cohort and row["condition"] == "factual" and row["support_cleared"] is False,"data schema/cohort")
    cells = row["query_cells"]
    require(len(cells) == 64 and all(type(c) is int and 0 <= c <= 3 for c in cells) and cells.count(2) == cells.count(3) == 1,"visible cells")
    a,g = cells.index(2),cells.index(3); delta = (g//8-a//8,g%8-a%8)
    directions = ((-1,0),(1,0),(0,-1),(0,1)); require(delta in directions,"adjacent roles")
    label = PERMS[row["permutation_id"]].index(directions.index(delta))
    require(all(type(row[k]) is int for k in ("agent_patch","goal_patch","correct_action")),"role/label integer")
    require((a,g,label) == (row["agent_patch"],row["goal_patch"],row["correct_action"]),"independent geometry/label")
    require(same(row["inferred_controls"],list(PERMS[row["permutation_id"]])),"audited bijection")
    actions = row["observed_support_action_ids"]
    require(len(actions) == len(set(actions)) == 3 and all(type(x) is int and 0 <= x < 4 for x in actions),"support actions")
    omit = next(iter(set(range(4))-set(actions))); require(omit == row["omitted_action"],"omitted action")
    return label,omit

def population(raw,audit,cohort,stage):
    rows = [decode(line) for line in raw.splitlines()]; m = len(MAPS[cohort])
    require(len(rows) == len(audit) == 64*m,"complete cohort")
    x,labels,omitted,native = [],[],[],[]
    for i,(row,identity) in enumerate(zip(rows,audit)):
        require(set(row) == set(identity)|EXTRAS and same(identity,{k:row[k] for k in identity}),"audit/output fields")
        require(row["checkpointstage"] == ("frozen" if stage == "initial" else "final"),"checkpoint stage")
        label,omit = geometry(identity,cohort,i)
        x.append(numbers(row["pooled"],(256,))); z = numbers(row["logits"],(4,))
        labels.append(label); omitted.append(omit); native.append(max(range(4),key=z.__getitem__))
    y = np.asarray(labels)
    require(all(np.all((y == a).reshape(64,m).sum(1) == m//4) for a in range(4)),"group action balance")
    return np.asarray(x),y,np.asarray(omitted),np.asarray(native)

def permutation_indices():
    generator = np.random.Generator(np.random.PCG64(1941))
    return np.concatenate([start+generator.permutation(16) for start in range(0,1024,16)])

def permute(labels):
    require(labels.shape == (1024,),"fit label count")
    return labels[permutation_indices()]

def fit(x,labels):
    n,d = x.shape
    require(n == 1024 and d == 256 and np.isfinite(x).all() and labels.shape == (n,) and np.all((labels >= 0)&(labels < 4)),"fit inputs")
    mean = np.array([math.fsum(float(v) for v in x[:,j])/n for j in range(d)])
    std = np.array([math.sqrt(math.fsum((float(v)-mean[j])**2 for v in x[:,j])/n) for j in range(d)])
    scale = np.maximum(std,1e-6); z = (x-mean)/scale
    target = np.eye(4)[labels]; intercept = target.mean(0); centered = target-intercept
    design = np.vstack((z/math.sqrt(n),math.sqrt(.01)*np.eye(d)))
    responses = np.vstack((centered/math.sqrt(n),np.zeros((d,4))))
    coefficients,_,rank,_ = np.linalg.lstsq(design,responses,rcond=None)
    require(rank == d,"augmented design rank")
    rhs = z.T@centered/n; normal = z.T@z/n + .01*np.eye(d)
    if not np.any(rhs): coefficients = np.zeros((d,4))
    residual = np.linalg.norm(normal@coefficients-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)
    require(np.isfinite(coefficients).all() and residual <= 1e-9,"independent normal residual")
    objective = float(np.sum((z@coefficients+intercept-target)**2)/n + .01*np.sum(coefficients**2))
    return dict(mean=mean,std=std,scale=scale,intercept=intercept,coefficients=coefficients),dict(
        objective=objective,relative_residual=float(residual),condition_number=float(np.linalg.cond(normal)),
        std_quantiles=np.quantile(std,[0,.01,.1,.5,.9,.99,1]).tolist(),
        scale_quantiles=np.quantile(scale,[0,.01,.1,.5,.9,.99,1]).tolist(),clamped_dimensions=int((std<1e-6).sum()))

def scores(model,x):
    out = ((x-model["mean"])/model["scale"])@model["coefficients"]+model["intercept"]
    require(np.isfinite(out).all(),"nonfinite affine scores")
    return out

def close(a,b,where):
    require(np.asarray(a).shape == np.asarray(b).shape and np.isfinite(b).all(),"parameter shape/finiteness "+where)
    require(np.all(abs(a-b) <= 1e-9+1e-7*abs(b)),"least-squares agreement "+where)

def summarize(pred,labels,omitted,m):
    hit=[int(p==y) for p,y in zip(pred,labels)];subsets={}
    for name,wanted in (("omitted",True),("demonstrated",False)):
        selected=[i for i,(y,o) in enumerate(zip(labels,omitted)) if bool(y==o)==wanted]
        require(selected,"empty subset");correct=sum(hit[i] for i in selected)
        subsets[name]=dict(rows=len(selected),correct=correct,accuracy=correct/len(selected))
    return dict(rows=len(labels),groups=64,maps_per_group=m,correct=sum(hit),accuracy=sum(hit)/len(labels),
                label_histogram=[sum(int(y==a) for y in labels) for a in range(4)],
                prediction_histogram=[sum(int(p==a) for p in pred) for a in range(4)],
                all_maps_correct_groups=sum(all(hit[start:start+m]) for start in range(0,len(hit),m)),subsets=subsets),np.asarray(hit).reshape(64,m).mean(1)

def interval(values,weights):
    ordered = np.sort(weights@values/64)
    def q(p):
        pos=9999*p; k=int(pos)
        return float(ordered[k]+(pos-k)*(ordered[k+1]-ordered[k]))
    return dict(estimate=float(values.mean()),ci95=[q(.025),q(.975)])

def gates_and_decision(summaries,contrasts,controls):
    gates = {}
    for arm in ("final_true","initial_true"):
        g = {c+"_accuracy":summaries[arm+"/"+c]["accuracy"] >= .9 for c in MAPS}
        for c in ("familiar","heldout"):
            g[c+"_beats_native"] = contrasts[arm+"/"+c+"/native"]["ci95"][0] > 0
            g[c+"_beats_constant"] = contrasts[arm+"/"+c+"/constant"]["ci95"][0] > .25
        gates[arm] = g
    passed = {a:all(g.values()) and controls for a,g in gates.items()}
    state = ("affine_witness_already_present_initially" if passed["initial_true"] else "final_affine_witness_supported_exploratorily") if passed["final_true"] else ("initial_only_affine_witness" if passed["initial_true"] else "registered_affine_witness_not_supported")
    return dict(controls_valid=bool(controls),true_arms=passed),state if controls else "inconclusive_failed_control"

def compare(expected,actual,path=""):
    differences = []
    if isinstance(expected,dict):
        require(isinstance(actual,dict),"report object "+path)
        for k,v in expected.items(): differences.extend(compare(v,actual[k],path+"/"+k))
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(expected)==len(actual),"report list "+path)
        for i,(a,b) in enumerate(zip(expected,actual)): differences.extend(compare(a,b,path+"/"+str(i)))
    elif type(expected) is float:
        require(type(actual) in (int,float) and math.isfinite(actual),"report numeric "+path)
        differences.append(abs(expected-actual)); require(abs(expected-actual)<=1e-10,"report numeric mismatch "+path)
    else: require(type(expected) is type(actual) and expected==actual,"report exact mismatch "+path)
    return differences

def recompute(populations,parameters):
    grouped,summaries,diagnostics,controls,native_summaries,fit_counts,dispersion = {},{},{},{},{},{},{}
    for arm in ARMS:
        stage = "initial" if arm=="initial_true" else "final"
        x,labels,_,_ = populations[stage+"/seen"]; fitted = permute(labels) if arm=="final_permuted" else labels
        model,diagnostics[arm] = fit(x,fitted)
        primary = {k:numbers(parameters[arm][k],v.shape) for k,v in model.items()}
        for k in model: close(model[k],primary[k],arm+"/"+k)
        stored=parameters[arm]
        for report_key,our_key in (("objective","objective"),("condition_number","condition_number")):
            close(np.asarray(diagnostics[arm][our_key]),numbers(stored[report_key],()),arm+"/"+report_key)
        require(type(stored["clamped_dimensions"]) is int and stored["clamped_dimensions"]==diagnostics[arm]["clamped_dimensions"],"clamped dimension count")
        require(np.array_equal(numbers(stored["quantiles"]["probabilities"],(7,)),[0,.01,.1,.5,.9,.99,1]),"quantile probabilities")
        for k in ("std","scale"):
            close(np.asarray(diagnostics[arm][k+"_quantiles"]),numbers(stored["quantiles"][k],(7,)),arm+"/"+k+" quantiles")
        stored_residual=numbers(stored["relative_normal_equation_residual"],())
        require(0<=stored_residual<=1e-9,"reported normal residual")
        z=(x-primary["mean"])/primary["scale"];centered=np.eye(4)[fitted]-primary["intercept"]
        rhs=z.T@centered/len(x);normal=z.T@z/len(x)+.01*np.eye(256)
        primary_residual=np.linalg.norm(normal@primary["coefficients"]-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)
        require(primary_residual<=1e-9 and (np.any(rhs) or not np.any(primary["coefficients"])),"stored normal residual/zero RHS")
        diagnostics[arm]["stored_relative_residual"]=float(primary_residual)
        for c,maps in MAPS.items():
            xx,y,omitted,native=populations[stage+"/"+c]; m=len(maps)
            ours, theirs = scores(model,xx),scores(primary,xx)
            close(ours,theirs,arm+"/"+c+"/scores")
            pred = ours.argmax(1); require(np.array_equal(pred,theirs.argmax(1)),"independent argmax differs")
            summaries[arm+"/"+c],grouped[arm+"/"+c]=summarize(pred,y,omitted,m)
            native_summaries[stage+"/"+c],grouped[stage+"/"+c+"/native"]=summarize(native,y,omitted,m)
            repeated=np.repeat(xx.reshape(64,m,256).mean(1),m,axis=0)
            nullpred=scores(model,repeated).argmax(1)
            null_ok=bool(np.all(nullpred.reshape(64,m)==nullpred.reshape(64,m)[:,:1]) and np.sum(nullpred==y)*4==len(y))
            require(null_ok,"group-mean invariance/quarter control")
            controls[arm+"/"+c]=dict(correct=int(np.sum(nullpred==y)),rows=len(y),identical_group_predictions=null_ok)
            block=xx.reshape(64,m,256);mean=block.mean(1)
            dispersion[stage+"/"+c]=dict(within_group_rms=float(np.sqrt(np.mean((block-mean[:,None,:])**2))),
                                        between_group_rms=float(np.sqrt(np.mean((mean-mean.mean(0))**2))))
        diagnostics[arm]["fitting_label_correct"] = int(np.sum(scores(model,x).argmax(1)==fitted))
        diagnostics[arm]["true_label_correct"] = summaries[arm+"/seen"]["correct"]
        fit_counts[arm]=dict(fitting_label_correct=diagnostics[arm]["fitting_label_correct"],true_label_correct=diagnostics[arm]["true_label_correct"],rows=1024)
    weights,draw_hash={},{}
    for c in MAPS:
        draw=np.random.Generator(np.random.PCG64(1942 if c=="seen" else 1943)).integers(0,64,(10000,64))
        weights[c]=np.stack([np.bincount(v,minlength=64) for v in draw]).astype(float)
        draw_hash[c]=hashlib.sha256(draw.astype("<u8").tobytes()).hexdigest()
    contrasts={}
    for arm in ARMS:
        stage="initial" if arm=="initial_true" else "final"
        for c in MAPS:
            v=grouped[arm+"/"+c]
            for name,other in (("native",grouped[stage+"/"+c+"/native"]),("constant",.25)):
                contrasts[arm+"/"+c+"/"+name]=interval(v-other,weights[c])
    for c in MAPS:
        contrasts["final_minus_initial/"+c]=interval(grouped["final_true/"+c]-grouped["initial_true/"+c],weights[c])
    null_pass=all(summaries["final_permuted/"+c]["accuracy"]<=.5 for c in ("familiar","heldout"))
    gates,state=gates_and_decision(summaries,contrasts,null_pass)
    return dict(summaries=summaries,contrasts=contrasts,gates=gates,decision=state,native=native_summaries,
                fit_counts=fit_counts,group_mean_controls=controls,dispersion=dispersion,
                permutation_sha256=hashlib.sha256(permutation_indices().astype("<u8").tobytes()).hexdigest(),
                bootstrap_seeds=dict(seen=1942,fresh=1943)),dict(fits=diagnostics,draws_u64le_sha256=draw_hash)

def review(config_path,config_sha,report_path,report_sha):
    started=time.monotonic(); config=decode(pinned(dict(path=str(config_path),sha256=config_sha)))
    report=decode(pinned(dict(path=str(report_path),sha256=report_sha)))
    require(report["accepted"] is True,"primary report rejected")
    require(report["config_sha256"]==config_sha and report["registration_sha256"]==config["registration"]["sha256"],"report binding")
    pinned(config["registration"])
    require(type(config["frozen_files"]) is dict and config["frozen_files"],"source freeze missing")
    for path,sha in config["frozen_files"].items(): verify_frozen(path,sha)
    require(set(config["streams"])=={s+"/"+c for s in ("initial","final") for c in MAPS} and set(config["audits"])==set(MAPS),"stream selectors")
    audits={c:[decode(x) for x in pinned(v).splitlines()] for c,v in config["audits"].items()}
    populations={}
    for key,binding in config["streams"].items():
        stage,c=key.split("/"); populations[key]=population(pinned(binding),audits[c],c,stage)
    root=Path(config["output_root"]); pp=Path(report["parameters"]["path"])
    require(root.is_absolute() and root.is_dir() and not root.is_symlink() and pp.resolve().is_relative_to(root.resolve()),"parameter output root")
    parameters=decode(pinned(report["parameters"]))
    require(set(parameters)==set(ARMS),"exact fitted arms")
    result,extra=recompute(populations,parameters); differences=compare(result,report)
    require(time.monotonic()-started<120,"review CPU deadline")
    return dict(accepted=True,recomputed=result,additional=extra,config_sha256=config_sha,report_sha256=report_sha,
                reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,
                maximum_absolute_difference=max(differences,default=0),compared_float_fields=len(differences),
                scope="Independent augmented least squares, parameter/score/argmax agreement and paired scoring; parent source/core/profiler/chronology provenance is shared through supplied frozen bindings. Exploratory reused panels, no information-absence or promotion claim.")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("config","config-sha256","report","report-sha256","output"):p.add_argument("--"+key,required=True)
    a=p.parse_args();require(not Path(a.output).exists(),"output exists")
    signal.signal(signal.SIGALRM,lambda *_:require(False,"120-second CPU deadline"));signal.alarm(120)
    try:result=review(a.config,a.config_sha256,a.report,a.report_sha256);code=0
    except (OSError,ValueError,TypeError,KeyError,OverflowError,np.linalg.LinAlgError) as error:
        result=dict(accepted=False,decision="failed_integrity",error=str(error));code=1
    finally:signal.alarm(0)
    with open(a.output,"x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    return code

if __name__=="__main__":raise SystemExit(main())
