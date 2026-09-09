"""Prepare all registered batch candidates before model outcomes exist."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c18_data_preparation',R/'data.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
root=R/'datasets';root.mkdir(exist_ok=False)
start=time.monotonic();D.require(D.file_sha(D.C17_DATA)==D.C17_DATA_SHA,'C17 source dataset changed');visual,audit=D.load_visual();entries=[]
for batch in (512,1024,2048,4096,8192,16384,32768):
    for kind,count in (('training',None),('smoke',2),('smoke',5)):
        value=D.dataset(batch,kind=kind,smoke_updates=count,visual=visual,visual_audit=audit)
        D.validate(value,visual=visual,visual_audit=audit)
        suffix='training' if kind=='training' else f'smoke-{count}'
        path=root/f'b{batch}-{suffix}.json'
        with path.open('xb') as f:f.write(D.canonical_bytes(value))
        entries.append(dict(path=str(path),sha256=D.file_sha(path),bytes=path.stat().st_size,schedule=value['schedule'],updates=len(value['updates']),last_update_rows=len(value['updates'][-1])))
        print(f'Prepared b{batch}-{suffix}',flush=True)
result=dict(accepted=True,entries=entries,elapsed_seconds=time.monotonic()-start,data_sha256=D.file_sha(R/'data.py'),producer_only=True)
with (R/'dataset-preparation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(accepted=True,files=len(entries),elapsed_seconds=result['elapsed_seconds'])))
