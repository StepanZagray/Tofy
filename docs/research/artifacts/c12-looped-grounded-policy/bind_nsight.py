#!/usr/bin/env python3
import argparse,collections,csv,hashlib,importlib.util,json,os,shlex,shutil,signal,socket,subprocess,time
if not __debug__:raise RuntimeError("Python optimization forbidden")
from pathlib import Path

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def command(cmd,dest,timeout=90):
    lifecycle=Path('/home/stepan/Research/_runs/2026-09-09T124206Z-tofy-looped-role-selector-witness/supervise_cpu.py')
    assert digest(lifecycle)=='6796e12fb06a87f713f76677baf543758943716287e8b4a9f2ef68a2ec7a0cd7'
    spec=importlib.util.spec_from_file_location('c12_binding_lifecycle',lifecycle)
    life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
    signal.signal(signal.SIGINT,life.stop_signal);signal.signal(signal.SIGTERM,life.stop_signal)
    def record(pid,start):save(dest.with_suffix('.process.json'),dict(pid=pid,supervisor_pid=os.getpid(),process_start_ticks=start,command=cmd))
    with dest.open('x') as out:state=life.supervise(cmd,Path.cwd(),os.environ.copy(),out,record,timeout=timeout)
    save(dest.with_suffix('.exit.json'),state)
    assert state['returncode']==0 and state['error'] is None and state['cleanup_error'] is None and state['pid_gone'] and state['group_gone'] and not state['owned_survivors'] and not state['group_survivors'],state

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;assert root.is_absolute() and root.is_dir() and not root.is_symlink()
    output=root.with_suffix('.bound');output.mkdir()
    nsdir=root.with_suffix('.nsight')
    state=json.loads(root.with_suffix('.exit.json').read_text());assert state['accepted'] is True and type(state['captures_expected']) is int and state['captures_expected']>0
    traces={}
    for trace in root.with_suffix('.profiles').glob('*/trace.jsonl'):
        meta=json.loads(trace.open().readline());assert meta['correlation_id'] not in traces,'duplicate correlation ID';traces[meta['correlation_id']]=(trace,meta)
    assert len(traces)==state['captures_expected']==len(list(nsdir.glob('*.nsys-rep'))),'capture population differs'
    metadata=json.loads((root/'metadata.json').read_text())
    proc=json.loads(root.with_suffix('.process.json').read_text())
    assert not Path('/proc',str(proc['pid'])).exists(),'wait for Nsight exit'
    nsys=proc['command'][0]; cg='/home/stepan/Projects/code/.tofy-target-profile/debug/cargo-candle-graph'
    assert digest(Path(cg))=='54a9dfd118c26b165efde74d02b9a03ec8db290d3dcb299a69cbdf1a9569ef8a'
    found=set();summaries=[]
    for index,raw in enumerate(sorted(nsdir.glob('*.nsys-rep'))):
        scratch=output/f'export-{index}';scratch.mkdir()
        attachment=output/f'attachment-{index}';attachment.mkdir();shutil.copy2(raw,attachment/'capture.nsys-rep')
        cmd=[nsys,'stats','--report','cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_gpu_proj_trace,cuda_gpu_trace,osrt_sum,nvtx_sum','--format','csv','--timeunit','nanoseconds','--output',str(attachment/'stats'),'--sqlite',str(scratch/'capture.sqlite'),str(attachment/'capture.nsys-rep')]
        command(cmd,scratch/'export.log')
        rows=list(csv.DictReader((attachment/'stats_nvtx_gpu_proj_trace.csv').open()))
        names=collections.Counter(r['Name'] for r in rows)
        matches=[key for key in traces if any(n.startswith(':'+key+'/') for n in names)]
        assert len(matches)==1, matches
        key=matches[0];assert key not in found;found.add(key);trace,meta=traces[key]
        contract=meta['capture_contract'];expected=contract['gpu_expected_semantic_labels']
        app={n[1:]:count for n,count in names.items() if n.startswith(':tofy.looped/')}
        assert app=={n:1 for n in expected},(app,expected)
        foreign={n:c for n,c in names.items() if not n.startswith(':tofy.looped/')}
        assert all(n.startswith(('cuBLAS:','cuDNN:')) for n in foreign),foreign
        assert all(int(r['NumGPUOps'])>0 for r in rows if r['Name'].startswith(':tofy.looped/'))
        manual={'trace':str(trace),'raw_report':str(raw),'application_labels_each_once':app,'expected_labels':expected,'library_domain_counts':foreign,'qualification':'Read-only application-label comparison accounts for Nsight default-domain colon. Raw CSV is unmodified. Candle Graph0.10.1 automatic correlation remains incomplete because it compares domain-prefixed/library labels literally.'}
        save(scratch/'application-label-check.json',manual)
        manifest={'schema':'candle-graph/nsight-capture/1','run':{'id':meta['run_id'],'started_at':meta['timestamp']},'correlation':{'id':key},'tool':{'name':'nsys','version':'2026.4.1.191-264138605071v0'},'commands':[shlex.join(proc['command']),shlex.join(cmd)],'hardware':{'host':socket.gethostname(),'devices':[metadata['provenance']['gpu']]},'source_revisions':{'tofy':metadata['provenance']['source_revision'],'candle_graph':metadata['provenance']['candle_graph_revision']},'required_semantic_labels':contract['required_semantic_labels'],'gpu_expected_semantic_labels':expected,'cpu_only_semantic_labels':contract.get('cpu_only_semantic_labels',[]),'artifacts':[{'path':f.name,'size_bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(attachment.iterdir())]}
        save(attachment/'capture-manifest.json',manifest)
        bundle=output/trace.parent.name
        command([cg,'report',str(trace),'--nsight-dir',str(attachment),'--bundle',str(bundle)],scratch/'publication.json')
        command([cg,'verify',str(bundle),'--semantic'],scratch/'verification.json')
        command([cg,'overview',str(bundle)],scratch/'overview.json')
        overview=json.loads((scratch/'overview.json').read_text())
        assert overview['health']['structurally_valid'] and overview['health']['capture_complete'],overview['health']
        command([cg,'query',str(bundle),'--kind','gpu-correlation'],scratch/'gpu-correlation.json')
        summary={'trace':str(trace),'bundle':str(bundle),'health':overview['health'],'gpu':overview['gpu'],'raw_application_labels_verified':True,'manual_check':str(scratch/'application-label-check.json')}
        summaries.append(summary)
    assert found==set(traces),(found,list(traces))
    save(output/'summary.json',summaries)
    print(json.dumps([{'bundle':s['bundle'],'health':s['health'],'gpu_status':s['gpu']['status'],'binding':s['gpu']['provenance_binding'],'raw_application_labels_verified':True} for s in summaries]),flush=True)
if __name__=='__main__':main()
