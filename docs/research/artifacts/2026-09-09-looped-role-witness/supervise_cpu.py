#!/usr/bin/env python3
"""Bounded CPU diagnostic from a pinned clean source; owns only its child group."""
import argparse, ctypes, datetime, hashlib, json, os, signal, subprocess, time
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def stop_signal(number, frame):
    raise InterruptedError(f'received signal {number}')

def process_table():
    result={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            result[int(entry.name)]=dict(state=fields[0],ppid=int(fields[1]),pgid=int(fields[2]),start=int(fields[19]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result

def collect_owned(pgid,baseline,owned):
    table=process_table();changed=True
    while changed:
        changed=False
        for pid,info in table.items():
            if pid in owned:continue
            parent=table.get(info['ppid'])
            if (info['pgid']==pgid or parent is not None and owned.get(info['ppid'])==parent['start']
                    or info['ppid']==os.getpid() and (pid,info['start']) not in baseline):
                owned[pid]=info['start'];changed=True
    return table

def cleanup(process,baseline,owned,grace):
    for sig in (signal.SIGTERM,signal.SIGKILL):
        deadline=time.monotonic()+grace
        while True:
            process.poll()
            table=collect_owned(process.pid,baseline,owned)
            for pid,start in list(owned.items()):
                info=table.get(pid)
                if info is not None and info['start']==start and info['state']=='Z' and pid!=process.pid:
                    try:os.waitpid(pid,os.WNOHANG)
                    except ChildProcessError:pass
            table=collect_owned(process.pid,baseline,owned)
            survivors=[pid for pid,start in owned.items() if pid in table and table[pid]['start']==start]
            if not survivors:return []
            for pid in survivors:
                current=process_table().get(pid)
                if current is not None and current['start']==owned[pid]:
                    try:os.kill(pid,sig)
                    except ProcessLookupError:pass
            if time.monotonic()>=deadline:break
            time.sleep(min(.025,max(0,deadline-time.monotonic())))
    table=collect_owned(process.pid,baseline,owned)
    return [pid for pid,start in owned.items() if pid in table and table[pid]['start']==start]

def supervise(command,cwd,env,log,record,timeout=90,grace=3):
    """Bound and reap the complete owned tree, including descendants that setsid."""
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'cannot enable child subreaper')
    baseline={(pid,info['start']) for pid,info in process_table().items() if info['ppid']==os.getpid()}
    started=time.monotonic();process=None;owned={};error=None;cleanup_error=None;survivors=[]
    try:
        pending=[]
        handlers={sig:signal.signal(sig,lambda number,frame:pending.append(number)) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            table=collect_owned(process.pid,baseline,owned)
            record(process.pid,owned.get(process.pid))
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
        if pending:raise InterruptedError(f'cancellation during spawn: {pending}')
        while True:
            table=collect_owned(process.pid,baseline,owned)
            if time.monotonic()-started>=timeout:raise TimeoutError('registered CPU deadline exceeded')
            if process.poll() is not None:
                if process.returncode!=0:raise RuntimeError(f'child exited {process.returncode}')
                if any(pid!=process.pid and info['state']!='Z' and owned.get(pid)==info['start'] for pid,info in table.items()):
                    raise RuntimeError('child exited with active owned descendants')
                break
            time.sleep(min(.1,max(0,started+timeout-time.monotonic())))
    except BaseException as exception:
        error=repr(exception)
    finally:
        handlers={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            if process is not None:survivors=cleanup(process,baseline,owned,grace)
        except BaseException as exception:
            cleanup_error=repr(exception);error=error or cleanup_error
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    table=process_table()
    survivors=[pid for pid,start in owned.items() if pid in table and table[pid]['start']==start]
    group=[pid for pid,info in table.items() if process is not None and info['pgid']==process.pid]
    if survivors or group:error=error or 'owned processes survived bounded cleanup'
    return dict(returncode=process.returncode if process else None,error=error,cleanup_error=cleanup_error,
        pid=process.pid if process else None,pid_gone=bool(process and process.pid not in table),
        pgid=process.pid if process else None,group_gone=not group,group_survivors=group,
        owned_survivors=survivors,owned_pids=sorted(owned),owned_process_start_ticks=owned,
        elapsed_seconds=time.monotonic()-started)

def verify_source(config,args):
    if sha(args.config)!=args.sha256:raise RuntimeError('launch config changed')
    cwd=Path(config['cwd'])
    if subprocess.check_output(['git','-C',str(cwd),'rev-parse','HEAD'],text=True,timeout=10).strip()!=config['source']:
        raise RuntimeError('source revision mismatch')
    if subprocess.check_output(['git','-C',str(cwd),'status','--porcelain'],text=True,timeout=10).strip():
        raise RuntimeError('dirty source checkout')
    for artifact,digest in config['frozen_files'].items():
        if sha(artifact)!=digest:raise RuntimeError('frozen input changed: '+artifact)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args=parser.parse_args()
    if sha(args.config)!=args.sha256:
        raise RuntimeError('launch config changed')
    config=json.loads(args.config.read_text())
    cwd=Path(config['cwd']); output=Path(config['output'])
    if not args.config.is_absolute() or not cwd.is_absolute() or not output.is_absolute():
        raise RuntimeError('config, checkout and output paths must be absolute')
    if output.exists() or output.is_symlink() or list(output.parent.glob(output.name+'.*')) or any(output.with_suffix(suffix).exists() for suffix in ('.stdout','.process.json','.exit.json')):
        raise RuntimeError('never reuse diagnostic root or evidence siblings')
    verify_source(config,args)
    if type(config['timeout_seconds']) is not int or config['timeout_seconds']!=90 or type(config['command']) is not list or not config['command'] or any(type(x) is not str for x in config['command']):
        raise RuntimeError('unregistered deadline or empty command')
    signal.signal(signal.SIGINT, stop_signal); signal.signal(signal.SIGTERM, stop_signal)
    env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    def record(pid,start_ticks):
        value=dict(pid=pid,supervisor_pid=os.getpid(),pgid=pid,process_start_ticks=start_ticks,
            command=config['command'],cwd=str(cwd),source=config['source'],config=str(args.config),
            config_sha256=args.sha256,supervisor_sha256=sha(__file__),
            started_local=datetime.datetime.now().astimezone().isoformat())
        save(output.with_suffix('.process.json'),value)
        print(json.dumps({key:value[key] for key in ('pid','supervisor_pid','started_local')}),flush=True)
    with output.with_suffix('.stdout').open('x') as log:
        state=supervise(config['command'],cwd,env,log,record,timeout=config['timeout_seconds'])
    handlers={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        try:
            verify_source(config,args);unchanged=True
        except BaseException as exception:
            unchanged=False;state['provenance_error']=repr(exception)
        state.update(inputs_unchanged=unchanged,finished_local=datetime.datetime.now().astimezone().isoformat())
        state['accepted']=(state['returncode']==0 and state['error'] is None and state['cleanup_error'] is None
            and state['group_gone'] and state['pid_gone'] and not state['owned_survivors'] and unchanged)
        save(output.with_suffix('.exit.json'),state);print(json.dumps(state),flush=True)
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if not state['accepted']:
        raise SystemExit(1)

if __name__=='__main__':main()
