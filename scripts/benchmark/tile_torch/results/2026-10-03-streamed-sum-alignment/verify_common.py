"""Frozen stdlib file/timing checks reused from the preceding public checkpoint."""

import hashlib, json, math, re, statistics

from pathlib import Path

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(data):return hashlib.sha256(data).hexdigest()

def near(a,b):return math.isclose(a,b,rel_tol=2e-7,abs_tol=1e-9)

def load(path):return json.loads(path.read_text(encoding='utf-8'))

def package_path(root,name):
    p=(root/name).resolve()
    require(not Path(name).is_absolute() and p.is_relative_to(root.resolve()),'public path escapes package')
    return p

def verify_files(root):
    manifest=load(root/'manifest.json')
    require(manifest['schema']==1 and manifest['files'],'invalid public manifest')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    require(actual==set(manifest['files']),'public file inventory differs')
    for name,record in manifest['files'].items():
        data=package_path(root,name).read_bytes()
        require(sha(data)==record['sha256'] and len(data)==record['bytes'],'public file hash mismatch: '+name)
        require(b'\r' not in data,'public LF normalization changed: '+name)
        text=data.decode('utf-8')
        require(not re.search(r'[A-Za-z]:[\\/]+Users[\\/]+',text) and not re.search(r'GPU-[0-9a-fA-F-]+',text),'machine-specific user path/UUID leaked')
    return manifest

def series(value):
    values=value['samples']
    require(len(values)==7 and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in values),'seven finite positive samples required')
    median=statistics.median(values)
    require(near(median,value['p50']),'stored median differs')
    return median

def sampling(record,warmup,event,host=None):
    require(record['protocol']=='adaptive_replay_span_v2' and record['sample_target_ms']==100 and
            record['replay_cap']==65536 and record['operation_cap']==10000000,'graph-v2 policy differs')
    r=record['replays_per_sample'];operations=record['operations_per_sample']
    require(type(r) is int and 1<=r<=65536 and operations==r*100,'normalization denominator differs')
    event_span=record['event_span_ms']['samples'];host_span=record['host_span_ms']['samples']
    require(len(event_span)==len(host_span)==7 and all(math.isfinite(x) and x>0 for x in event_span+host_span),'raw graph spans invalid')
    require(all(near(a*1000/operations,b) for a,b in zip(event_span,event['samples'])),'event span/operation normalization differs')
    if host is not None:
        require(len(host['samples'])==7 and all(near(a*1000/operations,b) for a,b in zip(host_span,host['samples'])),
                'host span/operation normalization differs')
    attempts=record['calibration']
    require(1<=len(attempts)<=4 and attempts[-1]['replays']==r,'last measured calibration differs')
    expected=1
    for index,attempt in enumerate(attempts):
        span=attempt['event_span_ms'];n=attempt['replays']
        require(n==expected and math.isfinite(span) and span>0 and attempt['host_wall_ms']>0,'invalid calibration attempt')
        stop=span>=80 or n==65536 or index==3
        if index+1<len(attempts):
            require(not stop,'calibration passed its stopping rule')
            expected=min(65536,max(n+1,math.ceil(n*100/span)))
        else:
            require(stop and record['calibration_target_reached'] is (span>=80),'calibration stop/target flag differs')
    require(warmup['target_ms']==500 and warmup['actual_ms']>=500 and warmup['replays']>=1,'warmup differs')
    require(record['prime_event_ms']>0 and record['prime_host_ms']>0,'event priming missing')

def archive_refs(root,value):
    if isinstance(value,dict):
        if {'file','sha256','bytes','original_raw_sha256','normalization'}<=set(value):
            data=package_path(root,value['file']).read_bytes()
            require(sha(data)==value['sha256'] and len(data)==value['bytes'],'included source/manifest hash differs')
            if value['normalization']=='identity':
                require(value['original_raw_sha256']==value['sha256'] and value['original_bytes']==value['bytes'],'identity normalization provenance differs')
        for item in value.values():archive_refs(root,item)
    elif isinstance(value,list):
        for item in value:archive_refs(root,item)
