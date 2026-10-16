"""Standard-library replay of the compact public SUM prototype checkpoint."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

STAGES=('default','chunk1024','chunk2048','recheck')

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

def verify_data(root,data):
    require(data['schema']==1 and data['status']=='completed_validated','checkpoint status differs')
    inventory=load(root/'cases12.json')['cases']
    require(len(inventory)==12 and len({x['id'] for x in inventory})==12,'complete case inventory required')
    measurements=samples=0
    for label in ('v3','v4'):
        cohort=data[label]
        require(cohort['variant']==label and [x['case'] for x in cohort['cases']]==inventory,'native inventory differs')
        for row in cohort['cases']:
            require(set(row['stages'])==set(STAGES),'all native stages required')
            fixture=None;default_source=None
            for name in STAGES:
                stage=row['stages'][name]
                median=series(stage['event_us']);sampling(stage['sampling'],stage['warmup'],stage['event_us'],stage['graph_host_us'])
                series(stage['graph_host_us']);series(stage['stream_event_us']);series(stage['host_wall_us'])
                require(stage['saved_output']['failed_elements']==0 and stage['recorded_correctness']['errors']==0 and
                    stage['recorded_correctness']['inputs_unchanged'] and stage['recorded_correctness']['guards_unchanged'] and
                    stage['recorded_correctness']['all_outputs_finite'],'native correctness claim differs')
                manifest=load(package_path(root,stage['manifest']['file']))
                for k in ('operation','precision','dimensions','seed','pattern'):
                    require(manifest[k]==row['case'][k],'included manifest case identity differs')
                if fixture is None:fixture=stage['fixture_sha256']
                else:require(fixture==stage['fixture_sha256'],'native fixture/oracle hashes differ across stages')
                source={k:v['original_raw_sha256'] for k,v in stage['sources'].items()}
                if name=='default':default_source=source
                if name=='recheck':require(default_source==source,'default source preservation differs')
                proof=stage['diagnostic']['proof'];chunk=0 if name in ('default','recheck') else int(name[5:])
                require(stage['chunk']==proof['chunk']==chunk and proof['transformed'] is bool(chunk),'transform request differs')
                if chunk:
                    rows,width=row['case']['dimensions'];size=4 if row['case']['precision']=='fp32' else 2
                    require(proof['input_slot']==0 and proof['output_slot']==3 and proof['input_bytes']==rows*width*size and
                        proof['output_bytes']==rows*size and proof['serial_iterations']==width//chunk and
                        proof['programs']==rows and proof['largest_materialized_tile_elements']==chunk,'actual IR proof differs')
                    require(proof['collective_invocations_per_program']==(width//chunk if label=='v3' else 1) and
                        proof['contribution_elements_per_program']==(width if label=='v3' else chunk),'V3/V4 contribution semantics differ')
                    require(stage['diagnostic'].get('disjoint_receipt'),'actual final-range guard receipt missing')
                measurements+=1;samples+=7
            a=row['stages']['default']['event_us']['p50'];b=row['stages']['recheck']['event_us']['p50']
            require(near(row['default_recheck_ratio'],b/a),'control drift differs')
            for candidate in ('chunk1024','chunk2048'):
                c=row['stages'][candidate]['event_us']['p50'];ratios=row['candidate_ratios'][candidate]
                require(near(ratios['over_default'],c/a) and near(ratios['over_recheck'],c/b),'candidate ratio differs')
    fresh=data['fresh_torch']
    require([x['case'] for x in fresh['cases']]==inventory,'fresh Torch inventory differs')
    for native,torch in zip(data['v4']['cases'],fresh['cases']):
        denominator=series(torch['event_us']);sampling(torch['sampling'],torch['warmup'],torch['event_us'],torch['graph']['host_wall_us_per_operation'])
        require(torch['fixture_sha256']==native['stages']['default']['fixture_sha256'] and
            torch['saved_output']['failed_elements']==0,'fresh Torch fixture/oracle differs')
        require(torch['compiler_evidence']['fullgraph'] is True,'fullgraph compiler evidence missing')
        for key in ('compiled_correctness_before','compiled_correctness_after'):
            require(torch['recorded_correctness'][key]['failed_elements']==0,'Torch correctness failed')
        for name in STAGES:require(near(torch['native_over_fresh_torch'][name],native['stages'][name]['event_us']['p50']/denominator),'fresh Torch denominator differs')
        measurements+=1;samples+=7
    archive_refs(root,data)
    actual_preserved={a['case']['id']:
        {k:v['original_raw_sha256'] for k,v in a['stages']['default']['sources'].items()}==
        {k:v['original_raw_sha256'] for k,v in b['stages']['default']['sources'].items()}
        for a,b in zip(data['v3']['cases'],data['v4']['cases'])}
    require(data['default_source_preservation']==actual_preserved and all(actual_preserved.values()),'cross-version default source preservation differs')
    return dict(native_measurements=96,torch_measurements=12,event_samples=samples,all_measurements=measurements)

def main():
    root=Path(__file__).resolve().parent
    verify_files(root)
    print(json.dumps(dict(status='passed',**verify_data(root,load(root/'evidence.json')))))

if __name__=='__main__':main()
