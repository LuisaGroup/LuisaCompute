"""Recompute the compact Service holdout projection. Stdlib only; no GPU or fit."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import statistics

SUMMARY_SHA='93c72c8919d9b4694c7be5b667f493d314bb78250c1ff94f3563ce398b3a0945'
FIT_SHA='fe7e04b6e5f5c9337b656615fc497d93854d340e6d800316f6cffcee101a6b01'
ORDER=[[128,2],[64,1],[128,1],[128,4],[256,4],[256,1]]
COEFFICIENTS=[1.0139535119316005,.00040283896787313774,.002103977310593496,0.,.0020014559747718054,0.]
STAGES=('native_initial','tirx_fixed','tirx_profile','native_recheck')


def need(value,message):
    if not value:raise ValueError(message)
def close(a,b):
    need(type(a)in(int,float) and math.isfinite(a) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),'numeric mismatch')
def read(p):
    def reject(v):raise ValueError('nonfinite JSON '+v)
    return json.loads(p.read_text(encoding='utf-8'),parse_constant=reject)
def receipt(r):
    need(isinstance(r,dict) and re.fullmatch('[0-9a-f]{64}',r.get('sha256','')) is not None and type(r.get('bytes'))is int and r['bytes']>0,'receipt digest/bytes')
    need(isinstance(r.get('reference'),str) and not Path(r['reference']).is_absolute() and not re.match(r'^[A-Za-z]:',r['reference']),'portable receipt reference')
def sample(values):
    need(isinstance(values,list) and len(values)==7 and all(type(v)in(int,float) and math.isfinite(v) and v>0 for v in values),'seven finite positive samples')
    return statistics.median(values)
def timing(r):
    e,s,w=r['event_us'],r['sampling'],r['warmup'];p=sample(e['samples']);close(e['p50'],p)
    need(e['range']==[min(e['samples']),max(e['samples'])],'sample range')
    need(s['protocol']=='adaptive_replay_span_v2' and type(s['replays_per_sample'])is int and s['replays_per_sample']>0 and s['operations_per_sample']==100*s['replays_per_sample'],'graph100/replay denominator')
    need(s['sample_target_ms']==100 and s['calibration_target_reached']is True and w['target_ms']==500 and w['actual_ms']>=500,'adaptive span/warmup')
    calibration=s['calibration'];need(isinstance(calibration,list) and calibration,'calibration history')
    last=calibration[-1]
    need(last['replays']==s['replays_per_sample'] and type(last['event_span_ms'])in(int,float) and math.isfinite(last['event_span_ms']) and last['event_span_ms']>=80,'final calibration achieved span')
    spans=s['event_span_ms']['samples'];close(s['event_span_ms']['p50'],sample(spans))
    for x,y in zip(spans,e['samples']):close(x*1000/s['operations_per_sample'],y)
    close(s['host_span_ms']['p50'],sample(s['host_span_ms']['samples']))
    return p
def outputcheck(r,elements):
    need(r['elements']==elements and r['failed_elements']==0,'complete saved output validation')
    receipt(r['receipt'])
def score(c):
    a=COEFFICIENTS;p=c['payload_accesses_per_program'];w=c['payload_accesses_per_worker']
    local=c['scalar_rounds']*a[1]+float(c['reductions'])*c['subgroups_per_program']*a[2]+(w['private_read_bytes']+w['private_write_bytes'])*a[5]
    waves=max(1.,float(c['threadgroups'])*c['subgroups_per_program']*c['programs_per_group']/24)
    reads=float(c['threadgroups'])*c['programs_per_group'] if c['subgroups_per_program']>1 and c['programs_per_group']>1 else float(c['programs'])
    kernel=a[0]+local*waves+(reads*p['global_read_bytes']+float(c['programs'])*p['global_write_bytes'])*a[3]+(w['global_read_bytes']+w['global_write_bytes'])*a[4]
    return dict(program_score=local,concurrent_waves=waves,kernel_score=kernel)


def selection(row):
    packet=row['selection'];d=packet['data'];receipt(packet['receipt'])
    need(d['schema']==1 and d['profile']=='private-service-packing-fast-v1' and d['policy']=='unchanged_ServiceExecutionCostPolicy' and d['fit_sha256']==FIT_SHA,'profile identity')
    need(d['coefficients']==COEFFICIENTS and d['concurrent_subgroups']==24 and d['candidate_order']==ORDER and d['incumbent_index']==0,'frozen coefficients/order')
    need(d['relative_switch_threshold']==.95 and d['threshold_comparison']=='strict_less' and d['tie_rule']=='first_legal_in_frozen_order','threshold/tie')
    need(d['math_scope']=='fast-only' and d['lane_elements']==1 and d['unroll_factor']==64 and d['cache_reduction_inputs']is False and d['private_scalar_budget']==64 and d['fatal_error']is False,'fixed capability')
    need(d['capacity_semantics']=='model-concurrent-subgroups-not-occupancy' and d['runtime_status']=='source-and-cost-matched','runtime/source scope')
    device=d['device'];need(all(device[k]==v for k,v in dict(driver=13040,toolkit=13040,major=8,minor=9,multiprocessors=24,warp_size=32,max_threads_per_block=1024,max_shared_memory_per_block=49152,query_status=0).items()),'device/profile family')
    need(math.isfinite(d['frontend_search_ms']) and d['frontend_search_ms']>=0,'search overhead')
    need(len(d['candidates'])==6,'six actual candidate attempts')
    scores={};rejected=0
    for i,(c,(t,p))in enumerate(zip(d['candidates'],ORDER)):
        need(c['index']==i and c['threads']==t and c['programs_per_group']==p and c['attempted']is True and c['training_schedule_seen']is(i!=5),'candidate identity')
        if not c['legal']:
            need(c['status']=='unsupported' and c['error'].startswith('TileIR ') and c['error'].endswith(": execution scope 'subgroup' on target 'cuda' cannot realize the exact reduction mapping (threads, packing, unrolling, lane elements or input caching)") and c['source_file']is None and c['observation']==dict(overflow=False,records=[]),'unsupported not failure/fallback')
            rejected+=1;continue
        obs=c['observation'];need(c['status']=='legal' and c['error']=='' and not obs['overflow'] and len(obs['records'])==1,'one callback')
        x=obs['records'][0];r=row['case']['dimensions'][0]
        need(x['threads']==t and x['programs_per_group']==p and x['subgroups_per_program']==t//p//32 and x['programs']==r and x['threadgroups']==(r+p-1)//p and x['unroll_factor']==64 and x['lane_elements']==1,'callback geometry')
        need(x['payload_accesses_known']is True,'known demand required')
        for key in ('payload_accesses_per_program','payload_accesses_per_worker'):
            need(set(x[key])=={'global_read_bytes','global_write_bytes','private_read_bytes','private_write_bytes'} and all(type(v)is int and v>=0 for v in x[key].values()),'known payload')
        for k,v in score(x).items():close(x['cost'][k],v)
        need(c['grid']==[x['threadgroups'],1,1] and c['block']==[t,1,1] and c['source_file']==f'service-candidate-{i}.cu','candidate artifact')
        receipt(packet['candidate_source_receipts'][c['source_file']]);scores[i]=x['cost']['kernel_score']
    need(scores,'this closed profile cohort has twelve selected candidates')
    best=min(scores,key=lambda i:(scores[i],i));selected=best if 0 not in scores or (best!=0 and scores[best]<.95*scores[0]) else 0
    reason='incumbent-unavailable' if 0 not in scores else 'predicted-more-than-five-percent' if selected!=0 else 'retained-incumbent'
    need(d['selected_index']==selected and d['reason']==reason,'score-only decision')
    chosen=d['candidates'][selected];need(d['runtime_observation']==chosen['observation'],'actual runtime cost callback equality')
    runtime=row['stages']['tirx_profile'];observed=runtime['observation']['observed']
    need(all(chosen[k]==observed[k] for k in ('entry','grid','block')) and chosen['buffer_arguments']==observed['device_slot_to_host_index'],'selected source ABI to observed graph')
    need(packet['candidate_source_receipts'][chosen['source_file']]['sha256']==runtime['source_receipt']['sha256'],'selected CPU/runtime source receipts')
    fixed=row['stages']['tirx_fixed'];need((fixed['status']=='passed')==(0 in scores),'incumbent admission')
    if 0 in scores:need(packet['candidate_source_receipts']['service-candidate-0.cu']['sha256']==fixed['source_receipt']['sha256'],'fixed/source incumbent identity')
    return selected,rejected


def validate(data):
    need(data['schema']==1 and data['status']=='compact_projection_of_closed_validated_holdout','projection scope')
    need(data['upstream']['summary']['sha256']==SUMMARY_SHA and data['upstream']['frozen_fit']['sha256']==FIT_SHA,'sealed inputs')
    profile=data['frozen_profile'];need(profile['current_holdout_used_for_fit']is False and profile['training_schedule_observations']==29 and profile['coefficients']==COEFFICIENTS and profile['concurrent_subgroups']==24,'no holdout refit')
    need(data['counts']==dict(native_numerical=47,torch=12,unsupported=1,primary_samples=413,actual_tirx_nodes=2300),'closed census')
    need(data['protocol']==dict(samples=7,sample_ms=100,warmup_ms=500,graph_batch=100,threads=4,affinity_mask=0x15400,native_timeout=180,telemetry_ms=500),'protocol')
    need(len(data['cases'])==12 and len({r['case']['id'] for r in data['cases']})==12,'twelve unique cases')
    native=torch=unsupported=nodes=attempts=rejections=0;table=[]
    for row in data['cases']:
        case=row['case'];need(case['fast_math']is True and set(row['stages'])==set(STAGES),'fast four-stage case')
        elements=math.prod(row['original_manifest']['output']['shape']);fixture=row['stages']['native_initial']['fixture_sha256'];medians={}
        for name in STAGES:
            st=row['stages'][name];receipt(st['result_receipt']);need(st['fixture_sha256']==fixture,'fixture/oracle hash identity')
            need(st['process']['status']=='exited' and st['fast_math']is True,'exited/math')
            if st['status']=='unsupported':
                need(name=='tirx_fixed' and st['process']['returncode']==3 and 'cannot realize the exact reduction mapping' in st['reason'],'fixed exact rejection')
                need(all(st[k]is None for k in ('event_us','sampling','warmup','source_receipt','reported_runtime_correctness','saved_output_recheck','observation')),'unsupported has no timing/source/observation')
                unsupported+=1;medians[name]=None;continue
            need(st['status']=='passed' and st['process']['returncode']==0,'native passed')
            receipt(st['source_receipt']);check=st['reported_runtime_correctness']
            need(check['checks']>=2 and check['elements_per_check']==elements and check['errors']==0 and all(check[k]is True for k in ('inputs_unchanged','guards_unchanged','all_outputs_finite')),'reported full Runtime validation')
            outputcheck(st['saved_output_recheck'],elements);medians[name]=timing(st);native+=1
            if name.startswith('native_'):need(st['observation']is None,'native has no private per-node observer')
            else:
                obs=st['observation']['observed'];p=obs['programs_per_group'];t=obs['threads_per_group'];r=case['dimensions'][0]
                need(obs['actual_graph_nodes_checked']==100 and obs['same_module_function_identity_checked']is True and obs['actual_final_pointers_checked']is True,'reported actual graph proof')
                need(obs['grid']==[(r+p-1)//p,1,1] and obs['block']==[t,1,1] and obs['math_policy']=='fast-elements-preserved-reductions-v1' and obs['lane_elements']==1 and obs['actual_plan_unroll_factor']==64,'actual schedule/math')
                resources=st['observation']['resources'];need(len(resources)==4 and {v['attribute']for v in resources}=={'registers','static_shared_bytes','local_bytes','max_threads'},'resource fields')
                need(len({(v['entry'],v['module'],v['function'])for v in resources})==1 and resources[0]['entry']==obs['entry'],'loaded-function resources')
                for value in st['observation']['receipts'].values():receipt(value)
                nodes+=100
        need(row['stages']['native_initial']['source_receipt']['sha256']==row['stages']['native_recheck']['source_receipt']['sha256'],'unchanged native controls')
        t=row['fresh_torch'];need(t['status']=='passed' and t['process']==dict(status='exited',returncode=0) and t['fixture_sha256']==fixture,'fresh Torch scope')
        for name in ('reported_correctness_before','reported_correctness_after'):
            need(t[name]['elements']==elements and t[name]['failed_elements']==0,'Torch full before/after oracle')
        outputcheck(t['saved_output_recheck'],elements);receipt(t['result_receipt']);medians['fresh_torch']=timing(t);torch+=1
        pick,reject=selection(row);attempts+=6;rejections+=reject
        for stage in ('tirx_fixed','tirx_profile'):
            if medians[stage]is None:need(row['ratios'][stage]is None,'no unsupported ratio');continue
            for denominator,key in (('native_initial','over_native_initial'),('native_recheck','over_native_recheck'),('fresh_torch','over_fresh_torch')):
                close(row['ratios'][stage][key],medians[stage]/medians[denominator])
        expected=None if medians['tirx_fixed']is None else medians['tirx_profile']/medians['tirx_fixed']
        if expected is None:need(row['profile_over_fixed_tirx']is None,'unsupported ratio remains null')
        else:close(row['profile_over_fixed_tirx'],expected)
        close(row['native_control_drift'],medians['native_recheck']/medians['native_initial'])
        table.append((case['id'],ORDER[pick],medians,row['selection']['data']['frontend_search_ms']))
    need((native,torch,unsupported,nodes,attempts,rejections)==(47,12,1,2300,72,8),'all results/candidates retained')
    return table


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent);args=p.parse_args()
    root=args.directory.resolve();index=read(root/'index.json')
    need(index['raw_summary']['sha256']==SUMMARY_SHA,'index upstream')
    for name,expected in index['files'].items():
        path=(root/name).resolve();need(path.is_relative_to(root),'escaped public path');raw=path.read_bytes()
        need(hashlib.sha256(raw).hexdigest()==expected['sha256'] and len(raw)==expected['bytes'],'public digest changed: '+name)
    rows=validate(read(root/'evidence.json'))
    print('Validated compact projection: 12 cases, 72 candidates (64 legal/8 rejected), 47 native + 12 Torch, 413 samples, 2300 reported TIRx nodes.')
    for case,geometry,m,search in rows:print(f'{case}: T{geometry[0]}/P{geometry[1]}, profile={m["tirx_profile"]:.6f} us, native_recheck={m["native_recheck"]:.6f} us, Torch={m["fresh_torch"]:.6f} us, CPU search={search:.3f} ms')
    print('No GPU/compile/fit or unbundled source, graph CSV, tensor or binary replay. Numerical/graph checks are retained upstream evidence.')


if __name__=='__main__':main()
