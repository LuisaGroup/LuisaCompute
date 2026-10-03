"""Replay this compact evidence with frozen coefficients; never fit or use a GPU."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import statistics

HERE=Path(__file__).resolve().parent
def read(name): return json.loads((HERE/name).read_text(encoding='utf-8'))
def close(a,b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-12),(a,b)
def same(a,b):
    if isinstance(a,(float,int)) and not isinstance(a,bool): close(a,b)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b,(a,b)
def median(samples):
    assert len(samples)==7 and all(math.isfinite(x) and x>0 for x in samples)
    return statistics.median(samples)
def score(coeff,features): return sum(c*x for c,x in zip(coeff,features))
def gm(values): return math.exp(sum(math.log(v) for v in values)/len(values))
def geometry_gm(cases, key):
    groups=defaultdict(list)
    for c in cases: groups[c['group']].append(c[key])
    return gm([gm(v) for v in groups.values()])
def features(case,resources=None,threads=0):
    rows,width=case['case']['dimensions']; sm=24; warp=32
    memory=rows*width*4/(sm*2**20)
    if threads==0: return [1.0,memory,math.ceil(rows/sm)*width]
    assert threads in (128,256,512,1024) and width%(threads*8)==0
    assert resources['resource-status']=='ok' and resources['resource-scope']=='installed-entry'
    assert resources['resource-entry']=='luisa_tile_cub_scan'
    assert all(isinstance(resources[k],int) and resources[k]>=0 for k in ('registers','static-shared-bytes','local-bytes','max-threads'))
    assert resources['local-bytes']==0 and resources['max-threads']>=threads
    assert resources['capacity-status']=='ok' and resources['capacity-threads']==threads and resources['capacity-dynamic-shared-bytes']==0
    capacity=resources['resident-cta-capacity']; assert isinstance(capacity,int) and capacity>0
    batches=math.ceil(rows/(sm*capacity)); chunks=width//(threads*8); warps=math.ceil(threads/warp)
    return [1.0,memory,batches*chunks*8,batches*chunks*warps]
def decide(case,coeff):
    scores={'0':score(coeff['tile'],features(case))}
    for option in case['options']:
        if option['admitted_to_model']:
            assert option['observed_cub_selected'] and option['requested_threads']!=0
            f=features(case,option['resources'],option['requested_threads'])
            same(f,option['features'])
            scores[str(option['requested_threads'])]=score(coeff['cub'],f)
        else:
            assert option['features'] is None and option['facts'] is None
            assert option['exclusion_reason']
    winner=min(scores,key=lambda t:(scores[t],int(t)))
    selected=int(winner) if scores[winner]/scores['0']<0.95 else 0
    return selected,scores
def check_decision(case,coeff,expected):
    chosen,scores=decide(case,coeff)
    assert chosen==expected['selected_threads'],(case['case']['id'],chosen,expected)
    assert scores.keys()==expected['scores_us_proxy'].keys()
    for k,v in scores.items(): close(v,expected['scores_us_proxy'][k])
    close(scores[str(chosen)]/scores['0'],expected['predicted_ratio'])
    return chosen

def main():
    manifest=read('manifest.json')
    for name,r in manifest['files'].items():
        assert Path(name).name==name
        raw=(HERE/name).read_bytes()
        assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'],name
    profile=read('profile.json'); data=read('observations.json'); diag=read('model-diagnostics.json')
    assert profile['default_enabled'] is False and profile['policy']['improvement_ratio']==0.95
    assert len(data['cases'])==28 and profile['source_profile_id']=='67127e819f80a395aeecae55cd999c8e3f8b1bcf894fdf0672c58611cca87f66'
    assert data['engineering_acceptance']['actual_automatic_policy_launch_observed'] is False
    original={(x['cohort'],x['case']['id']):x for x in data['cases']}
    metrics=defaultdict(list); native_count=0; torch_count=0; exclusions=defaultdict(int); cold_count=0
    for case in data['cases']:
        if case['cohort']=='fresh-repeat':
            initial=original['heldout',case['case']['id']]
            assert case['decision']==initial['decision']
            chosen=check_decision(initial,profile['coefficients'],case['decision'])
        else: chosen=check_decision(case,profile['coefficients'],case['decision'])
        options={o['stage']:o for o in case['options']}
        for option in case['options']:
            close(median(option['samples']),option['p50']); native_count+=1
            source=option['receipts']; route=source['native']
            assert source['status']=='validated' and source['original_case_status']=='passed'
            same(option['samples'],route['event_us']['samples']); close(option['p50'],route['event_us']['p50'])
            correct=route['recorded_correctness']
            assert correct['errors']==0 and correct['inputs_unchanged'] and correct['guards_unchanged']
            assert route['saved_output_recheck']['failed_elements']==0
            cold_count+=bool(route.get('cold_ms'))
            if option['admitted_to_model']: same(features(case,option['resources'],option['requested_threads']),option['features'])
            else: exclusions[option['exclusion_reason']]+=1
            if option['requested_threads']:
                selection=source['cub_selection']; assert len(selection)==1
                assert selection[0]['selected']==option['observed_cub_selected']
                assert selection[0]['threads_requested']==option['requested_threads']
        torch=case['torch']; close(median(torch['event_us']['samples']),torch['event_us']['p50']); torch_count+=1
        assert torch['saved_output_recheck']['failed_elements']==0
        for c in torch['recorded_correctness'].values(): assert c['failed_elements']==0
        chosen_option=next((o for o in case['options'] if o['requested_threads']==chosen and chosen!=0),options['default'])
        close(chosen_option['p50'],case['policy']['p50'])
        first=options['default']['p50']; last=options['recheck']['p50']; timed=chosen_option['p50']
        initial_ratio=timed/first if chosen else 1.0; recheck_ratio=timed/last if chosen else 1.0
        close(initial_ratio,case['policy']['ratio_over_initial']); close(recheck_ratio,case['policy']['ratio_over_recheck'])
        close(last/first-1,case['recheck_drift'])
        if case['cohort']!='calibration':
            close(timed/torch['event_us']['p50'],case['policy']['ratio_over_same_round_torch_initial'])
            assert initial_ratio<=1.03 and recheck_ratio<=1.03 and abs(case['recheck_drift'])<=0.03
            for control in ('default','recheck'):
                interval=[min(chosen_option['samples'])/max(options[control]['samples']),max(chosen_option['samples'])/min(options[control]['samples'])]
                key='initial' if control=='default' else 'recheck'
                same(interval,case['policy']['sample_extrema_ratio_intervals'][key])
                assert not(interval[0]<=1.03<=interval[1])
        if case['cohort']=='fresh-repeat':
            assert case['frozen_choice_execution_consistent']
            assert case['torch']['result_receipt']!=original['heldout',case['case']['id']]['torch']['result_receipt']
        metrics[case['cohort']].append({'group':case['group'],'initial':initial_ratio,'recheck':recheck_ratio,'torch':timed/torch['event_us']['p50']})
    actual={}
    for name,rows in metrics.items():
        m={key:geometry_gm(rows,key) for key in ('initial','recheck','torch')}; actual[name]=m
        recorded=data['expected_metrics'][name]
        close(m['initial'],recorded['geomean_over_initial']); close(m['recheck'],recorded['geomean_over_recheck'])
        if name!='calibration':
            close(m['torch'],recorded['policy_over_same_round_torch_initial'])
            assert m['initial']<=0.95 and m['recheck']<=0.95
    # Replay every retained calibration and LOGO prediction from frozen models,
    # not an optimizer. Geometry separation is explicitly checked per fold.
    observations={x['record_id']:x for x in diag['observations']}
    residual_count=0
    def residuals(rows,models):
        nonlocal residual_count
        for row in rows:
            o=observations[row['record_id']]
            predicted=score(models[o['family']]['coefficients'],o['features'])
            close(predicted,row['predicted_us']); close(o['measured_us'],row['observed_us'])
            close(predicted-o['measured_us'],row['signed_error_us']); close(predicted/o['measured_us']-1,row['relative_error'])
            residual_count+=1
    residuals(diag['full_training_evaluation']['residuals'],diag['full_training_models'])
    for fold in diag['leave_one_geometry_out']['folds']:
        for family,m in fold['models'].items():
            assert fold['held_group'] not in m['training_groups']
            assert all(observations[i]['group']!=fold['held_group'] for i in m['training_record_ids'])
        residuals(fold['residuals'],fold['models'])
        for d in fold['decisions']:
            case=original['calibration',d['case_id']]
            check_decision(case,{f:m['coefficients'] for f,m in fold['models'].items()},d['decision'])
    assert native_count==144 and torch_count==28
    print(json.dumps({'status':'passed_cpu_projection_replay','native_observations':native_count,'torch_observations':torch_count,
                      'raw_samples':7*(native_count+torch_count),'frozen_residuals_replayed':residual_count,'exclusions':dict(exclusions),
                      'metrics':actual,'automatic_policy_timing_in_this_dataset':False,
                      'separate_implementation_correctness_receipt':read('implementation-validation.json')['automatic_runtime_policy_correctness_tested']},indent=2))

if __name__=='__main__': main()
