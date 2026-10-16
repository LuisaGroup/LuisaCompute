"""CPU-only public heldout receipt, timing and frozen-policy reproduction; no refitting."""
import argparse, hashlib, json, math, re, statistics
from pathlib import Path
FIT='63e0677c8554b46b707aa9f1fca3f29f223c653b1ec35fd57b5534b79595b6c2'
PROFILE='sm89-24-cuda134-partition-linear-v1'
COEFFICIENTS=[0.9028899747245084,0.0,0.00011327031090119965]

def require(ok,message):
    if not ok:raise ValueError(message)

def close(a,b):return math.isclose(a,b,rel_tol=2e-13,abs_tol=2e-13)

def prediction(rows,width,old_rows):
    def score(b):
        programs=(rows+b-1)//b
        waves=(programs+23)//24
        return COEFFICIENTS[0]+COEFFICIENTS[1]*programs+COEFFICIENTS[2]*waves*b*width
    original=score(old_rows);best=original;chosen=old_rows
    for b in (1,2):
        q=score(b)
        if q<best:chosen,best=b,q
    if chosen!=old_rows and best/original<.95:return dict(status='selected',reason='predicted-saving',original_rows=old_rows,selected_rows=chosen,original_score=original,selected_score=best)
    return dict(status='retained',reason='predicted-original',original_rows=old_rows,selected_rows=old_rows,original_score=original,selected_score=original)

def decision_check(case,baseline,cohort):
    realization=baseline['routes']['native']['realization']
    fields=re.search(r'collective-work-v1: programs=(\d+)',realization)
    collective=re.findall(r'collective=kind(\d+):width(\d+):independent(\d+)',realization)
    require(fields and len(collective)==1,'missing original IR collective facts')
    _,width,old_rows=map(int,collective[0]);rows=case['dimensions'][0]
    require(width==case['tile'][1] and old_rows==case['tile'][0] and int(fields[1])==(rows+old_rows-1)//old_rows,'IR/case geometry mismatch')
    expected=prediction(rows,width,old_rows)
    decisions=cohort['policy_decisions'];require(len(decisions)==1,'one policy decision required')
    d=decisions[0];require(d['fit']==FIT and d['profile']==PROFILE,'frozen fit/profile mismatch')
    for key,value in expected.items():
        require(close(d[key],value) if isinstance(value,float) else d[key]==value,'policy differs from frozen formula: '+key)
    return expected

def ratio_check(comparison,before,after):
    require(comparison['status']=='valid_matched_pair','invalid pair')
    ratio=after/before
    require(close(comparison['candidate_over_default'],ratio) and close(comparison['delta_percent'],100*(ratio-1)),'recorded ratio differs from medians')
    return ratio

def validate(summary):
    require(summary['status']=='completed_validated_native' and summary['native_revalidation']['status']=='passed','not a completed native revalidation')
    require(all(k in summary for k in ('preserved_failed_rescue','preserved_schema_failure','corrected_retry_queue')),'schema failure/continuation provenance was lost')
    require(summary['native_revalidation']['original_cohort_status']=='failed','original failure was lost')
    failures=summary['native_revalidation']['original_failures'];require(len(failures)==2,'expected two preserved process failures')
    failed_ids={r['case']['id'] for r in failures}
    for r in failures:
        p=r['runs']['torch']['process'];require(p['status']=='exited' and (p['returncode']&0xffffffff)==0xc000070a,'failure changed classification')
    require(len(summary['cases'])==12 and len({r['case']['id'] for r in summary['cases']})==12,'wrong or duplicate inventory')
    native=torch=samples=0;table=[]
    for row in summary['cases']:
        case=row['case'];cohorts=row['cohorts'];require(set(cohorts)=={'default','policy','recheck','policy-repeat'},'four exact cohorts required')
        baseline=cohorts['default'];require(baseline['original_cohort_status']=='failed','original cohort relabeled')
        for name,c in cohorts.items():
            require(c['status']=='validated' and c['native_comparison_eligible'],'unvalidated route')
            require(c['manifest']==baseline['manifest'] and c['fixture_sha256']==baseline['fixture_sha256'],'fixture changed')
            require(set(c['source_files'])==set(baseline['source_files']),'source list changed')
            for key,source in c['source_files'].items():
                require(source['original_source_sha256']==baseline['source_files'][key]['normalized_source_sha256'],'original source prefix changed')
            if name in ('policy','policy-repeat'):decision_check(case,baseline,c)
            else:require(not c['policy_decisions'],'default has a policy decision')
        require(cohorts['policy']['policy_decisions']==cohorts['policy-repeat']['policy_decisions'],'repeat decision changed')
        require(cohorts['policy']['selection']==cohorts['policy-repeat']['selection'],'repeat selection changed')
        require(cohorts['policy']['source_files']==cohorts['policy-repeat']['source_files'] or all(cohorts['policy']['source_files'][k]['sha256']==cohorts['policy-repeat']['source_files'][k]['sha256'] for k in cohorts['policy']['source_files']),'repeat source changed')
        groups=list(cohorts.items())
        if case['id'] in failed_ids:
            require(baseline['routes']['torch']['status']=='failed' and baseline['routes']['torch']['comparison']=='missing_initial_torch_baseline','initial denominator was filled')
            retry=row['independent_torch_retest'];groups.append(('independent-retry',retry['evidence']))
            require(retry['evidence']['status']=='validated' and not retry['evidence']['policy_decisions'],'unvalidated retry or policy applied to retry')
            require(retry['evidence']['manifest']==baseline['manifest'] and retry['evidence']['fixture_sha256']==baseline['fixture_sha256'],'retry fixture changed')
            require(set(retry['evidence']['source_files'])==set(baseline['source_files']) and all(retry['evidence']['source_files'][k]['original_source_sha256']==baseline['source_files'][k]['normalized_source_sha256'] for k in baseline['source_files']),'retry source changed')
            ratio_check(retry['native_retest_vs_initial'],baseline['routes']['native']['event_us']['p50'],retry['evidence']['routes']['native']['event_us']['p50'])
        else:require('independent_torch_retest' not in row,'unexpected retry')
        for name,c in groups:
            for route,result in c['routes'].items():
                if 'event_us' not in result:continue
                ts=result['event_us']['samples'];require(len(ts)==7 and all(math.isfinite(x) and x>0 for x in ts),'seven finite samples required')
                require(statistics.median(ts)==result['event_us']['p50'],'median differs from raw samples')
                require(result['saved_output_recheck']['failed_elements']==0,'saved output failed original oracle')
                native+=route=='native';torch+=route=='torch';samples+=7
        base=baseline['routes']['native']['event_us']['p50'];recheck=cohorts['recheck']['routes']['native']['event_us']['p50']
        ratios={}
        for name in ('policy','recheck','policy-repeat'):
            ratios[name]=ratio_check(row['comparisons'][name],base,cohorts[name]['routes']['native']['event_us']['p50'])
        for name in ('policy','policy-repeat'):
            ratios[name+'_vs_recheck']=ratio_check(row['comparisons'][name+'_vs_recheck'],recheck,cohorts[name]['routes']['native']['event_us']['p50'])
        repeat=row['comparisons']['policy_repeat_consistency'];r=cohorts['policy-repeat']['routes']['native']['event_us']['p50']/cohorts['policy']['routes']['native']['event_us']['p50']
        require(repeat['status']=='valid_policy_repeat' and close(repeat['repeat_over_initial'],r),'repeat ratio mismatch')
        d=cohorts['policy']['policy_decisions'][0]
        table.append(dict(id=case['id'],case=case,status=d['status'],selected_rows=d['selected_rows'],default_us=base,
                          policy_us=cohorts['policy']['routes']['native']['event_us']['p50'],repeat_us=cohorts['policy-repeat']['routes']['native']['event_us']['p50'],recheck_us=recheck,
                          initial_torch_us=baseline['routes']['torch'].get('event_us',{}).get('p50'),
                          retry_torch_us=row.get('independent_torch_retest',{}).get('evidence',{}).get('routes',{}).get('torch',{}).get('event_us',{}).get('p50'),ratios=ratios))
    require((native,torch,samples)==(50,12,434),'execution/sample counts changed')
    selected=[r for r in table if r['status']=='selected'];retained=[r for r in table if r['status']=='retained']
    gm=lambda rows,key:math.exp(sum(math.log(r['ratios'][key]) for r in rows)/len(rows)) if rows else None
    groups={}
    for r in table:
        c=r['case'];key=(c['dimensions'][0],c['dimensions'][1],c['tile'][0],c['tile'][1])
        groups.setdefault(key,[]).append(r)
    geometry_gm=lambda key:math.exp(sum(math.log(gm(g,key)) for g in groups.values())/len(groups))
    return dict(status='passed',native_executions=native,torch_executions=torch,samples=samples,fit=FIT,selected=len(selected),retained=len(retained),
                independent_geometry_groups=len(groups),geometry_weighted_policy_over_default=geometry_gm('policy'),geometry_weighted_repeat_over_default=geometry_gm('policy-repeat'),
                policy_over_default_geomean=gm(table,'policy'),repeat_over_default_geomean=gm(table,'policy-repeat'),
                selected_policy_geomean=gm(selected,'policy'),retained_policy_geomean=gm(retained,'policy'),
                selected_regressions=[r['id'] for r in selected if r['ratios']['policy_vs_recheck']>1.05 and r['ratios']['policy-repeat_vs_recheck']>1.05],table=table)

def check_receipts(folder):
    package=json.loads((folder/'receipts.json').read_text())
    for name,recorded in package['files'].items():
        p=(folder/name).resolve();require(p.is_relative_to(folder.resolve()) and p.is_file(),'receipt path escaped/missing')
        data=p.read_bytes();require(hashlib.sha256(data).hexdigest()==recorded['sha256'] and len(data)==recorded['bytes'],'public bytes changed: '+name)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
    check_receipts(a.package);summary=json.loads((a.package/'validation.json').read_text());result=validate(summary)
    recorded=json.loads((a.package/'audit.json').read_text());require(result==recorded,'audit projection differs')
    print(json.dumps({k:v for k,v in result.items() if k!='table'}));return 0
if __name__=='__main__':raise SystemExit(main())
