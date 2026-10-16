"""Stdlib verification of V5 public bytes, all measurements and recorded entry proof."""
import csv
import json
from pathlib import Path
import re
from verify_common import require,load,near,series,sampling,archive_refs,package_path,verify_files

STAGES=('plain_original','aligned_original','plain_C1024','aligned_C1024','plain_C2048','aligned_C2048','plain_original_recheck')

def observation(root,stage,row):
    a=stage['alignment_observation'];obs=a['observed'];aligned=stage['alignment_mode']
    entry='luisa_tile_aligned16' if aligned else 'luisa_tile_main'
    require(obs['compile_mode']==obs['buffer_mask']==obs['partition_loads']==aligned and obs['selected_entry']==entry and
        obs['actual_graph_nodes_checked']==100 and obs['same_module_function_identity_checked'] and obs['actual_final_pointers_checked'] and
        obs['grid']==[row['dimensions'][0],1,1] and obs['block']==[1,1,1] and obs['shared_bytes']==0,'actual selection claim differs')
    require(load(package_path(root,a['included']['diagnostic-alignment.json']['file']))==obs,'included descriptor differs')
    with package_path(root,a['included']['diagnostic-graph.csv']['file']).open(newline='') as f:graph=list(csv.DictReader(f))
    require(len(graph)==100,'full graph node count missing')
    identities=set()
    for node in graph:
        require(node['entry']==entry and [int(node[k]) for k in ('grid_x','grid_y','grid_z')]==obs['grid'] and
            [int(node[k]) for k in ('block_x','block_y','block_z')]==[1,1,1] and int(node['shared_bytes'])==0 and
            int(node['alignment_mask'])==aligned,'actual graph descriptor changed')
        pointers=[int(node[f'arg{i}'],16) for i in range(4)]
        require(all(0<v<2**64 for v in pointers) and [v%16 for v in pointers]==obs['final_argument_mod16'],'actual pointers/residues differ')
        require(not aligned or pointers[0]%16==0,'aligned source selected without root alignment')
        identities.add((node['function'],*pointers))
    require(len(identities)==1,'graph nodes changed function or pointers')
    with package_path(root,a['included']['diagnostic-resources.csv']['file']).open(newline='') as f:resources=list(csv.DictReader(f))
    require(resources==a['resources'] and len(resources)==(8 if aligned else 5),'full resource records differ')
    require(len({r['module'] for r in resources})==1,'resource module differs')
    for name in ('luisa_tile_main','luisa_tile_aligned16'):
        rs=[r for r in resources if r['entry']==name]
        if not aligned and name.endswith('aligned16'):
            require(len(rs)==1 and rs[0]['function']=='none' and rs[0]['known']=='0' and rs[0]['value']=='unknown','absent entry became known')
        else:
            require(len(rs)==4 and {r['attribute'] for r in rs}=={'registers','static_shared_bytes','local_bytes','max_threads'} and
                len({r['function'] for r in rs})==1,'entry resource completeness differs')
    for r in resources:
        require((r['known']=='1' and int(r['value'])>=0 and r['cuda_status']=='0') or
                (r['known']=='0' and r['value']=='unknown'),'unknown resource was fabricated')
    require({r['function'] for r in resources if r['entry']==entry}=={next(iter(identities))[0]},'loaded entry not tied to graph')

def verify(root,data):
    require(data['status']=='completed_validated','closed checkpoint required')
    inventory=load(root/'cases8.json')['cases']
    require([r['case'] for r in data['cases']]==inventory and len(inventory)==8 and len({r['id'] for r in inventory})==8,'complete inventory required')
    require({(c['dimensions'][0],c['dimensions'][1],c['precision']) for c in inventory}==
        {(r,n,t) for r in (3,128) for n in (8192,16384) for t in ('fp16','bf16')},'case coverage differs')
    count=0
    for r in data['cases']:
        require(tuple(r['stages'])==STAGES,'all seven stages in original order required')
        fixture=None
        for name,s in r['stages'].items():
            chunk=1024 if 'C1024' in name else 2048 if 'C2048' in name else 0
            aligned=int(name.startswith('aligned_'))
            require(s['chunk']==chunk and s['alignment_mode']==aligned,'requested experiment differs')
            series(s['event_us']);series(s['graph_host_us']);series(s['stream_event_us']);series(s['host_wall_us'])
            sampling(s['sampling'],s['warmup'],s['event_us'],s['graph_host_us'])
            correctness=s['recorded_correctness']
            require(s['saved_output']['failed_elements']==correctness['errors']==0 and correctness['inputs_unchanged'] and
                correctness['guards_unchanged'] and correctness['all_outputs_finite'],'numerical/guard validation failed')
            manifest=load(package_path(root,s['manifest']['file']))
            for k in ('operation','dimensions','precision','pattern','seed'):require(manifest[k]==r['case'][k],'manifest case changed')
            if fixture is None:fixture=(s['fixture_sha256'],manifest['semantics'])
            else:require(fixture==(s['fixture_sha256'],manifest['semantics']),'same fixture/oracle/semantics failed')
            for filename,src in s['sources'].items():require(src['original_raw_sha256']==s['source_sha256'][filename],'source raw receipt differs')
            proof=s['diagnostic']['proof']
            require(load(package_path(root,s['diagnostic']['included']['receipt']['file']))==proof,'included IR proof differs')
            require(proof['chunk']==chunk and proof['transformed'] is bool(chunk),'IR request changed')
            if chunk:
                rows,width=r['case']['dimensions']
                require(proof['input_slot']==0 and proof['output_slot']==3 and proof['input_bytes']==rows*width*2 and
                    proof['output_bytes']==rows*2 and proof['programs']==rows and proof['serial_iterations']==width//chunk and
                    proof['collective_invocations_per_program']==1 and proof['contribution_elements_per_program']==chunk and
                    proof['largest_materialized_tile_elements']==chunk,'V4 logical transform facts changed')
                guard=load(package_path(root,s['diagnostic']['included']['disjoint_receipt']['file']))
                require(guard==dict(schema=1,actual_encoded_arguments_checked=True,whole_static_ranges_disjoint=True),'alias guard missing')
            observation(root,s,r['case']);count+=1
        ss=r['stages'];values={n:s['event_us']['p50'] for n,s in ss.items()}
        require(near(r['plain_control_recheck_over_initial'],values[STAGES[-1]]/values[STAGES[0]]),'drift changed')
        for stem in ('original','C1024','C2048'):
            plain=ss['plain_'+stem]['sources']['source.txt'];aligned=ss['aligned_'+stem]['sources']['source.txt']
            pb=package_path(root,plain['file']).read_bytes();ab=package_path(root,aligned['file']).read_bytes()
            require(pb.count(b'luisa_tile_main(')==1 and b'luisa_tile_aligned16' not in pb and
                ab.startswith(pb+b'\n') and ab.count(b'luisa_tile_main(')==1 and ab.count(b'luisa_tile_aligned16(')==1,'plain prefix changed')
            require(r['source_prefix_evidence'][stem]['plain']['sha256']==plain['original_raw_sha256'] and
                r['source_prefix_evidence'][stem]['aligned']['sha256']==aligned['original_raw_sha256'],'raw prefix receipt differs')
            require(near(r['layout_ratios'][stem],values['aligned_'+stem]/values['plain_'+stem]),'same-IR ratio changed')
        require(ss[STAGES[0]]['source_sha256']==ss[STAGES[-1]]['source_sha256'],'plain recheck source changed')
        for layout in ('plain','aligned'):
            for chunk in ('C1024','C2048'):
                name=layout+'_'+chunk
                require(near(r['ir_over_same_layout_original'][name],values[name]/values[layout+'_original']),'same-layout IR ratio changed')
        for name,v in values.items():
            require(near(r['all_stages_over_plain_initial'][name],v/values[STAGES[0]]) and
                near(r['all_stages_over_plain_recheck'][name],v/values[STAGES[-1]]),'plain denominator changed')
    archive_refs(root,data)
    require(count==56 and data['counts']['primary_graph_samples']==392 and data['counts']['torch_samples']==0,'sample total changed')
    return dict(native_measurements=count,primary_graph_samples=count*7,actual_graph_nodes=count*100,torch_measurements=0)

def static(root,data):
    forbidden={'graph','stream','native_recheck','nominal_input_bandwidth','native_over_fresh_torch','event_us'}
    def walk(v):
        if isinstance(v,dict):
            require(not forbidden.intersection(v),'timing denominator leaked into static appendix')
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(data);archive_refs(root,data)
    for c in data['cases']:
        require(len(c['kernels'])==c['generated_kernel_count'],'static kernel inventory differs')
        for k in c['kernels']:
            cfg=load(package_path(root,k['included']['best_config']['file']))
            require(cfg==k['best_config'],'retained config changed')
            meta=load(package_path(root,k['included']['.json']['file']))
            require(meta==k['metadata'] and meta['num_warps']==cfg['num_warps'],'specialization metadata differs')
            ptx=package_path(root,k['included']['.ptx']['file']).read_text()
            require(int(re.search(r'^\.reqntid (\d+)',ptx,re.M).group(1))==cfg['num_warps']*32,'ordinary thread count differs')
    return len(data['cases'])

def main():
    root=Path(__file__).resolve().parent
    verify_files(root)
    result=verify(root,load(root/'evidence.json'))
    result['separate_static_torch_cases']=static(root,load(root/'torch-static.json'))
    print(json.dumps(dict(status='passed',**result)))

if __name__=='__main__':main()
