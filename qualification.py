from __future__ import annotations
import argparse,json,time,random,dataclasses,gc
import pyarrow as pa
from evidence import ROOT,environment,save,require_gate,stats
from groupby import generate,edge_cases,oracle,arrow_aggregate,compare,GPUGroupBy,fingerprint,ipc,consume
MANIFEST=json.loads((ROOT/'manifests/workloads.json').read_text())

def datasets():
    yield from edge_cases().items()
    for spec in MANIFEST['benchmark']:yield spec['id'],generate(spec,MANIFEST['seed'])

def check():
    rows=[]
    for name,data in datasets():
        ref=oracle(data);cpu=arrow_aggregate(data);gpu=GPUGroupBy(data).run()
        rows.append({'case':name,'fingerprint':fingerprint(data),'cpu':compare(cpu.to_pylist(),ref,data),
                     'gpu':compare(gpu.to_pylist(),ref,data),'schema':str(data.schema)})
    data=next(iter(edge_cases().values()))
    try:GPUGroupBy(data,budget_bytes=1)
    except MemoryError as exc:budget={'status':'expected rejection','reason':str(exc)}
    else:raise AssertionError('capacity rejection did not occur')
    return {'status':'passed','environment':environment(),'records':rows,'capacity_test':budget}

def benchmark(gate,samples):
    require_gate(gate)
    import cupy as cp,rmm.statistics as memory
    results=[];rng=random.Random(1729)
    memory.enable_statistics()
    for spec in MANIFEST['benchmark']:
        data=generate(spec,MANIFEST['seed']);serialized=ipc(data);plan=GPUGroupBy(data);plan.upload();resident=plan.aggregate()
        def cpu_etl():return consume(arrow_aggregate(pa.ipc.open_stream(serialized).read_all()))
        def gpu_etl():return consume(GPUGroupBy(pa.ipc.open_stream(serialized).read_all()).run())
        functions={'arrow-operator':lambda:arrow_aggregate(data),'gpu-upload':plan.upload,
                   'gpu-operator':plan.aggregate,'gpu-download':lambda:plan.download(resident),'arrow-etl':cpu_etl,'gpu-etl':gpu_etl}
        for f in functions.values():
            for _ in range(3):f()
        cp.cuda.runtime.deviceSynchronize();raw={k:{'wall':[],'event':[]} for k in functions};order=[]
        for _ in range(samples):
            keys=list(functions);rng.shuffle(keys);order.append(keys)
            for key in keys:
                cp.cuda.runtime.deviceSynchronize();start=cp.cuda.Event();end=cp.cuda.Event()
                t=time.perf_counter_ns();start.record();output=functions[key]();end.record();end.synchronize()
                raw[key]['wall'].append((time.perf_counter_ns()-t)/1e6)
                if key.startswith('gpu-'):raw[key]['event'].append(cp.cuda.get_elapsed_time(start,end))
                del output
        # Memory accounting is a separate untimed execution. RMM covers libcudf allocations, not CUDA context/other apps.
        gc.collect();memory.push_statistics()
        measured=GPUGroupBy(data);measured.upload();device_output=measured.aggregate();host_output=measured.download(device_output)
        cp.cuda.runtime.deviceSynchronize();peak=dataclasses.asdict(memory.get_statistics())
        del measured,device_output,host_output;memory.pop_statistics()
        results.append({'case':spec,'fingerprint':fingerprint(data),'arrow_threads':pa.cpu_count(),'order':order,
                        'measurements':{k:{'wall':stats(v['wall']),**({'cuda_event_span':stats(v['event'])} if v['event'] else {})} for k,v in raw.items()},
                        'rmm_memory':peak,'estimated_capacity_bytes':plan.estimated_bytes})
    return {'status':'measured','environment':environment(),'results':results,
            'timing_scope':'GPU event span includes host dispatch gaps; kernel-only duration comes from separate Nsight Systems trace; wall ETL includes Arrow IPC load and checksum',
            'memory_scope':'per-workload RMM allocations; excludes process CUDA context and other applications'}

def profile_case():
    import nvtx,cupy as cp
    data=generate(MANIFEST['benchmark'][2]);plan=GPUGroupBy(data)
    with nvtx.annotate('H2D'):plan.upload();cp.cuda.runtime.deviceSynchronize()
    with nvtx.annotate('groupby'):result=plan.aggregate();cp.cuda.runtime.deviceSynchronize()
    with nvtx.annotate('D2H'):plan.download(result);cp.cuda.runtime.deviceSynchronize()

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['check','bench','profile']);p.add_argument('--gate')
    p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=21);a=p.parse_args()
    try:
        result=check() if a.phase=='check' else (benchmark(a.gate,a.samples) if a.phase=='bench' else profile_case())
        save(a.output,result or {'status':'profiled','environment':environment()})
    except Exception as exc:
        save(a.output,{'status':'failed','environment':environment(),'error':repr(exc)});raise
if __name__=='__main__':main()
