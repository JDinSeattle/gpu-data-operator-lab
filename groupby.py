from __future__ import annotations
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

SCHEMA=pa.schema([('key',pa.int64()),('value',pa.int64())])
NAMES=['key','sum','count_valid','count_all']

def table(keys,values):return pa.Table.from_arrays([pa.array(keys,type=pa.int64()),pa.array(values,type=pa.int64())],schema=SCHEMA)

def validate(data):
    if data.schema!=SCHEMA:raise ValueError('expected nullable int64 key/value schema; floating NaN is not null')
    if data.num_rows>2**31-1:raise ValueError('libcudf row/count range exceeded')
    # Conservative contract: absolute sum bound fits int64 for every possible grouping.
    bounds=pc.min_max(data['value']).as_py()
    maximum=max(abs(bounds['min'] or 0),abs(bounds['max'] or 0))
    if maximum*data.num_rows>2**63-1:raise OverflowError('int64 sum bound exceeded')

def generate(spec,seed=1729):
    if spec['rows']<0 or spec['groups']<1 or not 0<=spec['skew']<=1 or not 0<=spec['nulls']<=1:raise ValueError('invalid workload')
    rng=np.random.default_rng(seed);n=spec['rows']
    keys=rng.integers(0,spec['groups'],n,dtype=np.int64)
    keys[rng.random(n)<spec['skew']]=0
    values=rng.integers(-100000,100001,n,dtype=np.int64)
    return pa.Table.from_arrays([pa.array(keys,mask=rng.random(n)<.02),pa.array(values,mask=rng.random(n)<spec['nulls'])],schema=SCHEMA)

def edge_cases():
    return {
        'empty':table([],[]),'all-null':table([None,None],[None,None]),
        'null-groups':table([1,1,2,None,None],[None,None,3,4,None]),
        'duplicates':table([2,1,2,1,2],[5,2,-5,None,0]),
        'numeric-boundaries':table([0,0,1,1],[2**31-1,2**31-1,-2**31,-2**31]),
        'single-skew':table([7]*10001,[None]+[1]*10000),
        'null-key-values':table([None,None,0,0],[0,-1,None,0]),
    }

def normalize(rows):
    seen=set()
    for row in rows:
        if row['key'] in seen:raise AssertionError('duplicate output group')
        seen.add(row['key'])
    return sorted(rows,key=lambda row:(row['key'] is None,row['key'] if row['key'] is not None else 0))

def oracle(data):
    validate(data);groups={}
    for key,value in zip(data['key'].to_pylist(),data['value'].to_pylist()):
        row=groups.setdefault(key,{'key':key,'sum':None,'count_valid':0,'count_all':0});row['count_all']+=1
        if value is not None:row['sum']=(row['sum'] or 0)+value;row['count_valid']+=1
    return normalize(list(groups.values()))

def arrow_aggregate(data):
    return data.group_by('key',use_threads=True).aggregate([
        ('value','sum'),('value','count'),('value','count',pc.CountOptions(mode='all'))]).rename_columns(NAMES)

def compare(actual,expected,data):
    actual=normalize(actual);expected=normalize(expected)
    if actual!=expected:raise AssertionError(f'group mismatch: {actual[:5]} != {expected[:5]}')
    if sum(x['count_all'] for x in actual)!=data.num_rows:raise AssertionError('row conservation failed')
    if sum(x['count_valid'] for x in actual)!=data.num_rows-data['value'].null_count:raise AssertionError('valid-value conservation failed')
    return {'groups':len(actual),'rows':data.num_rows,'valid_values':data.num_rows-data['value'].null_count,'exact':True}

def ipc(data):
    sink=pa.BufferOutputStream()
    with pa.ipc.new_stream(sink,data.schema) as writer:writer.write_table(data)
    return sink.getvalue()

def fingerprint(data):return hashlib.sha256(ipc(data)).hexdigest()

def consume(data):
    # A small downstream checksum consumes all output columns; not a Python per-row loop.
    values=[pc.sum(data.column(i)).as_py() for i in range(data.num_columns)]
    return tuple(values)

class GPUGroupBy:
    def __init__(self,data,budget_bytes=None):
        import pylibcudf as p
        import cupy as cp
        validate(data)
        # Observable preflight capacity policy; conservative estimate is not an allocation guarantee.
        self.estimated_bytes=1024**2+data.num_rows*96
        available=int(cp.cuda.runtime.memGetInfo()[0]) if budget_bytes is None else budget_bytes
        if self.estimated_bytes>available:raise MemoryError(f'capacity budget {available} < estimate {self.estimated_bytes}')
        self.p=p;self.host=data;self.device=None

    def upload(self):
        self.device=self.p.Table.from_arrow(self.host);return self.device

    def aggregate(self):
        p=self.p
        if self.device is None:raise RuntimeError('upload first')
        grouped=p.groupby.GroupBy(p.Table([self.device.columns()[0]]),null_handling=p.types.NullPolicy.INCLUDE)
        keys,values=grouped.aggregate([p.groupby.GroupByRequest(self.device.columns()[1],[
            p.aggregation.sum(),p.aggregation.count(p.types.NullPolicy.EXCLUDE),p.aggregation.count(p.types.NullPolicy.INCLUDE)])])
        return p.Table(keys.columns()+values[0].columns())

    @staticmethod
    def download(result):return result.to_arrow().rename_columns(NAMES)

    def run(self):
        self.upload();return self.download(self.aggregate())
