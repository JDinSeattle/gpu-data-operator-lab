import copy
import pytest,pyarrow as pa
from groupby import table,edge_cases,oracle,arrow_aggregate,compare,validate,generate,ipc,fingerprint

@pytest.mark.parametrize('data',list(edge_cases().values()),ids=list(edge_cases()))
def test_arrow_matches_independent_integer_oracle(data):
    compare(arrow_aggregate(data).to_pylist(),oracle(data),data)

@pytest.mark.parametrize('seed',range(10))
def test_randomized_conservation(seed):
    data=generate({'rows':101,'groups':17,'skew':seed/10,'nulls':seed/10},seed)
    compare(arrow_aggregate(data).to_pylist(),oracle(data),data)
    assert fingerprint(data)==fingerprint(pa.ipc.open_stream(ipc(data)).read_all())

@pytest.mark.parametrize('mutation',['drop-null-key','zero-null-sum','count-all-as-valid','duplicate-group','corrupt-sum'])
def test_detect_semantic_mutations(mutation):
    data=edge_cases()['null-groups'];expected=oracle(data);rows=copy.deepcopy(expected)
    if mutation=='drop-null-key':rows=[r for r in rows if r['key'] is not None]
    if mutation=='zero-null-sum':
        for r in rows:
            if r['sum'] is None:r['sum']=0
    if mutation=='count-all-as-valid':
        for r in rows:r['count_valid']=r['count_all']
    if mutation=='duplicate-group':rows.append(copy.deepcopy(rows[0]))
    if mutation=='corrupt-sum':rows[1]['sum']+=1
    with pytest.raises(AssertionError):compare(rows,expected,data)

def test_float_nan_not_null_and_overflow():
    with pytest.raises(ValueError):validate(pa.table({'key':[1],'value':[float('nan')]}))
    with pytest.raises(OverflowError):validate(table([1,1],[2**62,2**62]))

def test_all_null_sum_is_not_zero():
    assert oracle(table([1],[None]))==[{'key':1,'sum':None,'count_valid':0,'count_all':1}]
