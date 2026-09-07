# Exact integer groupby contract

The ETL workload groups nullable int64 campaign/account keys and nullable int64 values representing cents. It computes sum, valid-value count and total-row count. The input is an Arrow table, not Python objects converted differently for each backend. Floating NaN is rejected by schema and never silently changed into null.

| Case | Expected semantics |
|---|---|
| Null key | Included as one group |
| Null value | Excluded from sum and valid count; retained in row count |
| All-null values | Sum is null, valid count zero, row count preserved |
| Empty table | No groups |
| Duplicate key | One aggregated output group |
| Output order | Unspecified; normalize by key with null last |
| Overflow | Conservative `max(abs(value)) * rows <= INT64_MAX` preflight; otherwise explicit failure |
| Capacity | Conservative estimate `1 MiB + 96 * rows` checked against available memory/budget; allocator failure still propagates |
| Oversized data | No silent truncation or fallback; no claimed chunked mode |

The Python dictionary oracle accumulates with arbitrary-width integers. Arrow's multithreaded hash groupby is a separate CPU implementation and the actual performance comparator. GPU operations use `pylibcudf.Table.from_arrow`, `GroupBy(..., null_handling=INCLUDE)` and explicit sum/count aggregations. The [pylibcudf groupby contract](https://docs.nvidia.com/cudf/latest/pylibcudf/api_docs/groupby/) supports the null-key choice and table/request structure. Installed pylibcudf 26.8.1 was inspected: Arrow interop lives on Table/Column, not the old interop module.

Exact normalized groups and row/valid-value conservation must all pass. Duplicate output keys are rejected; sorting must not conceal extra groups. Mutations cover dropped null keys, converting null sums to zero, wrong count policy, duplicated groups and corrupted sums.

## Measurement boundary

GPU timings separately measure upload, resident groupby and download. Device events record an API span including host gaps; they are not labeled kernel-only time. A separate Nsight Systems trace records actual CUDA kernels and memory operations, with H2D/groupby/D2H NVTX ranges. The trace is diagnostic and is not included in the benchmark sample pool.

The end-to-end comparison reads the same Arrow IPC buffer, aggregates, then consumes all output columns with a checksum. Both CPU and GPU paths include load and output consumption. GPU capacity/schema validation is included in its ETL timing. Output sorting is only part of correctness checking, not the timed operator.

RMM counters are measured in a separate untimed run and cover libcudf-owned allocations. They exclude the process CUDA context, CuPy, the desktop compositor and other applications. The memory-budget rejection test is a controlled preflight failure, not a claim to have exhausted the physical GPU.

The matrix includes 1K, 100K and 1M rows, many small groups, 95% skew with 50% value nulls, and a near-single-group workload (the independent 2% null-key group remains). The crossover is a measured region on this host, not a universal threshold. Integer cents make reconciliation exact; this says nothing about every SQL, floating point, decimal, join or streaming semantic.
