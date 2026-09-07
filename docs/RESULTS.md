# Measured qualification report

Run: `verified-final-20260907`. Execution status: **passed**.

[All commands and exit codes](../results/verified-final-20260907/execution.json) · [CPU test log](../results/verified-final-20260907/cpu-tests.log) · [GPU correctness](../results/verified-final-20260907/gpu-check.json) · [Raw timing](../results/verified-final-20260907/benchmark.json)

Source digest: `369b38036acba129fb0ecbe2cc50085d1054ce2e2abad429bbaa852f27845f82`.

Hardware: RTX 4090 (sm_89), driver 595.84, native toolkit CUDA 13.2. Clocks unchanged; display GPU. Results apply to this run and workload only.

CPU: 26 tests. All command return codes are retained; missing prerequisites fail the reproduction driver.

12 CPU/Arrow/pylibcudf cases passed with exact groups, null policy and count conservation. Controlled capacity rejection passed.

| Workload | Arrow operator ms | GPU operator ms | Arrow ETL ms | GPU ETL ms | ETL speedup | RMM peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| small-uniform | 0.143 | 0.436 | 0.200 | 0.745 | 0.27× | 0.04 |
| medium-uniform | 1.141 | 0.868 | 1.326 | 2.016 | 0.66× | 4.35 |
| large-uniform | 14.321 | 2.043 | 15.356 | 6.933 | 2.21× | 42.33 |
| large-skew | 13.090 | 3.040 | 13.945 | 9.665 | 1.44× | 42.33 |
| large-one-group | 11.888 | 2.094 | 12.468 | 5.303 | 2.35× | 34.70 |

| Workload | Upload wall ms | Download wall ms |
|---|---:|---:|
| small-uniform | 0.125 | 0.055 |
| medium-uniform | 0.493 | 0.125 |
| large-uniform | 2.723 | 0.479 |
| large-skew | 2.620 | 0.126 |
| large-one-group | 2.951 | 0.131 |

[CUDA kernel trace summary](../results/verified-final-20260907/profile_cuda_gpu_kern_sum.csv) · [CUDA transfer summary](../results/verified-final-20260907/profile_cuda_gpu_mem_time_sum.csv) · [NVTX phase summary](../results/verified-final-20260907/profile_nvtx_sum.csv)

Wall medians include Python API dispatch. GPU event spans are in raw JSON; kernel-only timings are in the separate diagnostic trace. Crossover conclusions are restricted to these rows, group counts, null ratios and this display GPU. RMM peak excludes CUDA context and other applications.

## Safety and reproducibility

- [gpu-memcheck log](../results/verified-final-20260907/gpu-memcheck.log): exit 0.

The reproduction driver fails on missing GPU, failed checks, stale gates, sanitizer errors or subprocess failure. CPU CI is labeled separately. All performance comparisons retain failures and unsupported cases.
