# Interview and résumé evidence

Suggested résumé bullet:

> Built an Arrow/libcudf groupby qualification and ETL benchmark with exact integer reconciliation, null-policy and count-conservation tests; validated 12 CPU/GPU workloads through one million rows and measured transfer, kernel, end-to-end and RMM memory costs.

This supports GPU data-library and data-infrastructure roles. Connect it to prior reconciliation/dataflow experience through invariants and evidence, not by presenting a simple framework swap as a new database.

Be ready to explain why all-null sum is null, count(valid) differs from count(all), and null keys must be included consistently. Demonstrate how sorting normalizes order without concealing duplicate groups. Explain the conservative overflow/capacity contract and why rejecting a budget is different from exhausting physical VRAM.

Open `docs/RESULTS.md` and identify where CPU is preferable. Distinguish a CUDA event around a Python call from actual kernel duration in the Nsight trace. RMM peak measures allocations belonging to the operation, not total process/device memory. Never generalize this integer groupby evidence to all SQL, joins, NaN or decimal semantics.
