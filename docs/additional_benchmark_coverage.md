# Additional Benchmark Coverage for Registry Models

This document records manual runs of registry models that were originally outside the `FULL_MATRIX` used by the full benchmark GitHub Action. Each model was exercised with:

- Command: `python eval.py --model <model> --dataset synthetic --config quick_benchmark_config.json`
- Configuration: `quick_benchmark_config.json` (1 iteration, 0 warmup iterations, 1 training epoch, 50 samples)

## Results

| Model | Status | Notes |
| --- | --- | --- |
| `ccl_cpc` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `cevae_m` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `cnflow` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `crf` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `crf_discrete` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `ctm_t` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `cycle_vat` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `deconfounder_cfm` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `diffusion_gnn_scm` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `eg_ddi` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `em` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `gflownet_treatment` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `gnn_ebm` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `jsbf` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `lp_knn` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `lt_flow_diff` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `m2_vae` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `scgm` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |
| `ss_dml` | ❌ Failed | Requires the optional `doubleml` dependency (`pip install xtylearner[causal]`). |
| `vime` | ✅ Success | Completed 1 measurement iteration on `synthetic`. Now part of the full benchmark matrix. |

Only `ss_dml` failed because the optional `doubleml` package is not installed in the benchmark environment. All successful models above have been promoted to the full benchmark matrix so they now run automatically.
