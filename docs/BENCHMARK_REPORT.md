# LatencyOps Sample Benchmark Report

This report documents a small, reproducible sample comparison for the LatencyOps public reference materials. It is an example of the reporting format, not a production performance claim or model-quality ranking.

## Scope

- Recorded: 2026-09-03
- Workload: synthetic single-sentence streaming prompt
- Runs: 3 per model
- Prompt and response retention: none
- Measurements: TTFT, TPOT, and end-to-end latency
- Quality evaluation: not configured for this aggregate comparison
- Provider paths: OpenAI API and LiteLLM/Qwen vLLM

## Aggregate results

| Provider | Model | Runs | Success | TTFT p50 / p95 | TPOT p50 | End-to-end p50 / p95 |
|---|---|---:|---:|---:|---:|---:|
| OpenAI | `gpt-4o-mini` | 3 | 100% | 935.19 / 1634.29 ms | 7.58 ms | 939.58 / 1643.24 ms |
| OpenAI | `gpt-4.1-mini` | 3 | 100% | 753.22 / 945.96 ms | 8.82 ms | 760.25 / 954.65 ms |
| OpenAI | `gpt-5-mini` | 3 | 100% | 1859.08 / 1901.65 ms | 8.58 ms | 1867.66 / 1913.66 ms |
| Qwen | `Qwen/Qwen3.6-35B-A3B-FP8` | 3 | 100% | 750.39 / 809.08 ms | 6.39 ms | 758.88 / 815.68 ms |

## Reproduce the comparison

Use an external dotenv file and external result directory. Never put credentials or generated reports in the repository.

PowerShell (Windows):

```powershell
$env:LATENCYOPS_ENV_FILE = "C:\path\to\private\latencyops\.env"
$env:LATENCYOPS_RESULTS_DIR = "C:\path\to\private\latencyops\results"
$env:PYTHONPATH = "src"
python examples\compare_real_endpoints.py
```

Ubuntu (bash):

```bash
export LATENCYOPS_ENV_FILE="/path/to/private/latencyops/.env"
export LATENCYOPS_RESULTS_DIR="/path/to/private/latencyops/results"
export PYTHONPATH=src
python3 examples/compare_real_endpoints.py
```

The script uses the configured model list and writes the aggregate report to the external results directory. Review the authenticated `/v1/models` response before selecting model IDs.

## Interpretation

The measurements show behavior for one client, one region, one workload, one deployment window, and three runs per model. They do not establish general model quality, an SLA, a price comparison, or a provider guarantee. Repeat the benchmark with the target workload, region, concurrency, warm-up policy, and quality evaluator before using results for an operational decision.

The Qwen deployment did not expose direct vLLM queue or KV-cache metrics through the public proxy. Any planning signals used in that test must therefore be treated as synthetic unless a direct metrics endpoint is configured.
