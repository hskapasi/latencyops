# LatencyOps Master Test Cases

| ID | Area | Scenario | Expected result |
|---|---|---|---|
| LOP-001 | Contracts | Reject a quality floor outside 0 to 1 | `ValueError` is raised |
| LOP-002 | Contracts | Reject pressure outside 0 to 1 | `ValueError` is raised |
| LOP-003 | Metrics | Calculate an interpolated percentile | Correct percentile is returned |
| LOP-004 | Metrics | Summarize request timings | Count, success rate, median, and tail latency are returned |
| LOP-005 | Planning | Select context reduction for long prompts | Selective or compressed context policy is returned |
| LOP-006 | Planning | Protect a critical request | Large model, FP16, and no early exit are selected |
| LOP-007 | Planning | Optimize an easy low-risk request | Small model and early exit are selected |
| LOP-008 | Planning | Use high-acceptance speculation | Positive speculative token count is selected |
| LOP-009 | Routing | Route a plan to its model tier | The selected provider handles the request |
| LOP-010 | Benchmarking | Continue after an individual provider failure | Successful and failed samples are both recorded |
| LOP-011 | Streaming | Measure time to first streamed chunk | TTFT is recorded independently from end-to-end latency |
| LOP-012 | Streaming | Measure time per generated token after first chunk | TPOT is calculated from post-first-token streaming time |
| LOP-013 | Routing | Forward a stream to the planned tier | The selected streaming provider handles the request |
| LOP-014 | Operations | Normalize cache usage into pressure | A bounded pressure ratio is exposed |
| LOP-015 | Scheduling | Select urgent work before normal work | Highest-priority queued request is selected |
| LOP-016 | Telemetry | Export content-free request measurements | Exported metrics contain measurements but no prompt content |
| LOP-017 | Benchmarking | Compare providers by tail latency | Provider summaries are ranked by end-to-end p95 |
| LOP-018 | Gateway | Process a completion through planning and routing | OpenAI-shaped result includes an explainable plan and telemetry is recorded |
| LOP-019 | Gateway | Stream a completion through the local service | Provider chunks are yielded without retaining prompt content |
| LOP-020 | Gateway | POST a completion to the local HTTP server | An OpenAI-shaped JSON response is returned |
| LOP-021 | Adapters | Map an abstract tier to a provider model and parse SSE | The mapped model is sent and streamed text is normalized |
| LOP-022 | Adapters | Send chat messages and provider-specific template options | Chat-compatible providers receive the expected model and message schema |
| LOP-023 | Real integration | Discover models from an authenticated LiteLLM endpoint | Only authorized model IDs are displayed and no credential is printed |
| LOP-024 | Real integration | Run a Qwen chat stream through the authenticated LiteLLM endpoint | A real response is returned with TTFT and TPOT measurements |
| LOP-025 | Real integration | Run an OpenAI chat stream with the separate OpenAI credential | A real OpenAI response is returned with TTFT and TPOT measurements |
| LOP-026 | Real integration | Compare OpenAI and Qwen on the same synthetic stream workload | Aggregate provider latency summaries are saved without prompt or credential content |
| LOP-027 | Planning | Preserve full safeguards for critical long-context work | Critical requests use large FP16 full-context execution without early exit |
| LOP-028 | Planning | Select a healthy provider predicted to meet the deadline | An unhealthy or predicted-late provider is not selected |
| LOP-029 | Planning | Re-plan after a quality-floor violation | A safer plan is produced with an explanation |
| LOP-030 | Planning | Compare selected plan with a static baseline | Shadow output contains both plans without request content |
| LOP-031 | Benchmarking | Run a workload matrix against a provider | Each case reports measurements and deadline satisfaction |
| LOP-032 | Benchmarking | Run warm-up and concurrent scenario measurements | Warm-ups are excluded and quality, cost, cache, and network metadata are reported |
| LOP-033 | Planning | Preserve critical model tier under queue pressure | Critical work remains on the large tier despite load signals |
| LOP-034 | Real integration | Execute proactive plans against OpenAI and Qwen endpoints | Real requests execute selected plans and report content-free latency/constraint results |
| LOP-035 | Real integration | Compare multiple OpenAI models with Qwen | Model-specific aggregate latency results are saved without credentials or content |
| LOP-036 | Planning | Disable speculation for critical requests | Critical work never enables speculative decoding |
| LOP-037 | Real integration | Run public-domain workload categories against OpenAI | Eight category checks report quality signals, plans, latency, and deadline status |
| LOP-038 | Real integration | Validate all eight public workload categories | Factual, extraction, JSON, summary, code, classification, reasoning, and safety checks are reported without content retention |
| LOP-039 | Enforcement | Apply provider capability constraints to a plan | Unsupported optimizations are removed and reported |
| LOP-040 | Signals | Normalize model-server queue and cache metrics | Provider-specific metrics become bounded system signals |
| LOP-041 | Experiments | Compare proactive and static plans | Both plans report latency, quality, and deadline outcomes |
| LOP-042 | Selection | Adapt provider choice from observed quality and latency | A quality-eligible provider is preferred over an unsafe faster provider |
| LOP-043 | Orchestrators | Normalize vLLM documented metrics | vLLM queue and KV-cache metrics become system signals |
| LOP-044 | Orchestrators | Normalize SGLang documented metrics | SGLang queue and token-usage metrics become system signals |
| LOP-045 | Orchestrators | Identify confirmed model-server metric families | Provider profiles use documented metric names rather than assumptions |
| LOP-046 | Release | Expose requested and enforced plans | Gateway responses show unsupported provider features explicitly |
| LOP-047 | Routing | Use adaptive provider candidates in the gateway | The gateway routes to the quality-eligible adaptive provider |
| LOP-048 | Orchestrators | Parse label-aware TensorRT-LLM fixture metrics | Waiting requests and KV-cache fraction are normalized without mixing labels |
| LOP-049 | Packaging | Load provider tiers from TOML and environment | Configuration contains model mappings without storing credential values |
| LOP-050 | Packaging | Start the installable gateway command | Configured service can be run through the package entry point |
| LOP-051 | Packaging | Generate a plan through the CLI without a provider | `latencyops plan` emits a valid proactive plan using only local input |
| LOP-052 | Release | Validate public package metadata and release hygiene files | Required metadata, license, security, contribution, and configuration files are present and safe |
| LOP-053 | Release | Inspect built wheel and source archive contents | Runtime package, CLI entry point, and license are included without secret or generated-result files |
| LOP-054 | Release | Install the wheel in an isolated environment and run the CLI | The installed package imports outside the source tree and `latencyops plan` succeeds |
| LOP-055 | Release | Normalize provider failures without request or credential leakage | HTTP, malformed-stream, and health failures return safe normalized results |
| LOP-056 | Release | Run packaging and artifact gates in CI | CI builds distributions, validates metadata, runs release-gated tests, and compiles the project |
| LOP-057 | Documentation | Publish verified model coverage and install paths | README separates verified models from configurable candidates and explains GitHub, wheel, and PyPI usage |
| LOP-058 | Documentation | Explain the LatencyOps architecture | README shows library and gateway paths, external providers, optional runtime metrics, and measured outputs |
| LOP-059 | Documentation | Explain multi-agent and cross-platform usage | README highlights single-system and multi-agent value and provides equivalent PowerShell and Ubuntu commands |
| LOP-060 | Documentation | Separate public README from detailed architecture reference | Internal architecture notes are excluded from the public README and public repository surface |
| LOP-061 | Documentation | Show public LatencyOps placement | README shows the application/agent workflow, LatencyOps policy layer, provider endpoints, and optional runtime metrics |
| LOP-062 | Examples | Keep credentials and generated results outside the repository | Real-endpoint examples load an external dotenv file and write result JSON to an external directory |
| LOP-063 | Documentation | Publish sanitized sample model results | README includes aggregate OpenAI/Qwen latency observations without credentials, prompts, or raw responses |
| LOP-064 | Packaging | Publish project contact and security URLs | PyPI metadata contains GitHub homepage, repository, issues, and security links for users |
| LOP-065 | Documentation | Publish discoverability assets | README, issue templates, sample request, terminal demo, and benchmark report are present and sanitized |
| LOP-066 | Packaging | Publish PyPI discovery metadata | Versioned package metadata contains searchable keywords and Python/AI package classifiers |
