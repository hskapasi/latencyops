# LatencyOps Compared with Existing LLM Infrastructure

## Executive summary

LatencyOps should not compete by rebuilding every part of the LLM infrastructure stack. Mature open-source and commercial projects already provide model serving, HTTP gateways, provider integrations, Kubernetes networking, tracing, and evaluation.

The recommended strategy is to use those projects as the execution substrate and provide a thin, provider-neutral **latency-and-quality policy layer** above them.

```text
Application
    |
    v
Nginx / Envoy / Gateway API
    |
    v
LatencyOps policy layer
    |
    +-- request contract and risk policy
    +-- latency-aware plan selection
    +-- queue/cache/provider signals
    +-- explainable routing decision
    +-- TTFT/TPOT/SLO measurement
    |
    v
Existing gateway or inference router
    |
    v
vLLM / SGLang / TensorRT-LLM / hosted provider
```

The current repository contains a dependency-light reference implementation of this policy layer and a local gateway for development. It is not intended to replace mature production proxies or model servers.

## Existing tools by function

### Model-serving engines

These systems load models and execute inference:

- **vLLM**: broad model support, OpenAI-compatible serving, continuous batching, and KV-cache management.
- **SGLang**: serving and runtime optimizations, especially useful for structured generation and shared-prefix workloads.
- **TensorRT-LLM**: NVIDIA-oriented compilation and runtime optimization for maximum performance on supported hardware.
- **NVIDIA Triton**: general model-serving platform with production deployment integrations.
- **Hugging Face TGI**: text-generation serving with streaming and production-oriented features.

LatencyOps should integrate with these systems instead of implementing model execution.

### General LLM gateways

These systems provide a unified API across providers and common gateway controls:

- **LiteLLM**: open-source gateway, provider adapters, routing, fallbacks, budgets, rate limits, and telemetry integrations.
- **Portkey**: commercial gateway and governance platform with routing, fallbacks, guardrails, and enterprise controls.
- **Helicone**: gateway and observability platform with provider routing, cost tracking, latency tracking, and dashboards.
- **OpenRouter**: hosted multi-model and multi-provider access layer.

LatencyOps should integrate with these systems where they already solve provider translation, authentication, rate limiting, and basic fallback behavior.

### Inference-aware routing and scheduling

- **llm-d Router** and the **Gateway API Inference Extension** are the closest architectural comparison for self-hosted inference operations.
- They combine Envoy or another Gateway API implementation with an endpoint picker that can use request priority, queue state, KV-cache locality, and model-server metrics.
- They are especially relevant for Kubernetes deployments with multiple model-server replicas.

LatencyOps should be able to act as a policy source or decision component alongside this ecosystem rather than duplicating its proxy and cluster-management functions.

### Observability and evaluation

- **Arize Phoenix**: open-source tracing and evaluation with OpenTelemetry/OpenInference alignment.
- **Langfuse**: self-hostable tracing, cost, token, prompt/version, and evaluation capabilities.
- **LangSmith**: tracing and evaluation for LangChain/LangGraph-centered applications.
- **Braintrust**: datasets, scorers, experiments, and CI quality gates.
- **Prometheus/OpenTelemetry**: standard infrastructure for metrics and traces.

LatencyOps should emit standard telemetry and integrate with these systems. It should not retain prompt or response content by default in the community core.

## Capability comparison

| Capability | LatencyOps | LiteLLM | llm-d | Portkey | Helicone | vLLM |
|---|---|---|---|---|---|---|
| OpenAI-compatible gateway | Basic reference | Strong | Via Gateway API/Envoy | Strong | Strong | Yes |
| Multi-provider translation | Basic adapters | Strong | Serving-focused | Strong | Strong | No |
| Model execution | No | No | No | No | No | Yes |
| Queue-aware routing | Primitive foundation | Limited/configurable | Strong | Varies | Varies | Internal scheduler |
| KV-cache-aware routing | Signal contract | Limited | Strong | Varies | Varies | Internal cache |
| TTFT/TPOT measurement | Core focus | Telemetry | Inference-routing focus | Yes | Yes | Server metrics |
| Explainable execution plan | Core focus | Configuration/routing | Scheduler decision | Policy-focused | Routing-focused | No |
| Risk-aware quality safeguards | Core focus | Possible extension | Not central | Governance-related | Not central | No |
| Adaptive quality/latency controller | Planned | Limited | Routing/scheduling focus | Policy focus | Routing/observability focus | No |
| Hosted dashboard | No | Product options | No central dashboard | Yes | Yes | No |
| Self-hosted core | Yes | Yes | Yes | Product-dependent | Product-dependent | Yes |

The table is directional, not a claim that every product has identical editions or deployment modes. Exact capabilities must be verified against the selected version and license before adoption.

## What LatencyOps should reuse

LatencyOps should preferentially reuse the following open-source capabilities:

1. **HTTP data plane**: Nginx, Envoy, or Gateway API for TLS, connection handling, limits, and proxying.
2. **Inference routing substrate**: llm-d/Gateway API Inference Extension for Kubernetes endpoint picking, queue-aware routing, and cache-aware placement.
3. **Model servers**: vLLM, SGLang, TensorRT-LLM, or Triton for actual model execution.
4. **Provider gateway**: LiteLLM when multiple hosted or self-hosted providers need one API and common fallbacks.
5. **Telemetry**: Prometheus and OpenTelemetry, optionally consumed by Phoenix or Langfuse.
6. **Quality evaluation**: Braintrust, Phoenix, Langfuse, LangSmith, or an organization-approved evaluator.

This avoids maintaining duplicate implementations of mature infrastructure and lets LatencyOps focus on the missing cross-system decision problem.

## The thin layer LatencyOps should provide

The differentiated layer should combine signals and requirements that are often exposed separately by other tools:

```text
Request contract
+ workload difficulty
+ risk class
+ queue pressure
+ cache pressure
+ provider health
+ observed TTFT/TPOT
+ quality outcome
+ cost constraints
        |
        v
Explainable execution plan
```

A plan may select or constrain:

- model tier or provider,
- precision,
- context policy,
- speculative decoding,
- early exit,
- queue priority,
- fallback eligibility,
- quality safeguards.

The plan must be explainable and should be reversible when observed latency or quality moves outside the contract.

## What is genuinely different

Existing products commonly optimize one or more of these boundaries:

- provider selection,
- endpoint placement,
- queue scheduling,
- cache locality,
- numerical representation,
- speculative decoding,
- early exit,
- tracing,
- quality evaluation.

LatencyOps is intended to coordinate transitions between those boundaries at the request level:

```text
If queue pressure rises:
    move eligible low-risk work to a faster tier.

If cache pressure rises:
    use a context or cache-reduction policy.

If draft acceptance falls:
    reduce or disable speculation.

If quality risk rises:
    restore model capacity, precision, or full context.

If the request is critical:
    preserve explicit safeguards regardless of load.
```

This is a differentiation hypothesis, not a patentability conclusion. Any patent or prior-art conclusion requires a separate legal and technical review.

## Recommended integration modes

### Local development

```text
Client
  -> LatencyOps local gateway
  -> mock provider or local OpenAI-compatible server
```

Use the existing Python gateway, deterministic providers, and benchmark tests.

### Single-node real-model testing

```text
Client
  -> LatencyOps policy layer
  -> vLLM or SGLang
  -> local model
```

Use identical prompts, concurrency, model weights, precision, and warm-up conditions when comparing policies.

### Multi-provider testing

```text
Client
  -> LatencyOps plan
  -> LiteLLM or provider adapter
  -> OpenAI / Azure / Bedrock / local provider
```

LatencyOps chooses an abstract tier or policy. The gateway translates that decision to provider-specific model names and credentials.

### Kubernetes production architecture

```text
Client
  -> Nginx/Envoy/Gateway API
  -> llm-d or inference endpoint picker
  -> LatencyOps policy decision component
  -> vLLM/SGLang/TensorRT-LLM replicas
  -> Prometheus/OpenTelemetry
```

The division of responsibility should be explicit:

- Edge proxy: network security and HTTP operations.
- Inference router: endpoint placement and cluster-level scheduling.
- LatencyOps: request-level latency/quality policy and explainable tradeoffs.
- Model server: model execution and serving metrics.
- Evaluation system: quality scoring and regression gates.

## Benchmark plan

LatencyOps should be compared against existing tools using the same model server and workload.

### Baseline 1: direct model server

Compare vLLM, SGLang, and TensorRT-LLM where hardware permits:

- TTFT p50/p95/p99
- TPOT p50/p95/p99
- end-to-end latency
- throughput
- queue wait
- KV-cache utilization
- error rate

### Baseline 2: gateway overhead

Compare:

```text
Client -> direct provider
Client -> LiteLLM -> provider
Client -> LatencyOps -> provider
Client -> Nginx/Envoy -> LatencyOps -> provider
```

Measure added latency, streaming behavior, throughput, and failure handling.

### Baseline 3: inference-aware routing

Compare LatencyOps with llm-d for:

- one versus multiple replicas,
- cold versus warm prefix cache,
- low versus high concurrency,
- mixed short and long prompts,
- queue saturation,
- provider failure,
- model rollout,
- critical versus low-risk requests.

### Baseline 4: quality-latency frontier

Run the same approved evaluation set through each candidate policy and record:

- quality score,
- TTFT,
- TPOT,
- end-to-end latency,
- p95 latency,
- token usage,
- cost,
- error rate.

A policy is successful only when its quality and latency constraints are both satisfied.

## Community versus commercial boundary

### Suitable for the public community repository

- Contracts and data models
- Provider-neutral routing interfaces
- Streaming and latency measurement
- Static explainable policies
- Queue and cache signal interfaces
- Open-source provider adapters
- Prometheus/OpenTelemetry integrations
- Benchmark harnesses
- Mock providers
- Synthetic examples
- Integrations with public open-source systems

### Keep private initially

- Adaptive policy-learning engine
- Proprietary latency predictors
- Proprietary quality-risk scoring formulas
- Customer prompts and evaluation data
- Hosted dashboard and managed control plane
- Production rollout automation
- Customer-specific integrations
- Internal research-ranking logic

The proposed licensing model remains:

```text
Community core: Apache-2.0
Commercial control plane: proprietary
Enterprise integrations and support: paid
```

Before publication, review the complete repository and Git history for secrets, customer data, production configuration, incompatible licenses, and accidental proprietary material.

## Conclusion

LatencyOps should be positioned as a **thin inference policy and measurement layer**, not as a replacement for Nginx, LiteLLM, llm-d, vLLM, or evaluation platforms.

The strongest product architecture is compositional:

```text
Use existing systems for execution.
Use LatencyOps to decide how each request should be executed
under latency, quality, risk, cache, queue, and cost constraints.
```
