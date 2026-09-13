# LatencyOps Project Guidance

## Scope

LatencyOps is a separate inference-latency measurement and quality-aware planning toolkit. It must remain separate from the compliance-document-auditor and arxiv-agent repositories.

## Design

- Keep the community core provider-neutral and dependency-light.
- Do not include proprietary prompts, scoring formulas, customer data, credentials, or production configuration.
- Treat latency as TTFT, TPOT, queue, cache, and end-to-end measurements rather than one aggregate number.
- High-risk workloads must preserve explicit quality safeguards.

## Testing

- The master test case file is `MASTER_TEST_CASES.md`.
- Every behavior change requires a corresponding automated regression test and test-case entry.
- Run `python -m unittest discover -s tests -v` and `python -m py_compile src\latencyops\*.py tests\test_latencyops.py`.

## Publishing

- Do not create a GitLab repository, push commits, deploy infrastructure, or publish artifacts without explicit user approval.
- Review the public/community versus commercial boundary before any repository is made public.
