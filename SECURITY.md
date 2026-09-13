# Security Policy

## Scope

LatencyOps is an experimental, dependency-light reference implementation, not a hardened production service. Do not use the local gateway directly on an untrusted network.

## Reporting

Do not include API keys, customer prompts, response content, or other sensitive data in issue reports. Report security concerns privately to the project maintainers before public disclosure.

## Safe operation

- Keep credentials in environment variables or a secret manager.
- Do not commit `.env` files.
- Use synthetic or approved evaluation data.
- Put a hardened TLS/authenticated proxy in front of the reference gateway.
- Review generated benchmark files before sharing them.
