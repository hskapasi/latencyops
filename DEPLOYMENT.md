# Deployment Reference

This repository provides a production-style package layout and a reference gateway. It is not a hardened production service by itself.

## Install as a package

```powershell
& "C:\Users\hskap\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe" -m pip install .
latencyops plan --request request.json
```

## Configure the gateway

Copy `latencyops.example.toml` to a private configuration location and set the environment variables named by `api_key_env`. Do not place keys in TOML.

```powershell
latencyops gateway --config latencyops.toml
```

The gateway listens on the configured address and exposes the OpenAI-shaped completion endpoint. Put Nginx, Envoy, or a managed load balancer in front of it for TLS, authentication, rate limiting, request limits, and production traffic management.

## Container reference

```powershell
docker build -t latencyops:local .
docker run --rm -p 8080:8080 --env-file .env latencyops:local
```

The image does not contain model weights or inference runtimes. Providers must be reachable through the configured OpenAI-compatible endpoints.

## Production requirements

Before production use, add an authenticated edge proxy, secret manager, durable queue or external scheduler, cancellation, circuit breaking, resource limits, provider-specific metrics access, and a quality evaluation process. The reference gateway should not be exposed directly to an untrusted network.
