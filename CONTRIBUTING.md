# Contributing

LatencyOps is provider-neutral and dependency-light. Contributions should preserve the standalone core and use adapters for provider-specific behavior.

## Development checks

Use the project Python interpreter and run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
```

Every behavior change must include an automated regression test and an entry in `MASTER_TEST_CASES.md`. Do not add credentials, customer data, proprietary prompts, or production configuration.

## Release verification

Build and validate the distributions before a public release:

```powershell
python -m pip install build==1.2.2 twine==6.1.0
python -m build --sdist --wheel --outdir dist
python -m twine check dist\*
$env:LATENCYOPS_RELEASE_ARTIFACTS = "1"
python -m unittest discover -s tests -v
python -m compileall -q src tests examples
```

The release-gated tests install the wheel in a temporary isolated environment and run the installed CLI without `PYTHONPATH`. Real OpenAI and LiteLLM tests are opt-in, require approved credentials, and must not run in public CI. Use exact model IDs available to the configured credential; do not commit credentials or real endpoint configuration.
