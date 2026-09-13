import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
import unittest
import venv
import zipfile
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

from real_endpoint_test import load_dotenv, result_path
from latencyops.adapters import CallableProvider, OpenAIChatCompatibleProvider, OpenAICompatibleProvider, ProviderError
from latencyops.capabilities import ProviderCapabilities, enforce_plan
from latencyops.benchmark import BenchmarkRunner
from latencyops.benchmark_suite import BenchmarkConfig, BenchmarkScenario, ScenarioBenchmarkRunner
from latencyops.comparison import WorkloadCase, compare, run_workload_matrix
from latencyops.config import load_config
from latencyops.cli import main as cli_main
from latencyops.controller import ProactiveController
from latencyops.experiments import PolicyExperiment
from latencyops.signals import RuntimeSignalNormalizer
from latencyops.orchestrators import (
    PROFILES,
    PrometheusMetricsCollector,
    SGLangMetricsCollector,
    TensorRTLLMMetricsCollector,
    VLLMMetricsCollector,
)
from latencyops.gateway import GatewayService, create_server
from latencyops.metrics import percentile, summarize
from latencyops.models import (
    InferencePlan,
    LatencyContract,
    LatencySample,
    ProviderCandidate,
    QualityOutcome,
    RequestProfile,
    StreamChunk,
)
from latencyops.policy import AdaptiveProviderSelector, ElasticInferencePlanner
from latencyops.router import ModelRouter, StaticProvider
from latencyops.scheduling import CachePressureSignal, PriorityScheduler
from latencyops.telemetry import PrometheusExporter, TelemetryRecord, TelemetryRecorder


class ContractTests(unittest.TestCase):
    def test_contract_rejects_invalid_quality_floor(self):
        with self.assertRaises(ValueError):
            LatencyContract(deadline_ms=500, quality_floor=1.2)

    def test_profile_rejects_invalid_pressure(self):
        with self.assertRaises(ValueError):
            RequestProfile(prompt_tokens=10, expected_output_tokens=10, queue_pressure=2)


class MetricsTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([10, 20, 30, 40], 50), 25)

    def test_summary_reports_tail_latency(self):
        summary = summarize([
            LatencySample(10, 2, 30),
            LatencySample(20, 3, 60),
            LatencySample(40, 5, 100),
        ])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["e2e_p95_ms"], 96)
        self.assertEqual(summary["success_rate"], 1.0)


class PackagingTests(unittest.TestCase):
    def test_toml_config_loads_provider_model_tiers_from_env(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                """[server]\nport = 9000\n\n[providers.test]\nendpoint = \"http://test/v1/chat/completions\"\napi_key_env = \"TEST_LATENCYOPS_KEY\"\n[providers.test.models]\nsmall = \"small-model\"\nstandard = \"standard-model\"\nlarge = \"large-model\"\n"""
            )
            os.environ["TEST_LATENCYOPS_KEY"] = "synthetic-key"
            try:
                config = load_config(path)
            finally:
                del os.environ["TEST_LATENCYOPS_KEY"]
            self.assertEqual(config.port, 9000)
            self.assertEqual(config.providers["test"].model_names["large"], "large-model")

    def test_cli_plan_emits_json_without_external_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "request.json"
            request.write_text('{"prompt_tokens": 10, "expected_output_tokens": 5, "difficulty": 0.1}')
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli_main(["plan", "--request", str(request)]), 0)
            self.assertEqual(json.loads(output.getvalue())["model_tier"], "small")


class PublicReleaseGateTests(unittest.TestCase):
    def test_public_release_files_are_present(self):
        required = {
            "LICENSE",
            "README.md",
            "SECURITY.md",
            "CONTRIBUTING.md",
            ".env.example",
            "MASTER_TEST_CASES.md",
            "DEPLOYMENT.md",
            "Dockerfile",
            "pyproject.toml",
            ".github/workflows/ci.yml",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
            ".github/ISSUE_TEMPLATE/config.yml",
            "examples/request.json",
            "docs/BENCHMARK_REPORT.md",
            "docs/terminal-demo.svg",
        }
        missing = sorted(name for name in required if not (PROJECT_ROOT / name).is_file())
        self.assertEqual(missing, [])

    def test_pyproject_declares_installable_package_and_cli(self):
        metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = metadata["project"]
        self.assertEqual(project["name"], "latencyops")
        self.assertEqual(project["version"], "0.1.2")
        self.assertEqual(project["requires-python"], ">=3.12")
        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(project.get("dependencies", []), [])
        self.assertEqual(project["scripts"]["latencyops"], "latencyops.cli:main")
        self.assertEqual(project["urls"], {
            "Homepage": "https://github.com/hskapasi/latencyops",
            "Repository": "https://github.com/hskapasi/latencyops",
            "Issues": "https://github.com/hskapasi/latencyops/issues",
            "Security": "https://github.com/hskapasi/latencyops/blob/main/SECURITY.md",
        })
        self.assertIn("inference-latency", project["keywords"])
        self.assertIn("vllm", project["keywords"])
        self.assertIn("Development Status :: 3 - Alpha", project["classifiers"])
        self.assertIn("Topic :: Scientific/Engineering :: Artificial Intelligence", project["classifiers"])
        self.assertEqual(metadata["build-system"]["build-backend"], "setuptools.build_meta")

    def test_env_example_contains_only_placeholder_credentials(self):
        contents = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertRegex(contents, r"(?m)^OPENAI_API_KEY=\s*$")
        self.assertRegex(contents, r"(?m)^QWEN_API_KEY=\s*$")
        self.assertNotRegex(contents, r"sk-[A-Za-z0-9]{20,}")
        self.assertNotRegex(contents, r"(?i)bearer\s+\S+")

    def test_ignore_rules_protect_credentials_and_build_artifacts(self):
        contents = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        for pattern in (".env", ".env.*", "!.env.example", "build/", "dist/", "*.egg-info/"):
            self.assertIn(pattern, contents)

    def test_public_docs_state_security_and_alpha_boundaries(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
        security = (PROJECT_ROOT / "SECURITY.md").read_text(encoding="utf-8").lower()
        self.assertIn("not a hardened production serving platform", readme)
        for phrase in (
            "## why use latencyops?",
            "## install and run",
            "## verified real model coverage",
            "## configured comparison candidates",
            "a single planner can serve one application",
            "multi-agent workflow",
            "## where latencyops sits",
            "your application / agent workflow",
            "latencyops public policy + measurement layer",
            "optional runtime metrics from vllm",
            "powershell (windows):",
            "ubuntu (bash):",
            "python3 -m pip install .",
            "gpt-4o-mini",
            "qwen/qwen3.6-35b-a3b-fp8",
            "sample streaming comparison",
            "ttft p50 / p95",
            "gpt-5-mini",
            "latencyops_env_file",
            "latencyops_results_dir",
            "## terminal demo",
            "examples/request.json",
            "docs/benchmark_report.md",
            "## support and contact",
            "github issues",
            "not yet been published to pypi",
        ):
            self.assertIn(phrase, readme)
        self.assertNotIn("## architecture\n", readme)
        self.assertNotIn("agent_contracts", readme)
        self.assertIn("not a hardened production service", security)
        self.assertIn("untrusted network", security)
        self.assertIn("authenticated", security)

    def test_internal_architecture_reference_is_excluded_from_public_surface(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
        ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertNotIn("ARCHITECTURE.md", ignore_rules)
        self.assertNotIn("LATENCYOPS_CONVERSATION_HANDOFF.md", ignore_rules)
        self.assertNotIn("REAL_*.json", ignore_rules)
        self.assertNotIn("architecture.md", readme)

    def test_real_endpoint_examples_use_external_env_and_results_paths(self):
        env_file_name = "LATENCYOPS_TEST_EXTERNAL_VALUE"
        previous_env_file = os.environ.get("LATENCYOPS_ENV_FILE")
        previous_results_dir = os.environ.get("LATENCYOPS_RESULTS_DIR")
        previous_value = os.environ.pop(env_file_name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env_file = root / ".env"
                results_dir = root / "results"
                env_file.write_text(f"{env_file_name}=configured\n", encoding="utf-8")
                os.environ["LATENCYOPS_ENV_FILE"] = str(env_file)
                os.environ["LATENCYOPS_RESULTS_DIR"] = str(results_dir)
                load_dotenv()
                output = result_path("REAL_TEST.json")
                self.assertEqual(os.environ[env_file_name], "configured")
                self.assertEqual(output.parent, results_dir)
                self.assertTrue(results_dir.is_dir())
        finally:
            if previous_env_file is None:
                os.environ.pop("LATENCYOPS_ENV_FILE", None)
            else:
                os.environ["LATENCYOPS_ENV_FILE"] = previous_env_file
            if previous_results_dir is None:
                os.environ.pop("LATENCYOPS_RESULTS_DIR", None)
            else:
                os.environ["LATENCYOPS_RESULTS_DIR"] = previous_results_dir
            if previous_value is None:
                os.environ.pop(env_file_name, None)
            else:
                os.environ[env_file_name] = previous_value

    def test_discoverability_assets_are_valid(self):
        request = json.loads((PROJECT_ROOT / "examples" / "request.json").read_text(encoding="utf-8"))
        for field in ("prompt_tokens", "expected_output_tokens", "difficulty", "quality_floor", "risk_class", "deadline_ms"):
            self.assertIn(field, request)
        benchmark = (PROJECT_ROOT / "docs" / "BENCHMARK_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("## Aggregate results", benchmark)
        self.assertIn("gpt-4.1-mini", benchmark)
        self.assertIn("Qwen/Qwen3.6-35B-A3B-FP8", benchmark)
        self.assertIn("Prompt and response retention: none", benchmark)
        self.assertIn("name: Bug report", (PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.md").read_text(encoding="utf-8"))
        self.assertIn("name: Feature request", (PROJECT_ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.md").read_text(encoding="utf-8"))
        self.assertIn("LatencyOps planning CLI demo", (PROJECT_ROOT / "docs" / "terminal-demo.svg").read_text(encoding="utf-8"))

    def test_public_text_files_do_not_contain_credential_values(self):
        allowed_suffixes = {".md", ".py", ".toml", ".txt", ".json", ".yml", ".yaml", ".example"}
        excluded_directories = {".git", ".venv", "__pycache__", "build", "dist", "latencyops.egg-info"}
        for path in PROJECT_ROOT.rglob("*"):
            if not path.is_file() or path.name == ".env" or path.name.startswith("REAL_"):
                continue
            if any(part in excluded_directories for part in path.relative_to(PROJECT_ROOT).parts[:-1]):
                continue
            if path.suffix not in allowed_suffixes and path.name not in {"Dockerfile", ".gitignore"}:
                continue
            contents = path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotRegex(contents, r"sk-[A-Za-z0-9]{20,}", str(path))
            self.assertNotRegex(contents, r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}", str(path))
            self.assertNotRegex(contents, r"(?m)^[ \t]*(OPENAI_API_KEY|QWEN_API_KEY)[ \t]*=[ \t]*\S+", str(path))

    def test_ci_defines_build_metadata_and_release_artifact_gates(self):
        workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        for command in (
            "python -m build --sdist --wheel --outdir dist",
            "python -m twine check",
            "python -m unittest discover -s tests -v",
            "python -m compileall -q src tests examples",
        ):
            self.assertIn(command, workflow)
        self.assertIn("LATENCYOPS_RELEASE_ARTIFACTS", workflow)

    def test_master_test_case_ids_are_unique_ordered_and_include_release_gates(self):
        contents = (PROJECT_ROOT / "MASTER_TEST_CASES.md").read_text(encoding="utf-8")
        ids = [
            line.split("|", 2)[1].strip()
            for line in contents.splitlines()
            if line.startswith("| LOP-")
        ]
        numbers = [int(case_id.split("-")[1]) for case_id in ids]
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))
        for case_id in ("LOP-052", "LOP-053", "LOP-054", "LOP-055", "LOP-056", "LOP-057", "LOP-058", "LOP-059", "LOP-060", "LOP-061", "LOP-062", "LOP-063", "LOP-064", "LOP-065", "LOP-066"):
            self.assertIn(case_id, ids)

    def test_openai_model_tiers_map_to_exact_provider_model_ids(self):
        provider = OpenAIChatCompatibleProvider(
            "openai",
            "https://api.openai.com/v1/chat/completions",
            model_names={
                "small": "gpt-4o-mini",
                "standard": "gpt-4.1-mini",
                "large": "gpt-5-mini",
            },
        )
        self.assertEqual(
            provider._model_name(InferencePlan("standard", "fp16", "full", 0, False)),
            "gpt-4.1-mini",
        )

    def test_config_rejects_missing_credential_without_leaking_values(self):
        key_name = "LATENCYOPS_RELEASE_MISSING_KEY"
        previous = os.environ.pop(key_name, None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(
                    f"""[providers.test]\nendpoint = \"https://example.invalid/v1/chat/completions\"\napi_key_env = \"{key_name}\"\n[providers.test.models]\nsmall = \"small\"\nstandard = \"standard\"\nlarge = \"large\"\n""",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, key_name) as raised:
                    load_config(path)
                self.assertNotIn("secret", str(raised.exception).lower())
        finally:
            if previous is not None:
                os.environ[key_name] = previous

    def test_config_rejects_missing_model_tier(self):
        key_name = "LATENCYOPS_RELEASE_TIER_KEY"
        previous = os.environ.get(key_name)
        os.environ[key_name] = "synthetic-key"
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.toml"
                path.write_text(
                    f"""[providers.test]\nendpoint = \"https://example.invalid/v1/chat/completions\"\napi_key_env = \"{key_name}\"\n[providers.test.models]\nsmall = \"small\"\nstandard = \"standard\"\n""",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "large"):
                    load_config(path)
        finally:
            if previous is None:
                os.environ.pop(key_name, None)
            else:
                os.environ[key_name] = previous

    def test_provider_failure_is_normalized_without_prompt_or_credential(self):
        secret = "synthetic-release-secret"
        prompt = "private release prompt"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"upstream failure details")

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OpenAICompatibleProvider(
                "release-provider",
                f"http://127.0.0.1:{server.server_port}/v1/completions",
                api_key=secret,
                timeout=1,
            )
            with self.assertRaises(ProviderError) as raised:
                provider.generate(prompt, InferencePlan("standard", "fp16", "full", 0, False))
            self.assertNotIn(secret, str(raised.exception))
            self.assertNotIn(prompt, str(raised.exception))
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()

    def test_malformed_stream_is_normalized_without_prompt_content(self):
        prompt = "private streaming prompt"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: {not-json}\n\n")

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OpenAICompatibleProvider(
                "release-provider",
                f"http://127.0.0.1:{server.server_port}/v1/completions",
                timeout=1,
            )
            with self.assertRaises(ProviderError) as raised:
                list(provider.stream(prompt, InferencePlan("standard", "fp16", "full", 0, False)))
            self.assertNotIn(prompt, str(raised.exception))
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()

    def test_failed_health_probe_is_not_reported_healthy(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(503)
                self.end_headers()

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OpenAICompatibleProvider(
                "release-provider",
                "https://example.invalid/v1/completions",
                health_endpoint=f"http://127.0.0.1:{server.server_port}/health",
                timeout=1,
            )
            self.assertFalse(provider.health().healthy)
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()

    @unittest.skipUnless(
        os.environ.get("LATENCYOPS_RELEASE_ARTIFACTS") == "1",
        "set LATENCYOPS_RELEASE_ARTIFACTS=1 after building release artifacts",
    )
    def test_built_artifacts_contain_runtime_package_and_no_secret_files(self):
        wheels = sorted((PROJECT_ROOT / "dist").glob("latencyops-*.whl"))
        sdists = sorted((PROJECT_ROOT / "dist").glob("latencyops-*.tar.gz"))
        self.assertTrue(wheels)
        self.assertTrue(sdists)

        with zipfile.ZipFile(wheels[-1]) as archive:
            names = archive.namelist()
            self.assertIn("latencyops/__init__.py", names)
            self.assertIn("latencyops/cli.py", names)
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            entry_points_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
            metadata = archive.read(metadata_name).decode("utf-8")
            entry_points = archive.read(entry_points_name).decode("utf-8")
            self.assertIn("Name: latencyops", metadata)
            self.assertIn("Version: 0.1.2", metadata)
            self.assertIn("latencyops = latencyops.cli:main", entry_points)
            self.assertFalse(any(Path(name).name == ".env" for name in names))
            self.assertFalse(any(Path(name).name.startswith("REAL_") for name in names))

        with tarfile.open(sdists[-1], "r:gz") as archive:
            names = archive.getnames()
            self.assertTrue(any(name.endswith("/pyproject.toml") for name in names))
            self.assertTrue(any(name.endswith("/LICENSE") for name in names))
            self.assertTrue(any(name.endswith("/src/latencyops/cli.py") for name in names))
            self.assertFalse(any(Path(name).name == ".env" for name in names))
            self.assertFalse(any(Path(name).name.startswith("REAL_") for name in names))

    @unittest.skipUnless(
        os.environ.get("LATENCYOPS_RELEASE_ARTIFACTS") == "1",
        "set LATENCYOPS_RELEASE_ARTIFACTS=1 after building release artifacts",
    )
    def test_wheel_installs_in_isolated_environment_and_runs_cli(self):
        wheels = sorted((PROJECT_ROOT / "dist").glob("latencyops-*.whl"))
        self.assertTrue(wheels)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "venv"
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
            scripts = environment / ("Scripts" if os.name == "nt" else "bin")
            python = scripts / ("python.exe" if os.name == "nt" else "python")
            executable = scripts / ("latencyops.exe" if os.name == "nt" else "latencyops")
            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)
            subprocess.run(
                [str(python), "-m", "pip", "install", "--no-index", str(wheels[-1])],
                check=True,
                cwd=root,
                env=clean_env,
                capture_output=True,
                text=True,
            )
            imported = subprocess.run(
                [str(python), "-c", "import latencyops; print(latencyops.__file__)"],
                check=True,
                cwd=root,
                env=clean_env,
                capture_output=True,
                text=True,
            )
            self.assertNotIn(str((PROJECT_ROOT / "src").resolve()).lower(), imported.stdout.lower())
            help_result = subprocess.run(
                [str(executable), "--help"],
                check=True,
                cwd=root,
                env=clean_env,
                capture_output=True,
                text=True,
            )
            self.assertIn("gateway", help_result.stdout)
            request = root / "request.json"
            request.write_text(
                json.dumps({"prompt_tokens": 10, "expected_output_tokens": 5, "difficulty": 0.1}),
                encoding="utf-8",
            )
            plan_result = subprocess.run(
                [str(executable), "plan", "--request", str(request)],
                check=True,
                cwd=root,
                env=clean_env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(json.loads(plan_result.stdout)["model_tier"], "small")


class OrchestratorIntegrationTests(unittest.TestCase):
    def test_vllm_metrics_profile_normalizes_documented_names(self):
        collector = VLLMMetricsCollector("http://unused/metrics", queue_capacity=10)
        signals = collector.normalize({
            "vllm:num_requests_waiting": 8,
            "vllm:kv_cache_usage_perc": 0.5,
        })
        self.assertEqual(signals.queue_pressure, 0.8)
        self.assertEqual(signals.cache_pressure, 0.5)

    def test_sglang_metrics_profile_normalizes_queue_and_token_usage(self):
        collector = SGLangMetricsCollector("http://unused/metrics", queue_capacity=10)
        signals = collector.normalize({
            "sglang:num_queue_reqs": 4,
            "sglang:token_usage": 0.75,
        })
        self.assertEqual(signals.queue_pressure, 0.4)
        self.assertEqual(signals.cache_pressure, 0.75)

    def test_profiles_identify_confirmed_orchestrator_metric_families(self):
        self.assertEqual(PROFILES["vllm"]["queue_depth"], "vllm:num_requests_waiting")
        self.assertEqual(PROFILES["sglang"]["cache_pressure"], "sglang:token_usage")

    def test_metric_fixtures_cover_vllm_sglang_and_tensorrtllm(self):
        fixture_dir = Path(__file__).parent / "fixtures"
        vllm = VLLMMetricsCollector("http://unused/metrics", queue_capacity=10)
        sglang = SGLangMetricsCollector("http://unused/metrics", queue_capacity=10)
        tensorrtllm = TensorRTLLMMetricsCollector("http://unused/metrics", queue_capacity=10)
        self.assertEqual(vllm.normalize(vllm.parse((fixture_dir / "vllm_metrics.txt").read_text())).cache_pressure, 0.5)
        self.assertEqual(sglang.normalize(sglang.parse((fixture_dir / "sglang_metrics.txt").read_text())).queue_pressure, 0.4)
        self.assertEqual(tensorrtllm.normalize(tensorrtllm.parse((fixture_dir / "tensorrtllm_metrics.txt").read_text())).cache_pressure, 0.6)


class PolicyIntegrationTests(unittest.TestCase):
    def test_enforcement_reports_unsupported_plan_features(self):
        contract = LatencyContract(deadline_ms=1000)
        requested = InferencePlan("standard", "int8", "compressed", 8, True)
        result = enforce_plan(contract, requested, ProviderCapabilities())
        self.assertEqual(result.enforced.precision, "fp16")
        self.assertEqual(result.enforced.context_policy, "full")
        self.assertEqual(result.enforced.speculative_tokens, 0)
        self.assertEqual(result.enforced.early_exit, False)
        self.assertIn("precision", result.unsupported)

    def test_runtime_signals_normalize_provider_metrics(self):
        signals = RuntimeSignalNormalizer().normalize(
            {"queue_depth": 8, "queue_capacity": 10, "kv_cache_usage": 5, "kv_cache_capacity": 10, "healthy": 1},
            "qwen",
        )
        self.assertEqual(signals.queue_pressure, 0.8)
        self.assertEqual(signals.cache_pressure, 0.5)
        self.assertTrue(signals.provider_health["qwen"])

    def test_policy_experiment_compares_selected_and_baseline(self):
        contract = LatencyContract(deadline_ms=1000, risk_class="low")
        profile = RequestProfile(prompt_tokens=2, expected_output_tokens=2, difficulty=0.1)
        result = PolicyExperiment().run(
            StaticProvider("mock", "answer"), "synthetic", contract, profile,
            quality_evaluator=lambda text: QualityOutcome(0.95, True),
        )
        self.assertTrue(result.selected["deadline_satisfied"])
        self.assertTrue(result.baseline["quality_satisfied"])

    def test_adaptive_selector_learns_quality_eligible_provider(self):
        selector = AdaptiveProviderSelector()
        selector.observe("fast", 10, 0.5)
        selector.observe("safe", 20, 0.95)
        selected = selector.select(
            LatencyContract(deadline_ms=1000, quality_floor=0.9),
            [ProviderCandidate("fast", "standard"), ProviderCandidate("safe", "standard")],
        )
        self.assertEqual(selected.name, "safe")


class ComprehensiveBenchmarkTests(unittest.TestCase):
    def test_scenario_runner_reports_quality_cost_and_metadata(self):
        scenario = BenchmarkScenario(
            "long-warm-cache", "synthetic prompt", LatencyContract(deadline_ms=1000, quality_floor=0.8),
            RequestProfile(prompt_tokens=100, expected_output_tokens=50),
            cache_state="warm", network_region="test", estimated_cost_per_1k_tokens=0.01,
        )
        result = ScenarioBenchmarkRunner().run(
            StaticProvider("standard", "answer"), scenario,
            InferencePlan("standard", "fp16", "full", 0, False),
            BenchmarkConfig(warmup_runs=1, measured_runs=2, concurrency=2),
            quality_evaluator=lambda text: 0.9 if text == "answer" else 0.0,
        )
        self.assertEqual(result["runs"], 2)
        self.assertEqual(result["warmup_excluded"], True)
        self.assertEqual(result["cache_state"], "warm")
        self.assertEqual(result["network_region"], "test")
        self.assertEqual(result["quality_satisfied"], True)
        self.assertEqual(result["estimated_cost"], 0.0015)


class ProactivePlanningTests(unittest.TestCase):
    def test_critical_request_disables_speculation(self):
        plan = ElasticInferencePlanner().plan(
            LatencyContract(deadline_ms=1000, risk_class="critical"),
            RequestProfile(prompt_tokens=100, expected_output_tokens=64, draft_acceptance=0.95),
        )
        self.assertEqual(plan.speculative_tokens, 0)

    def test_critical_request_overrides_queue_pressure(self):
        plan = ElasticInferencePlanner().plan(
            LatencyContract(deadline_ms=1000, risk_class="critical"),
            RequestProfile(prompt_tokens=100, expected_output_tokens=20, difficulty=0.9, queue_pressure=1.0),
        )
        self.assertEqual(plan.model_tier, "large")
        self.assertEqual(plan.precision, "fp16")
        self.assertFalse(plan.early_exit)

    def test_critical_long_context_preserves_full_context(self):
        plan = ElasticInferencePlanner().plan(
            LatencyContract(deadline_ms=500, quality_floor=0.95, risk_class="critical"),
            RequestProfile(prompt_tokens=10000, expected_output_tokens=20, difficulty=0.9, cache_pressure=0.9),
        )
        self.assertEqual(plan.model_tier, "large")
        self.assertEqual(plan.precision, "fp16")
        self.assertEqual(plan.context_policy, "full")
        self.assertFalse(plan.early_exit)

    def test_provider_selection_avoids_unhealthy_provider(self):
        planner = ElasticInferencePlanner()
        selected = planner.select_provider(
            LatencyContract(deadline_ms=1000),
            [
                ProviderCandidate("slow", "standard", True, 900, 200),
                ProviderCandidate("healthy", "standard", True, 100, 20),
                ProviderCandidate("down", "standard", False, 1, 1),
            ],
        )
        self.assertEqual(selected.name, "healthy")

    def test_quality_failure_triggers_replan(self):
        controller = ProactiveController()
        contract = LatencyContract(deadline_ms=1000, quality_floor=0.9)
        profile = RequestProfile(prompt_tokens=10, expected_output_tokens=10, difficulty=0.1)
        current = controller.initial_plan(contract, profile)
        decision = controller.replan(contract, profile, current, LatencySample(10, 2, 20), QualityOutcome(0.5, False))
        self.assertTrue(decision.changed)
        self.assertEqual(decision.plan.model_tier, "standard")

    def test_shadow_comparison_contains_selected_and_baseline(self):
        shadow = ProactiveController().shadow(
            LatencyContract(deadline_ms=1000, risk_class="low"),
            RequestProfile(prompt_tokens=10, expected_output_tokens=10, difficulty=0.1),
        )
        self.assertTrue(shadow.selected.early_exit)
        self.assertEqual(shadow.baseline.rationale, ("static baseline",))

    def test_workload_matrix_reports_deadline_satisfaction(self):
        plan = InferencePlan("standard", "fp16", "full", 0, False)
        case = WorkloadCase(
            "short-low-risk", "synthetic", LatencyContract(deadline_ms=1000),
            RequestProfile(prompt_tokens=1, expected_output_tokens=1),
        )
        rows = run_workload_matrix(StaticProvider("standard", "ok"), [case], plan)
        self.assertTrue(rows[0]["deadline_satisfied"])


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = ElasticInferencePlanner()

    def test_long_context_uses_context_reduction(self):
        plan = self.planner.plan(
            LatencyContract(deadline_ms=1000),
            RequestProfile(prompt_tokens=10000, expected_output_tokens=20, cache_pressure=0.6),
        )
        self.assertIn(plan.context_policy, {"selective", "compressed"})
        self.assertIn(plan.precision, {"int8", "fp16"})

    def test_high_risk_request_preserves_full_precision(self):
        plan = self.planner.plan(
            LatencyContract(deadline_ms=1000, risk_class="critical"),
            RequestProfile(prompt_tokens=100, expected_output_tokens=20, difficulty=0.9, cache_pressure=0.9),
        )
        self.assertEqual(plan.model_tier, "large")
        self.assertEqual(plan.precision, "fp16")
        self.assertFalse(plan.early_exit)

    def test_easy_low_risk_request_can_exit_early(self):
        plan = self.planner.plan(
            LatencyContract(deadline_ms=1000, risk_class="low"),
            RequestProfile(prompt_tokens=100, expected_output_tokens=10, difficulty=0.1),
        )
        self.assertEqual(plan.model_tier, "small")
        self.assertTrue(plan.early_exit)

    def test_high_acceptance_long_output_uses_speculation(self):
        plan = self.planner.plan(
            LatencyContract(deadline_ms=1000),
            RequestProfile(prompt_tokens=100, expected_output_tokens=64, draft_acceptance=0.9),
        )
        self.assertGreater(plan.speculative_tokens, 0)


class RouterTests(unittest.TestCase):
    def test_router_forwards_streaming_to_planned_tier(self):
        router = ModelRouter({
            tier: StaticProvider(tier, response=tier)
            for tier in ("small", "standard", "large")
        })
        plan = ElasticInferencePlanner().plan(
            LatencyContract(deadline_ms=1000),
            RequestProfile(prompt_tokens=10, expected_output_tokens=10),
        )
        self.assertEqual(
            [chunk.text for chunk in router.stream("synthetic prompt", plan)],
            ["standard"],
        )

    def test_router_selects_planned_tier(self):
        router = ModelRouter({
            tier: StaticProvider(tier, response=tier)
            for tier in ("small", "standard", "large")
        })
        plan = ElasticInferencePlanner().plan(
            LatencyContract(deadline_ms=1000, risk_class="low"),
            RequestProfile(prompt_tokens=10, expected_output_tokens=10, difficulty=0.1),
        )
        self.assertEqual(router.generate("synthetic prompt", plan), "small")


class OperationsTests(unittest.TestCase):
    def test_openai_compatible_adapter_maps_tier_and_streams(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                requests.append(json.loads(self.rfile.read(length)))
                if requests[-1]["stream"]:
                    body = (
                        b'data: {"choices":[{"text":"hello"}]}\n\n'
                        b'data: [DONE]\n\n'
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                else:
                    body = b'{"choices":[{"text":"hello"}]}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OpenAICompatibleProvider(
                "fake-vllm", f"http://127.0.0.1:{server.server_port}/v1/completions",
                model_names={"standard": "synthetic-model"},
            )
            plan = ElasticInferencePlanner().plan(
                LatencyContract(deadline_ms=1000),
                RequestProfile(prompt_tokens=10, expected_output_tokens=10),
            )
            self.assertEqual(provider.generate("synthetic", plan), "hello")
            self.assertEqual("".join(chunk.text for chunk in provider.stream("synthetic", plan)), "hello")
            self.assertEqual(requests[0]["model"], "synthetic-model")
            self.assertEqual(requests[1]["model"], "synthetic-model")
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()

    def test_chat_adapter_sends_messages_and_chat_template_options(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                requests.append(json.loads(self.rfile.read(length)))
                body = b'{"choices":[{"message":{"content":"chat hello"}}]}'
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            provider = OpenAIChatCompatibleProvider(
                "qwen", f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                model_names={"standard": "Qwen/Qwen3.6-35B-A3B-FP8"},
                chat_template_kwargs={"enable_thinking": False},
            )
            plan = ElasticInferencePlanner().plan(
                LatencyContract(deadline_ms=1000),
                RequestProfile(prompt_tokens=10, expected_output_tokens=10),
            )
            self.assertEqual(provider.generate("Say hello.", plan), "chat hello")
            self.assertEqual(requests[0]["model"], "Qwen/Qwen3.6-35B-A3B-FP8")
            self.assertEqual(requests[0]["messages"][0]["content"], "Say hello.")
            self.assertEqual(requests[0]["chat_template_kwargs"]["enable_thinking"], False)
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()

    def test_cache_signal_exposes_normalized_pressure(self):
        self.assertEqual(CachePressureSignal(3, 4).pressure, 0.75)

    def test_priority_scheduler_selects_high_priority_request(self):
        scheduler = PriorityScheduler()
        scheduler.submit("normal", priority=1)
        scheduler.submit("urgent", priority=5)
        self.assertEqual(scheduler.next().payload, "urgent")
        self.assertEqual(scheduler.depth, 1)

    def test_telemetry_exporter_receives_content_free_record(self):
        exporter = PrometheusExporter()
        recorder = TelemetryRecorder([exporter])
        recorder.record(TelemetryRecord("mock", "standard", LatencySample(10, 2, 20), cache_pressure=0.5))
        self.assertIn("latencyops_requests_total 1", exporter.render())

    def test_benchmark_comparison_ranks_by_tail_latency(self):
        rows = compare({
            "slow": [LatencySample(10, 2, 100)],
            "fast": [LatencySample(8, 2, 50)],
        })
        self.assertEqual([row["provider"] for row in rows], ["fast", "slow"])

    def test_http_gateway_returns_openai_shaped_json(self):
        providers = {
            tier: CallableProvider(tier, lambda prompt, plan: "http response")
            for tier in ("small", "standard", "large")
        }
        service = GatewayService(ModelRouter(providers), ElasticInferencePlanner())
        server = create_server("127.0.0.1", 0, service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_port}/v1/completions",
                data=json.dumps({"prompt": "synthetic"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                result = json.loads(response.read())
            self.assertEqual(result["choices"][0]["text"], "http response")
        finally:
            server.shutdown()
            thread.join(timeout=1)
            server.server_close()

    def test_gateway_stream_returns_provider_chunks(self):
        providers = {
            tier: CallableProvider(tier, lambda prompt, plan: "streamed")
            for tier in ("small", "standard", "large")
        }
        service = GatewayService(ModelRouter(providers), ElasticInferencePlanner())
        self.assertEqual(
            [chunk.text for chunk in service.stream({"prompt": "synthetic", "stream": True})],
            ["streamed"],
        )

    def test_gateway_uses_adaptive_provider_candidate(self):
        providers = {
            "fast": CallableProvider("fast", lambda prompt, plan: "fast"),
            "safe": CallableProvider("safe", lambda prompt, plan: "safe"),
        }
        service = GatewayService(
            ModelRouter({tier: providers["safe"] for tier in ("small", "standard", "large")}),
            ElasticInferencePlanner(),
            provider_candidates=[
                ProviderCandidate("fast", "standard"),
                ProviderCandidate("safe", "standard"),
            ],
            provider_lookup=providers,
        )
        service.provider_selector.observe("fast", 10, 0.5)
        service.provider_selector.observe("safe", 20, 0.95)
        result = service.complete({"prompt": "synthetic request", "quality_floor": 0.9})
        self.assertEqual(result["choices"][0]["text"], "safe")

    def test_gateway_returns_plan_and_telemetry_without_prompt_storage(self):
        providers = {
            tier: CallableProvider(tier, lambda prompt, plan: "synthetic")
            for tier in ("small", "standard", "large")
        }
        exporter = PrometheusExporter()
        service = GatewayService(
            ModelRouter(providers), ElasticInferencePlanner(), TelemetryRecorder([exporter])
        )
        result = service.complete({"prompt": "synthetic request", "cache_used": 1, "cache_capacity": 2})
        self.assertEqual(result["choices"][0]["text"], "synthetic")
        self.assertEqual(result["latencyops"]["plan"]["model_tier"], "standard")
        self.assertEqual(result["latencyops"]["plan_requested"]["model_tier"], "standard")
        self.assertEqual(result["latencyops"]["plan_enforced"]["precision"], "fp16")
        self.assertEqual(result["latencyops"]["unsupported_features"], ["precision"])
        self.assertEqual(len(service.telemetry.records), 1)
        self.assertNotIn("synthetic request", exporter.render())


class BenchmarkTests(unittest.TestCase):
    def test_streaming_benchmark_measures_ttft_and_tpot(self):
        class Clock:
            def __init__(self):
                self.values = iter([0.0, 0.1, 0.3, 0.5])

            def __call__(self):
                return next(self.values)

        class StreamingTestProvider:
            def stream(self, prompt, plan):
                del prompt, plan
                yield StreamChunk("Hello ", token_count=2)
                yield StreamChunk("world", token_count=1)

        plan = ElasticInferencePlanner().plan(
            LatencyContract(deadline_ms=1000),
            RequestProfile(prompt_tokens=10, expected_output_tokens=10),
        )
        result = BenchmarkRunner(clock=Clock()).run_streaming(
            StreamingTestProvider(), "synthetic prompt", plan
        )
        self.assertEqual(result.text, "Hello world")
        self.assertEqual(result.sample.ttft_ms, 100)
        self.assertEqual(result.sample.tpot_ms, 400)
        self.assertEqual(result.sample.end_to_end_ms, 500)

    def test_benchmark_continues_after_provider_failure(self):
        class FailingProvider(StaticProvider):
            def generate(self, prompt, plan):
                if prompt == "bad":
                    raise RuntimeError("synthetic failure")
                return super().generate(prompt, plan)

        plan = ElasticInferencePlanner().plan(
            LatencyContract(deadline_ms=1000),
            RequestProfile(prompt_tokens=10, expected_output_tokens=10),
        )
        samples = BenchmarkRunner().run(FailingProvider("standard"), ["good", "bad"], plan)
        self.assertEqual(len(samples), 2)
        self.assertTrue(samples[0].success)
        self.assertFalse(samples[1].success)


if __name__ == "__main__":
    unittest.main()
