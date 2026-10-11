"""Run a Hermes sweep on Modal instead of your laptop.

    modal run experiments/modal_sweep.py --file configs/sweeps/paper.yaml --only @qwen

The sweep runs in a Modal container (CPU only: the models are behind APIs or
a vLLM server from deploy/modal_vllm.py), so long runs do not hold your
machine. Results and the reply cache live in Modal Volumes and survive
interruptions; rerunning the same command resumes unfinished runs, and
--fresh starts new runs beside the old ones. Fetch the results afterwards:

    modal volume get hermes-results / results/ --force

API keys are passed from your environment if set: OPENAI_API_KEY,
ANTHROPIC_API_KEY, TOGETHER_API_KEY, HF_TOKEN, HERMES_LLM_API_KEY. ${VAR}
references in the sweep (such as HERMES_VLLM_URL) are expanded on your
machine before the sweep is sent.

Ollama models are local to your machine; on Modal, serve the same weights
with deploy/modal_vllm.py and use provider: vllm.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY", "HF_TOKEN",
        "HERMES_LLM_API_KEY", "MODAL_PROXY_TOKEN", "HERMES_VLLM_URL"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("rdflib>=7", "pyoxigraph>=0.4", "httpx>=0.27", "pyyaml>=6")
    .env({"PYTHONPATH": "/hermes/src", "HERMES_DATA": "/hermes/data"})
    .add_local_dir(ROOT / "src", "/hermes/src")
    .add_local_dir(ROOT / "data", "/hermes/data")
    .add_local_dir(ROOT / "configs", "/hermes/configs")
)
results = modal.Volume.from_name("hermes-results", create_if_missing=True)
cache = modal.Volume.from_name("hermes-llm-cache", create_if_missing=True)

app = modal.App("hermes-sweep")


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=24 * 60 * 60,
    volumes={"/hermes/results": results, "/hermes/.cache": cache},
    secrets=[modal.Secret.from_dict({k: os.environ[k] for k in KEYS if os.environ.get(k)})],
)
def run_sweep(sweep_text, only=None, limit=None, commit="", dirty=False, fresh=False):
    import sys

    os.chdir("/hermes")
    os.environ["HERMES_GIT_COMMIT"] = commit or "unknown"
    os.environ["HERMES_GIT_DIRTY"] = "1" if dirty else "0"
    from hermes.experiments import Progress, Run, expand_sweep, latest_unfinished

    path = Path("/tmp/sweep.yaml")
    path.write_text(sweep_text)
    finished = []
    for config in expand_sweep(path):
        if only and only not in config["name"]:
            continue
        if limit:
            config["limit"] = limit
        config.setdefault("llm", {})["cache"] = ".cache/llm"
        if not wait_until_ready(config["llm"]):
            print(f"  skipped: {config['llm'].get('base_url')} did not become ready")
            continue
        resume = None if fresh else latest_unfinished("results", config)
        print(f"{config['benchmark']}/{config['name']}" + (f" (resuming {resume})" if resume else ""))
        try:
            metrics = Run(config, "results", resume, observers=[Progress()]).execute()
            finished.append((config["benchmark"], config["name"],
                             metrics["official"]["average"].get("set_F")))
        except Exception as exc:
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        results.commit()
        cache.commit()
    return finished


def wait_until_ready(llm, minutes=20):
    """Wake a vLLM server (it scales to zero when idle) and wait for it."""
    import time

    import httpx

    url = (llm.get("base_url") or "").rstrip("/")
    if llm.get("provider") != "vllm" or not url:
        return True
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/models", timeout=25).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(5)
    return False


@app.local_entrypoint()
def main(file: str, only: str = "", limit: int = 0, fresh: bool = False):
    import subprocess

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", "src"))
    text = os.path.expandvars(Path(file).read_text(encoding="utf-8"))
    for bench, name, f1 in run_sweep.remote(text, only or None, limit or None, commit, dirty, fresh):
        print(f"{bench}/{name}: set F1 {f1:.3f}")
    print("Fetch the results with: modal volume get hermes-results / results/ --force")
