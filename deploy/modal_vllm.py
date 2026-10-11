"""Serve a Hugging Face model on a Modal GPU behind an OpenAI-compatible API.

One command deploys it; Modal prints the URL:

    HERMES_MODEL=Qwen/Qwen2.5-7B-Instruct modal deploy deploy/modal_vllm.py

Then point Hermes at it, in a config or a sweep:

    llm: {provider: vllm, model: Qwen/Qwen2.5-7B-Instruct, base_url: https://<...>.modal.run/v1}

The server scales to zero after five idle minutes, so it costs nothing
between runs; the first request after that waits for a cold start (one to a
few minutes while the model loads). Weights are cached in a Modal Volume, so
they download once. Gated models (for example meta-llama/Llama-3.1-8B-Instruct)
need a Hugging Face token: create it once with
`modal secret create huggingface HF_TOKEN=...` and deploy with
HERMES_HF_SECRET=huggingface.

Settings, all optional, read when you deploy:
    HERMES_MODEL     Hugging Face model id   (default Qwen/Qwen2.5-7B-Instruct)
    HERMES_GPU       Modal GPU type          (default L40S; A100-80GB, H100 for large models)
    HERMES_MAX_LEN   context length          (default 16384)
"""

import os
import subprocess
import threading

import modal

# Settings are read on your machine at deploy time and baked into the image,
# so the container reads the same values and the app's definition (including
# whether a secret is attached) is identical on both sides.
SETTINGS = {
    "HERMES_MODEL": os.environ.get("HERMES_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    "HERMES_GPU": os.environ.get("HERMES_GPU", "L40S"),
    "HERMES_MAX_LEN": os.environ.get("HERMES_MAX_LEN", "16384"),
    "HERMES_HF_SECRET": os.environ.get("HERMES_HF_SECRET", ""),
}
MODEL = SETTINGS["HERMES_MODEL"]
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.29.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", **SETTINGS})
)
weights = modal.Volume.from_name("hermes-hf-cache", create_if_missing=True)
compiled = modal.Volume.from_name("hermes-vllm-cache", create_if_missing=True)


def _secrets():
    name = SETTINGS["HERMES_HF_SECRET"]
    return [modal.Secret.from_name(name)] if name else []


app = modal.App("hermes-vllm-" + MODEL.split("/")[-1].lower().replace(".", "-")[:40])


@app.function(
    image=image,
    gpu=SETTINGS["HERMES_GPU"],
    scaledown_window=5 * 60,
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": weights, "/root/.cache/vllm": compiled},
    secrets=_secrets(),
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=15 * 60)
def serve():
    model = os.environ["HERMES_MODEL"]
    server = subprocess.Popen([
        "vllm", "serve", model,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", os.environ["HERMES_MAX_LEN"],
        "--enable-prefix-caching",
    ])

    # If vLLM exits (a gated model, out of memory), end the container at once
    # instead of holding the GPU until the startup timeout.
    def watch():
        code = server.wait()
        print(f"vLLM exited with code {code}; stopping the container.", flush=True)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
