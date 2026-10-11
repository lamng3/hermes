"""LLM clients. Systems receive one, so experiments choose the model.

`from_env()` builds the default client: a local Ollama model, or an
OpenAI-compatible API when HERMES_LLM_PROVIDER=openai. `CachedLLM` wraps any
client with an on-disk cache keyed by model, system prompt and prompt, so a
rerun of an experiment replays the same completions.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

_OLLAMA_PREFERENCE = ("llama3.1", "llama3", "mistral", "phi3", "qwen2.5", "qwen")
SYSTEM = (
    "You translate competency questions into one SPARQL query. "
    "Return only the query."
)


class LLM:
    """Interface: complete(prompt, system) -> text.

    Clients also count what they spend in `usage`: calls, input and output
    tokens, and seconds of model time.
    """

    provider = ""
    model = ""

    def complete(self, prompt, system=SYSTEM):
        raise NotImplementedError

    @property
    def usage(self):
        if "_usage" not in self.__dict__:
            self._usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        return self._usage

    def record(self, input_tokens, output_tokens, seconds):
        usage = self.usage
        usage["calls"] += 1
        usage["input_tokens"] += int(input_tokens or 0)
        usage["output_tokens"] += int(output_tokens or 0)
        usage["seconds"] += float(seconds or 0.0)


class Ollama(LLM):
    provider = "ollama"

    def __init__(self, model=None, base_url=None, temperature=0, timeout=180.0):
        self.base_url = (
            base_url or os.environ.get("HERMES_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self._model = model or os.environ.get("HERMES_OLLAMA_MODEL", "").strip() or None
        self.temperature = temperature
        self.timeout = timeout

    @property
    def model(self):
        if self._model is None:
            self._model = self._installed_model()
        return self._model

    def complete(self, prompt, system=SYSTEM):
        try:
            model = self.model
            response = _with_retries(
                lambda: httpx.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "keep_alive": "10m",
                        "options": {"temperature": self.temperature},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=self.timeout,
                )
            )
        except httpx.ConnectError as exc:
            raise LookupError(
                f"Ollama is not running at {self.base_url}. Start it with `ollama serve`."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LookupError(
                "Ollama took too long to answer. Try again, or set "
                "HERMES_OLLAMA_MODEL to a smaller model."
            ) from exc
        if response.status_code == 404:
            raise LookupError(
                f"Ollama model {model} is not installed. Run `ollama pull {model}`."
            )
        response.raise_for_status()
        data = response.json()
        self.record(data.get("prompt_eval_count"), data.get("eval_count"),
                    (data.get("total_duration") or 0) / 1e9)
        return data["message"]["content"]

    def _installed_model(self):
        response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        names = [
            item.get("name") or item.get("model")
            for item in response.json().get("models") or []
        ]
        names = [name for name in names if name]
        for preferred in _OLLAMA_PREFERENCE:
            for name in names:
                if name == preferred or name.startswith(preferred + ":"):
                    return name
        if not names:
            raise LookupError(
                "Ollama has no models. Run `ollama pull llama3.1`, then translate again."
            )
        return names[0]


# Hosted APIs that speak the OpenAI chat-completions protocol:
# name -> (base URL, environment variable holding the key).
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "huggingface": ("https://router.huggingface.co/v1", "HF_TOKEN"),
    # A server you run, such as vLLM on Modal (deploy/modal_vllm.py): give base_url.
    "vllm": (None, "HERMES_LLM_API_KEY"),
}


class OpenAICompatible(LLM):
    """Any OpenAI-compatible chat API: a hosted provider or a vLLM server.

    The key comes from `api_key`, the provider's own variable (for example
    OPENAI_API_KEY), or HERMES_LLM_API_KEY; the URL from `base_url`,
    HERMES_LLM_BASE_URL, or the provider preset.
    """

    def __init__(self, model=None, base_url=None, api_key=None, temperature=0, timeout=120.0,
                 provider="openai"):
        preset_url, key_env = PROVIDERS.get(provider, (None, "HERMES_LLM_API_KEY"))
        self.provider = provider
        self.api_key = (
            api_key
            or os.environ.get(key_env or "", "").strip()
            or os.environ.get("HERMES_LLM_API_KEY", "").strip()
        )
        self.key_env = key_env
        base = base_url or os.environ.get("HERMES_LLM_BASE_URL") or preset_url
        if not base:
            raise ValueError(f"Provider {provider!r} needs base_url, for example a vLLM server.")
        self.base_url = base.rstrip("/")
        self.model = model or os.environ.get("HERMES_LLM_MODEL", "")
        if not self.model:
            raise ValueError(f"Provider {provider!r} needs a model name.")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt, system=SYSTEM):
        if not self.api_key and self.provider != "vllm":
            raise LookupError(
                f"Set {self.key_env} (or HERMES_LLM_API_KEY) for provider {self.provider!r}."
            )
        started = time.time()
        response = _with_retries(
            lambda: httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key or 'none'}"},
                json={
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.timeout,
            )
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        self.record(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    time.time() - started)
        return data["choices"][0]["message"]["content"]


class CachedLLM(LLM):
    """Decorator: replay completions from a directory of JSON files."""

    def __init__(self, llm, directory):
        self.llm = llm
        self.directory = Path(directory)
        self.calls = 0
        self.hits = 0

    @property
    def provider(self):
        return self.llm.provider

    @property
    def model(self):
        return self.llm.model

    def complete(self, prompt, system=SYSTEM):
        key = hashlib.sha256(
            json.dumps([self.provider, self.model, system, prompt]).encode()
        ).hexdigest()
        path = self.directory / key[:2] / f"{key}.json"
        self.calls += 1
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["completion"]
        before = dict(self.llm.usage)
        completion = self.llm.complete(prompt, system)
        spent = {k: self.llm.usage[k] - before[k] for k in before}
        self.record(spent["input_tokens"], spent["output_tokens"], spent["seconds"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": self.model, "system": system, "prompt": prompt,
                        "completion": completion, "usage": spent}),
            encoding="utf-8",
        )
        return completion


def create(provider=None, cache=None, **options):
    """Factory: build a client from a config section.

    provider is "ollama" (local, the default) or a key of PROVIDERS:
    openai, anthropic, together, huggingface, or vllm (with base_url).
    """
    provider = (provider or os.environ.get("HERMES_LLM_PROVIDER", "") or "ollama").lower()
    if provider == "ollama":
        llm = Ollama(**options)
    elif provider in PROVIDERS:
        llm = OpenAICompatible(provider=provider, **options)
    else:
        known = ", ".join(["ollama", *PROVIDERS])
        raise ValueError(f"Unknown LLM provider {provider!r}. Use one of: {known}.")
    return CachedLLM(llm, cache) if cache else llm


def from_env():
    return create()


def complete(prompt, system=SYSTEM):
    """One completion with the default client: (content, model, provider)."""
    llm = from_env()
    content = llm.complete(prompt, system)
    return content, llm.model, llm.provider


def _with_retries(send, attempts=3):
    for attempt in range(attempts):
        try:
            response = send()
        except (httpx.ReadError, httpx.RemoteProtocolError):
            if attempt == attempts - 1:
                raise
        else:
            if response.status_code not in (429, 502, 503, 504) or attempt == attempts - 1:
                return response
        time.sleep(2**attempt)
