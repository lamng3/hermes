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
    """Interface: complete(prompt, system) -> text."""

    provider = ""
    model = ""

    def complete(self, prompt, system=SYSTEM):
        raise NotImplementedError


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
        return response.json()["message"]["content"]

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


class OpenAICompatible(LLM):
    provider = "openai"

    def __init__(self, model=None, base_url=None, api_key=None, temperature=0, timeout=60.0):
        self.api_key = api_key or os.environ.get("HERMES_LLM_API_KEY", "").strip()
        self.base_url = (
            base_url or os.environ.get("HERMES_LLM_BASE_URL", "https://api.together.xyz/v1")
        ).rstrip("/")
        self.model = model or os.environ.get(
            "HERMES_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
        )
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt, system=SYSTEM):
        if not self.api_key:
            raise LookupError(
                "Set HERMES_LLM_API_KEY, or leave HERMES_LLM_PROVIDER unset to use Ollama."
            )
        response = _with_retries(
            lambda: httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
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
        return response.json()["choices"][0]["message"]["content"]


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
        completion = self.llm.complete(prompt, system)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"model": self.model, "system": system, "prompt": prompt,
                        "completion": completion}),
            encoding="utf-8",
        )
        return completion


def create(provider=None, cache=None, **options):
    """Factory: build a client from a config section."""
    provider = (provider or os.environ.get("HERMES_LLM_PROVIDER", "") or "ollama").lower()
    if provider == "openai":
        llm = OpenAICompatible(**options)
    elif provider == "ollama":
        llm = Ollama(**options)
    else:
        raise ValueError(f"Unknown LLM provider {provider!r}. Use ollama or openai.")
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
