"""LLM clients. Systems receive one, so experiments choose the model.

`from_env()` builds the default client: a local Ollama model, or an
OpenAI-compatible API when HERMES_LLM_PROVIDER=openai. `CachedLLM` wraps any
client with an on-disk cache keyed by model, system prompt and prompt, so a
rerun of an experiment replays the same completions.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx

from hermes import runtime

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
            self._usage_lock = threading.Lock()
        return self._usage

    def record(self, input_tokens, output_tokens, seconds, meter=True):
        usage = self.usage
        with self._usage_lock:
            usage["calls"] += 1
            usage["input_tokens"] += int(input_tokens or 0)
            usage["output_tokens"] += int(output_tokens or 0)
            usage["seconds"] += float(seconds or 0.0)
        if meter:
            runtime.add(llm_calls=1, input_tokens=int(input_tokens or 0),
                        output_tokens=int(output_tokens or 0), llm_seconds=float(seconds or 0.0))


class Ollama(LLM):
    provider = "ollama"

    def __init__(self, model=None, base_url=None, temperature=0, timeout=180.0, max_tokens=2048):
        self.base_url = (
            base_url or os.environ.get("HERMES_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self._model = model or os.environ.get("HERMES_OLLAMA_MODEL", "").strip() or None
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens

    @property
    def model(self):
        if self._model is None:
            self._model = self._installed_model()
        return self._model

    def complete(self, prompt, system=SYSTEM):
        runtime.check()
        try:
            model = self.model
            response = _with_retries(
                lambda: httpx.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "keep_alive": "10m",
                        "options": {"temperature": self.temperature,
                                    "num_predict": self.max_tokens},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    timeout=runtime.remaining(self.timeout),
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
        return strip_thinking(data["message"]["content"])

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
    # Modal Shared Endpoints (billed per token): give the endpoint's base_url; the key is
    # "<token id>.<token secret>" from `modal workspace proxy-tokens create`.
    "modal": (None, "MODAL_PROXY_TOKEN"),
}


class OpenAICompatible(LLM):
    """Any OpenAI-compatible chat API: a hosted provider or a vLLM server.

    The key comes from `api_key`, the provider's own variable (for example
    OPENAI_API_KEY), or HERMES_LLM_API_KEY; the URL from `base_url`,
    HERMES_LLM_BASE_URL, or the provider preset.
    """

    def __init__(self, model=None, base_url=None, api_key=None, temperature=0, timeout=120.0,
                 provider="openai", extra=None, max_tokens=2048):
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
        # Extra request fields, for example {"chat_template_kwargs": {"enable_thinking": false}}
        # to turn off Qwen3's thinking on a vLLM server.
        self.extra = dict(extra or {})
        # Bound each reply: a small model that starts repeating itself would otherwise
        # generate up to its context length and run into the request timeout.
        self.max_tokens = max_tokens

    def complete(self, prompt, system=SYSTEM):
        runtime.check()
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
                    "max_tokens": self.max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    **self.extra,
                },
                timeout=runtime.remaining(self.timeout),
                follow_redirects=True,  # Modal answers slow requests with a 303 to poll
            )
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        self.record(usage.get("prompt_tokens"), usage.get("completion_tokens"),
                    time.time() - started)
        return strip_thinking(data["choices"][0]["message"]["content"] or "")


class CachedLLM(LLM):
    """Decorator: replay completions from a directory of JSON files.

    Its usage is the wrapped client's usage, so replayed completions cost
    nothing; calls and hits count every request. With refresh, every request
    goes to the model and the cache is only written, so a fresh run measures
    real time and tokens.
    """

    def __init__(self, llm, directory, refresh=False):
        self.llm = llm
        self.directory = Path(directory)
        self.refresh = refresh
        self.calls = 0
        self.hits = 0
        self._lock = threading.Lock()

    @property
    def provider(self):
        return self.llm.provider

    @property
    def model(self):
        return self.llm.model

    @property
    def usage(self):
        return self.llm.usage

    def complete(self, prompt, system=SYSTEM):
        runtime.check()
        settings = {k: getattr(self.llm, k, None) for k in ("temperature", "max_tokens", "extra")}
        key = hashlib.sha256(
            json.dumps([self.provider, self.model, settings, system, prompt],
                       sort_keys=True, default=str).encode()
        ).hexdigest()
        path = self.directory / key[:2] / f"{key}.json"
        with self._lock:
            self.calls += 1
        if path.exists() and not self.refresh:
            with self._lock:
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


class NeedReply(Exception):
    """A ReplayLLM was asked for a reply it does not have yet."""

    def __init__(self, prompt, system):
        super().__init__("a model reply is needed")
        self.prompt, self.system = prompt, system


class ReplayLLM(LLM):
    """Replies given in advance, in order; past the last, raise NeedReply with the prompt.

    A caller that cannot wait for a model inside a call (a web page reaching a
    model asynchronously) runs the pipeline, gets the prompt from NeedReply,
    obtains the reply its own way, and runs the pipeline again with one more
    reply: every earlier step repeats exactly, so the replies line up.
    """

    def __init__(self, replies=(), model="replayed", provider="replay"):
        self.replies = list(replies)
        self.model = model
        self.provider = provider
        self.prompts = []

    def complete(self, prompt, system=SYSTEM):
        self.prompts.append((prompt, system))
        index = len(self.prompts) - 1
        if index >= len(self.replies):
            raise NeedReply(prompt, system)
        return self.replies[index]


def strip_thinking(text):
    """Drop a reasoning block such as <think>...</think> that some models emit first."""
    import re

    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def create(provider=None, cache=None, refresh_cache=False, **options):
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
    return CachedLLM(llm, cache, refresh_cache) if cache else llm


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
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException,
                httpx.ConnectError):
            if attempt == attempts - 1:
                raise
        else:
            if response.status_code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                return response
        time.sleep(2**attempt)
