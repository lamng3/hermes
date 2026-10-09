"""Call a local Ollama model, or an OpenAI-compatible API when requested."""

import os

import httpx

_OLLAMA_PREFERENCE = ("llama3.1", "llama3", "mistral", "phi3", "qwen2.5", "qwen")
_SYSTEM = (
    "You translate competency questions into one SPARQL query. "
    "Return only the query."
)


def complete(prompt):
    provider = _provider()
    if provider == "openai":
        content, model = _openai_chat(prompt)
    else:
        content, model = _ollama_chat(prompt)
    return content, model, provider


def _provider():
    configured = os.environ.get("HERMES_LLM_PROVIDER", "").strip().lower()
    if configured == "openai":
        return "openai"
    return "ollama"


def _ollama_chat(prompt):
    base_url = os.environ.get(
        "HERMES_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
    ).rstrip("/")
    try:
        model = _ollama_model(base_url)
        response = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=180.0,
        )
    except httpx.ConnectError as exc:
        raise LookupError(
            f"Ollama is not running at {base_url}. Start it with `ollama serve`."
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
    return response.json()["message"]["content"], model


def _ollama_model(base_url):
    configured = os.environ.get("HERMES_OLLAMA_MODEL", "").strip()
    if configured:
        return configured
    response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
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


def _openai_chat(prompt):
    api_key = os.environ.get("HERMES_LLM_API_KEY", "").strip()
    if not api_key:
        raise LookupError(
            "Set HERMES_LLM_API_KEY, or leave HERMES_LLM_PROVIDER unset to use Ollama."
        )
    base_url = os.environ.get(
        "HERMES_LLM_BASE_URL", "https://api.together.xyz/v1"
    ).rstrip("/")
    model = os.environ.get(
        "HERMES_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    )
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=40.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"], model
