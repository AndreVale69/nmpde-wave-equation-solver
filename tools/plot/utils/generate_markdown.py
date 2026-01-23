import os
from typing import Iterator, Optional, Dict

from ollama import Client


def ollama_generate_markdown(
        prompt: str,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        headers: dict = None,
) -> str:
    """
    Generate a Markdown response using Ollama.

    Works for:
    - Local Ollama daemon (default): host=http://localhost:11434 (no auth)
    - Remote Ollama daemon: host=http://<server>:11434 (no auth unless you add your own proxy)
    - Ollama Cloud API: host=https://ollama.com (requires OLLAMA_API_KEY)
    """

    headers = headers or {}

    # If you're targeting Ollama Cloud API (ollama.com), you need an API key header.
    # Docs example uses: Authorization: Bearer $OLLAMA_API_KEY
    if "ollama.com" in host:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY is not set. Required for Ollama Cloud API (host=https://ollama.com)."
            )
        headers["Authorization"] = f"Bearer {api_key}"

    client = Client(host=host, headers=headers)

    # Use /api/generate style call
    resp = client.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options={"temperature": temperature},
    )

    return resp.get("response", "")


def _ollama_headers_for_host(host: str, headers: Optional[Dict] = None) -> Dict:
    headers = dict(headers or {})
    if "ollama.com" in host:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY is not set. Required for Ollama Cloud API (host=https://ollama.com)."
            )
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def ollama_generate_markdown_stream(
        prompt: str,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        headers: Optional[Dict] = None,
) -> Iterator[str]:
    """
    Stream a Markdown response from Ollama.

    Yields incremental text chunks (strings) that you can append to an accumulator.
    """
    client = Client(host=host, headers=_ollama_headers_for_host(host, headers))

    # stream=True -> returns an iterator of partial responses
    for part in client.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options={"temperature": temperature},
    ):
        # Each 'part' is a dict; 'response' holds the incremental text
        text = part.get("response", "")
        if text:
            yield text
