"""Shared helpers: start a local llama.cpp server on demand and stream one chat turn."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import requests

HOST = os.getenv("LLAMA_CPP_HOST", "127.0.0.1")
CONTEXT_SIZE = os.getenv("LLAMA_CPP_CTX", "8192")
GPU_LAYERS = os.getenv("LLAMA_CPP_NGL", "99")
STARTUP_TIMEOUT = float(os.getenv("LLAMA_CPP_STARTUP_TIMEOUT", "600"))


def resolve_port(hf_repo: str) -> int:
    """Give each model its own default port so warm servers can coexist."""
    override = os.getenv("LLAMA_CPP_PORT")
    if override:
        return int(override)
    return 8080 + zlib.crc32(hf_repo.encode()) % 100


def _is_healthy(session: requests.Session, base_url: str) -> bool:
    try:
        return session.get(f"{base_url}/health", timeout=1).status_code == 200
    except requests.RequestException:
        return False


def _log_tail(log_path: Path, lines: int = 15) -> str:
    try:
        tail = log_path.read_text(errors="replace").strip().splitlines()[-lines:]
    except OSError:
        return ""
    return "\n".join(tail)


@contextmanager
def llama_server(session: requests.Session, base_url: str, hf_repo: str):
    """Yield once the server is ready; spawn one only if nothing is listening."""
    if _is_healthy(session, base_url):
        yield None
        return

    binary = shutil.which("llama-server")
    if binary is None:
        raise RuntimeError("llama-server not found on PATH (brew install llama.cpp)")

    host, port = base_url.removeprefix("http://").split(":")
    log_path = Path(
        os.getenv("LLAMA_CPP_LOG")
        or Path(tempfile.gettempdir()) / f"llama-server-{port}.log"
    )
    print(f"Starting llama-server (log: {log_path})")

    log_file = log_path.open("w")
    proc = subprocess.Popen(
        [
            binary,
            "-hf",
            hf_repo,
            "--host",
            host,
            "--port",
            port,
            "-c",
            CONTEXT_SIZE,
            "-ngl",
            GPU_LAYERS,
            "-fa",
            "on",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not _is_healthy(session, base_url):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {proc.returncode}\n{_log_tail(log_path)}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"llama-server not ready after {STARTUP_TIMEOUT:.0f}s\n{_log_tail(log_path)}"
                )
            time.sleep(0.25)
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()


def stream_chat(
    session: requests.Session,
    base_url: str,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    started_at = time.perf_counter()
    first_token_at = None
    usage: dict = {}
    pieces: list[str] = []

    with session.post(
        f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=(5, 600)
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                break
            data = json.loads(chunk)
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(piece)
                    print(piece, end="", flush=True)

    elapsed = time.perf_counter() - started_at
    completion = usage.get("completion_tokens", 0)
    ttft = (first_token_at - started_at) if first_token_at else float("nan")
    rate = f" | {completion / elapsed:.1f} tok/s" if completion and elapsed else ""
    print(f"\n\nTTFT: {ttft:.2f}s | total: {elapsed:.2f}s{rate}")
    return "".join(pieces)


@contextmanager
def chat_session(*, title: str, hf_repo: str):
    """Keep one llama-server alive and yield an `ask` callable for many turns."""
    base_url = f"http://{HOST}:{resolve_port(hf_repo)}"

    print(f"=== {title} ===")
    print(f"Repo: {hf_repo}")
    print(f"Server: {base_url}\n")

    with requests.Session() as session:
        with llama_server(session, base_url, hf_repo) as proc:
            if proc is None:
                print("(reusing running server)\n")

            def ask(
                user_message: str,
                *,
                system_message: str = "",
                temperature: float = 0.7,
                max_tokens: int = 128,
            ) -> str:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ]
                return stream_chat(
                    session,
                    base_url,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            yield ask


def run_example(
    *,
    title: str,
    hf_repo: str,
    system_message: str,
    user_message: str,
    temperature: float = 0.7,
    max_tokens: int = 128,
) -> str:
    with chat_session(title=title, hf_repo=hf_repo) as ask:
        return ask(
            user_message,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
