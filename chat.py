#!/usr/bin/env python3
"""
chat1.py - minimal Ollama harness with read_file / write_file tools.

Improvement over chat.py:
    * Shows an animated spinner while waiting for the model (before the first
      token or tool-call arrives), and while executing a tool, so the user can
      see the harness is working instead of a frozen terminal.

Usage:
    python chat1.py              # defaults to model "llama3"
    python chat1.py mistral      # use a specific model
"""

import json
import sys
import os

import requests
from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED
from rich.status import Status

# ----------------------------- config -----------------------------
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_ENDPOINT = f"{OLLAMA}/api/chat"
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OLLAMA_MODEL", "llama3")

console = Console()

# ------------------------ tool definitions ------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
]

# -------------------------- tool executor --------------------------
def execute_tool(name: str, args: dict) -> str:
    try:
        if name == "read_file":
            with open(args["path"], "r") as f:
                return f.read()
        if name == "write_file":
            with open(args["path"], "w") as f:
                f.write(args["content"])
            return f"OK - wrote {len(args['content']):,} chars to {args['path']}"
        return f"unknown tool: {name}"
    except Exception as e:
        return f"error: {e}"

# -------------------------- spinner helper --------------------------
def spinner(message: str) -> Status:
    """A ready-to-start spinner bound to the shared console."""
    return console.status(message, spinner="dots", speed=1.0)

# -------------------------- streaming call --------------------------
def ollama_stream(messages: list) -> tuple[str, list | None]:
    """
    POST to Ollama, stream tokens to the terminal.

    A spinner is shown while waiting for the very first token or tool-call so
    the user knows the model is thinking. As soon as real output starts the
    spinner is stopped and normal streaming resumes.

    Returns (accumulated_text, tool_calls_or_None).
    """
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "stream": True}

    # Show a spinner even during the network wait for the first byte.
    wait = spinner("Waiting for model...")
    wait.start()

    try:
        resp = requests.post(CHAT_ENDPOINT, json=payload, stream=True, timeout=120)
        resp.raise_for_status()

        text_parts: list[str] = []
        tool_calls: list | None = None
        started = False  # becomes True once the first visible output arrives

        for raw in resp.iter_lines():
            if not raw:
                continue
            chunk = json.loads(raw)
            msg = chunk.get("message", {})

            if msg.get("content"):
                if not started:
                    wait.stop()  # first token - stop spinner, stream normally
                    started = True
                text_parts.append(msg["content"])
                console.print(msg["content"], end="", markup=False, highlight=False)

            if msg.get("tool_calls"):
                if not started:
                    wait.stop()
                    started = True
                tool_calls = msg["tool_calls"]

            if chunk.get("done"):
                break

        # Safety: if the loop ends without any output (e.g. only 'done'),
        # make sure we don't leave the spinner running.
        if not started:
            wait.stop()

        return "".join(text_parts), tool_calls

    except Exception:
        # Guarantee the spinner never stays spinning on an error path.
        wait.stop()
        raise

# ---------------------------- one turn ----------------------------
def turn(user_prompt: str) -> None:
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    # first call - model may answer directly *or* request a tool
    _, tool_calls = ollama_stream(messages)

    # handle (possibly multiple) tool calls, loop until model gives text
    while tool_calls:
        for tc in tool_calls:
            fn = tc["function"]
            name, args = fn["name"], fn["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            console.print(f"\n[bold magenta]>> {name}[/]  {json.dumps(args)}")

            # Show a spinner while the (possibly slow) tool runs.
            with spinner("Running tool...") as run:
                result = execute_tool(name, args)

            preview = result[:300] + " ..." if len(result) > 300 else result
            console.print(f"[green]  -> {preview}[/]\n")

            # append the assistant's tool-call message + the tool result
            messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
            messages.append({"role": "tool", "name": name, "content": result})

        # ask the model to continue with the real answer (spinner shown again)
        _, tool_calls = ollama_stream(messages)

    console.print()  # trailing newline

# ---------------------------- main loop ----------------------------
def main() -> None:
    # liveness check (with a spinner while we probe the server)
    with spinner("Checking Ollama..."):
        try:
            requests.get(f"{OLLAMA}/api/tags", timeout=2)
        except requests.exceptions.ConnectionError:
            console.print(f"[bold red]Ollama not reachable at {OLLAMA}[/]")
            sys.exit(1)

    console.print(
        Panel(
            f"[bold]LLM Mini CLI[/]  -  model [cyan]{MODEL}[/]  -  tools [magenta]read_file, write_file[/]",
            subtitle="type a prompt, 'q' to quit",
            box=ROUNDED,
            border_style="blue",
        )
    )

    while True:
        try:
            prompt = console.input("\n[bold cyan]>>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ("q", "quit", "exit"):
            break

        try:
            turn(prompt)
        except requests.exceptions.ConnectionError:
            console.print("[red]Lost connection to Ollama.[/]")
        except json.JSONDecodeError:
            console.print("[red]Malformed response from Ollama.[/]")

    console.print("[dim]bye[/]")


if __name__ == "__main__":
    main()
