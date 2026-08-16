# llm-mini-cli

A minimal, dependency-light CLI harness that gives local LLMs (via Ollama) the ability to interact with your local file system through Tool Calling (Function Calling).

---

## Overview

Running local LLMs in a terminal is useful, but their utility is limited when they cannot read or write local files. `llm-mini-cli` solves this by implementing a lightweight execution loop around Ollama's Chat API, providing the model with `read_file` and `write_file` tools.

Key features:
* **Zero Heavy Frameworks**: Built using standard Python with only `requests` and `rich`—no LangChain or LlamaIndex required.
* **Tool Calling Execution Loop**: Automatically detects when the LLM requests a tool call, executes it locally, feeds the result back, and streams the final answer.
* **Real-time Terminal Output**: Live response streaming with clean formatting using `rich`.

---

## Prerequisites

* **Python 3.10+**
* **Ollama** running locally (or accessible over your network)

Ensure you have pulled your preferred model in Ollama before running:

```bash
ollama pull llama3
